#!/usr/bin/env python3
"""Browser demo gateway for the InnoLive gRPC ProcessVideo service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, aclosing, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from grpc_client import VideoClientError, VideoFrame, VideoProcessorClient
from service.protocol import (
    HEADER,
    MAGIC,
    MAX_JPEG_BYTES,
    VERSION,
    decode_request,
    encode_response,
    recover_sequence,
)
from service.recognition import validate_session_id

LOGGER = logging.getLogger("innolive.demo")
ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
PROFILE = "B1-640-Q90-W5"
GRPC_SERVICE = "AiProcessor"
REQUEST_WINDOW = 5
TARGET_FPS = 30
IMAGE_SIZE = 640
JPEG_QUALITY = 90
RECOVERABLE_GRPC_ERRORS = frozenset({"DECODE_FAILED", "INVALID_BATCH_SIZE"})
STAT_FIELDS = (
    "detections",
    "raw_detections",
    "continuation_candidates",
    "detector_backed_tracks",
    "low_confidence_continuations",
    "held_tracks",
    "tracks",
    "tracker_frame",
    "adaface_calls",
    "adaface_queue_overflow",
    "whitelisted_tracks",
)
_END_OF_FRAMES = object()


@dataclass(frozen=True, slots=True)
class ServerSettings:
    session_id: str
    grpc_target: str = "127.0.0.1:50051"
    grpc_connect_timeout_seconds: float = 10.0
    grpc_health_timeout_seconds: float = 2.0
    max_jpeg_bytes: int = MAX_JPEG_BYTES
    max_streams: int = 4

    def __post_init__(self) -> None:
        target = self.grpc_target.strip()
        validate_session_id(self.session_id)
        if not target:
            raise ValueError("grpc_target must not be empty")
        if not math.isfinite(self.grpc_connect_timeout_seconds):
            raise ValueError("grpc_connect_timeout_seconds must be finite")
        if self.grpc_connect_timeout_seconds <= 0:
            raise ValueError("grpc_connect_timeout_seconds must be positive")
        if not math.isfinite(self.grpc_health_timeout_seconds):
            raise ValueError("grpc_health_timeout_seconds must be finite")
        if self.grpc_health_timeout_seconds <= 0:
            raise ValueError("grpc_health_timeout_seconds must be positive")
        if not 1 <= self.max_jpeg_bytes <= MAX_JPEG_BYTES:
            raise ValueError(f"max_jpeg_bytes must be in 1..{MAX_JPEG_BYTES}")
        if self.max_streams < 1:
            raise ValueError("max_streams must be at least one")
        object.__setattr__(self, "grpc_target", target)

    @property
    def max_message_bytes(self) -> int:
        return HEADER.size + self.max_jpeg_bytes


class ConnectionLimiter:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.active = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> bool:
        async with self._lock:
            if self.active >= self.capacity:
                return False
            self.active += 1
            return True

    async def leave(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)


class ServerMetrics:
    def __init__(self) -> None:
        self.accepted = 0
        self.results = 0
        self.errors: Counter[str] = Counter()
        self.jpeg_bytes = 0
        self.metadata_bytes = 0
        self.detections = 0
        self.tracks = 0
        self._latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=300))

    def record_result(self, payload: dict[str, Any], response_bytes: int) -> None:
        self.results += 1
        self.metadata_bytes += response_bytes
        stats = payload.get("stats", {})
        self.detections += int(stats.get("detections", 0))
        self.tracks += int(stats.get("tracks", 0))
        for stage, value in payload.get("timing_ms", {}).items():
            if isinstance(value, (int, float)):
                self._latency[stage].append(float(value))

    def record_error(self, stage: str, response_bytes: int) -> None:
        self.errors[stage] += 1
        self.metadata_bytes += response_bytes

    def snapshot(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "terminal_results": self.results,
            "terminal_errors": sum(self.errors.values()),
            "errors_by_stage": dict(self.errors),
            "jpeg_bytes": self.jpeg_bytes,
            "metadata_bytes": self.metadata_bytes,
            "detections": self.detections,
            "tracks": self.tracks,
            "latency_ms": {stage: _percentiles(values) for stage, values in self._latency.items()},
        }


class GrpcDemoGateway:
    def __init__(
        self,
        settings: ServerSettings,
        client_factory: Callable[..., Any],
    ) -> None:
        self.settings = settings
        self.client_factory = client_factory
        self.limiter = ConnectionLimiter(settings.max_streams)
        self.metrics = ServerMetrics()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        stack = AsyncExitStack()
        app.state.grpc_client = None
        app.state.ready = False
        app.state.startup_error = None
        try:
            client = self.client_factory(
                self.settings.grpc_target,
                connect_timeout=self.settings.grpc_connect_timeout_seconds,
            )
            client = await stack.enter_async_context(client)
            if not await client.is_serving(
                GRPC_SERVICE,
                timeout=self.settings.grpc_health_timeout_seconds,
            ):
                raise RuntimeError(f"gRPC health service reports {GRPC_SERVICE} NOT_SERVING")
            app.state.grpc_client = client
            app.state.ready = True
            LOGGER.info("gRPC demo gateway connected to %s", self.settings.grpc_target)
        except Exception as error:
            app.state.startup_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("gRPC demo gateway startup failed")
        try:
            yield
        finally:
            app.state.ready = False
            app.state.grpc_client = None
            await stack.aclose()

    async def index(self) -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    async def health(self, request: Request) -> JSONResponse:
        serving, health_error = await self._grpc_serving(request.app)
        payload = self._health_payload(serving)
        if request.app.state.startup_error:
            payload["startup_error"] = request.app.state.startup_error
        if health_error:
            payload["grpc"]["error"] = health_error
        return JSONResponse(payload, status_code=200 if serving else 503)

    async def readiness(self, request: Request) -> JSONResponse:
        serving, health_error = await self._grpc_serving(request.app)
        payload = {
            "ready": serving,
            "profile": PROFILE,
            "grpc_target": self.settings.grpc_target,
        }
        if health_error:
            payload["error"] = health_error
        return JSONResponse(payload, status_code=200 if serving else 503)

    async def stream(self, websocket: WebSocket) -> None:
        serving, _ = await self._grpc_serving(websocket.app)
        client = getattr(websocket.app.state, "grpc_client", None)
        if not serving or client is None:
            await self._reject(websocket, "gRPC backend is not serving")
            return
        if not await self.limiter.enter():
            await self._reject(websocket, "demo stream capacity reached")
            return

        await websocket.accept()
        try:
            await GrpcWebSocketSession(
                websocket,
                client,
                self.settings,
                self.metrics,
            ).run()
        except WebSocketDisconnect:
            return
        except Exception:
            LOGGER.exception("gRPC demo stream failed")
            with suppress(RuntimeError):
                await websocket.close(code=1011, reason="gRPC demo stream failed")
        finally:
            await self.limiter.leave()

    async def _grpc_serving(self, app: FastAPI) -> tuple[bool, str | None]:
        client = getattr(app.state, "grpc_client", None)
        if not getattr(app.state, "ready", False) or client is None:
            return False, None
        try:
            serving = await client.is_serving(
                GRPC_SERVICE,
                timeout=self.settings.grpc_health_timeout_seconds,
            )
            return bool(serving), None
        except Exception as error:
            return False, f"{type(error).__name__}: {error}"

    def _health_payload(self, serving: bool) -> dict[str, Any]:
        return {
            "status": "ok" if serving else "not_ready",
            "profile": PROFILE,
            "protocol": {"name": "ILF1", "version": VERSION},
            "transport_path": ["browser-websocket", "grpc-bidi-ProcessVideo"],
            "grpc": {
                "target": self.settings.grpc_target,
                "service": GRPC_SERVICE,
                "serving": serving,
            },
            "serving_profile": {
                "engine_batch": 1,
                "image_size": IMAGE_SIZE,
                "max_long_edge": IMAGE_SIZE,
                "jpeg_quality": JPEG_QUALITY,
                "client_window": REQUEST_WINDOW,
                "target_fps": TARGET_FPS,
                "max_streams": self.settings.max_streams,
            },
            "active_streams": self.limiter.active,
            "metrics": self.metrics.snapshot(),
        }

    @staticmethod
    async def _reject(websocket: WebSocket, reason: str) -> None:
        await websocket.accept()
        await websocket.close(code=1013, reason=reason)


class GrpcWebSocketSession:
    def __init__(
        self,
        websocket: WebSocket,
        client: VideoProcessorClient,
        settings: ServerSettings,
        metrics: ServerMetrics,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.settings = settings
        self.metrics = metrics
        self.frames: asyncio.Queue[VideoFrame | object] = asyncio.Queue()
        self.slots = asyncio.Semaphore(REQUEST_WINDOW)
        self.outstanding: deque[int] = deque()
        self.sent_at: dict[int, float] = {}
        self.last_sequence = -1
        self.stop_receiving = False
        self.send_lock = asyncio.Lock()

    async def run(self) -> None:
        receiver = asyncio.create_task(self._receive_frames(), name="demo-websocket-reader")
        try:
            async with aclosing(
                self.client.process_video(
                    self._frame_source(),
                    session_id=self.settings.session_id,
                    window=REQUEST_WINDOW,
                    raise_frame_errors=False,
                )
            ) as results:
                async for result in results:
                    if not await self._forward_result(result.response):
                        return
            await receiver
        except VideoClientError as error:
            await self._fail_grpc_stream(error)
        finally:
            if not receiver.done():
                receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def _receive_frames(self) -> None:
        try:
            while not self.stop_receiving:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                payload = message.get("bytes")
                if payload is None:
                    await self.websocket.close(
                        code=1003,
                        reason="binary ILF1 requests are required",
                    )
                    return
                frame = await self._parse_frame(payload)
                if frame is None:
                    continue
                await self.slots.acquire()
                self.outstanding.append(frame.frame_id)
                self.sent_at[frame.frame_id] = time.perf_counter()
                self.frames.put_nowait(frame)
        finally:
            self.frames.put_nowait(_END_OF_FRAMES)

    async def _parse_frame(self, payload: bytes) -> VideoFrame | None:
        try:
            sequence, jpeg = decode_request(
                payload,
                max_jpeg_bytes=self.settings.max_jpeg_bytes,
            )
        except ValueError as error:
            sequence = recover_sequence(payload)
            if sequence is None:
                self.stop_receiving = True
                await self.websocket.close(code=1002, reason="truncated ILF1 header")
                return None
            if payload[:4] != MAGIC:
                await self._send_error(sequence, "INVALID_HEADER", str(error), "protocol")
                self.stop_receiving = True
                await self.websocket.close(code=1002, reason="invalid ILF1 header")
                return None
            if sequence <= self.last_sequence:
                await self._sequence_error(sequence)
                return None
            self.last_sequence = sequence
            self.metrics.accepted += 1
            await self._send_error(sequence, "INVALID_FRAME", str(error), "boundary")
            return None

        if sequence <= self.last_sequence:
            await self._sequence_error(sequence)
            return None
        self.last_sequence = sequence
        self.metrics.accepted += 1
        self.metrics.jpeg_bytes += len(jpeg)
        return VideoFrame(
            data=jpeg,
            timestamp=time.perf_counter_ns(),
            frame_id=sequence,
        )

    async def _frame_source(self) -> AsyncIterator[VideoFrame]:
        while True:
            item = await self.frames.get()
            if item is _END_OF_FRAMES:
                return
            if not isinstance(item, VideoFrame):
                raise TypeError("demo frame queue contained an invalid item")
            yield item

    async def _forward_result(self, response: Any) -> bool:
        sequence = int(response.frame_id)
        if not self.outstanding or self.outstanding[0] != sequence:
            raise VideoClientError(f"unexpected gRPC terminal for frame {sequence}")
        self.outstanding.popleft()
        self.slots.release()
        started = self.sent_at.pop(sequence)
        round_trip_ms = (time.perf_counter() - started) * 1_000

        if str(response.status_message).casefold() != "success":
            code = str(response.error_code).strip() or "GRPC_FRAME_FAILED"
            message = str(response.error_message).strip() or str(response.status_message)
            await self._send_error(sequence, code, message, "grpc")
            if code not in RECOVERABLE_GRPC_ERRORS:
                self.stop_receiving = True
                await self.websocket.close(code=1011, reason="gRPC frame failed")
                return False
            return True

        payload = self._result_payload(response, round_trip_ms)
        encoded = encode_response(payload)
        await self._send_text(encoded)
        self.metrics.record_result(payload, len(encoded.encode("utf-8")))
        return True

    @staticmethod
    def _result_payload(response: Any, round_trip_ms: float) -> dict[str, Any]:
        timing = response.timing
        timing_ms = {
            "queue": round(float(timing.queue_ms), 2),
            "decode": round(float(timing.decode_ms), 2),
            "inference": round(float(timing.inference_ms), 2),
            "tracking": round(float(timing.tracking_ms), 2),
            "serialize": round(float(timing.serialize_ms), 2),
            "runtime_total": round(float(timing.runtime_total_ms), 2),
            "server_total": round(float(timing.server_total_ms), 2),
            "grpc_round_trip": round(round_trip_ms, 2),
        }
        stats = response.stats
        return {
            "type": "result",
            "seq": int(response.frame_id),
            "width": int(response.width),
            "height": int(response.height),
            "objects": [_face_object(face) for face in response.faces],
            "timing_ms": timing_ms,
            "stats": {field: int(getattr(stats, field)) for field in STAT_FIELDS},
            "transport": "grpc",
        }

    async def _sequence_error(self, sequence: int) -> None:
        await self._send_error(
            sequence,
            "NON_MONOTONIC_SEQUENCE",
            "sequence must be strictly increasing",
            "sequence",
        )
        self.stop_receiving = True
        await self.websocket.close(code=1002, reason="sequence regression")

    async def _fail_grpc_stream(self, error: Exception) -> None:
        sequence = self.outstanding[0] if self.outstanding else max(self.last_sequence, 0)
        with suppress(RuntimeError, WebSocketDisconnect):
            await self._send_error(
                sequence,
                "GRPC_STREAM_FAILED",
                str(error),
                "grpc",
            )
            await self.websocket.close(code=1011, reason="gRPC stream failed")

    async def _send_error(self, sequence: int, code: str, message: str, stage: str) -> None:
        encoded = encode_response(
            {
                "type": "error",
                "seq": sequence,
                "code": code,
                "message": message,
            }
        )
        self.metrics.record_error(stage, len(encoded.encode("utf-8")))
        await self._send_text(encoded)

    async def _send_text(self, payload: str) -> None:
        async with self.send_lock:
            await self.websocket.send_text(payload)


def _face_object(face: Any) -> dict[str, Any]:
    item = {
        "class_id": 0,
        "class_name": str(face.class_name) or "face",
        "confidence": round(float(face.confidence), 4),
        "bbox": [
            round(float(face.bbox.x1), 1),
            round(float(face.bbox.y1), 1),
            round(float(face.bbox.x2), 1),
            round(float(face.bbox.y2), 1),
        ],
        "mask_polygon": [
            [round(float(point.x), 1), round(float(point.y), 1)] for point in face.polygon
        ],
        "mask_area_px": round(float(face.mask_area_px), 1),
        "source": str(face.source),
        "held": bool(face.held),
        "hold_frames": int(face.hold_frames),
        "whitelisted": bool(face.whitelisted),
    }
    if face.HasField("track_id"):
        item["track_id"] = int(face.track_id)
    return item


def create_app(
    settings: ServerSettings,
    *,
    client_factory: Callable[..., Any] = VideoProcessorClient,
) -> FastAPI:
    gateway = GrpcDemoGateway(settings, client_factory)
    app = FastAPI(title="InnoLive gRPC browser demo", lifespan=gateway.lifespan)
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")
    app.add_api_route("/", gateway.index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/healthz", gateway.health, methods=["GET"])
    app.add_api_route("/readyz", gateway.readiness, methods=["GET"])
    app.add_api_websocket_route("/ws", gateway.stream)
    app.state.gateway = gateway
    return app


def _percentiles(values: deque[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    samples = sorted(values)
    return {
        "p50": round(samples[math.ceil(len(samples) * 0.50) - 1], 2),
        "p95": round(samples[math.ceil(len(samples) * 0.95) - 1], 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--grpc-target", default="127.0.0.1:50051")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--grpc-connect-timeout", type=float, default=10.0)
    parser.add_argument("--grpc-health-timeout", type=float, default=2.0)
    parser.add_argument("--max-streams", type=int, default=4)
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be in 1..65535")
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        raise SystemExit("--ssl-certfile and --ssl-keyfile must be supplied together")
    settings = ServerSettings(
        session_id=args.session_id,
        grpc_target=args.grpc_target,
        grpc_connect_timeout_seconds=args.grpc_connect_timeout,
        grpc_health_timeout_seconds=args.grpc_health_timeout,
        max_streams=args.max_streams,
    )
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        ws_max_size=settings.max_message_bytes,
        ws_max_queue=REQUEST_WINDOW,
        ssl_certfile=str(args.ssl_certfile.resolve()) if args.ssl_certfile else None,
        ssl_keyfile=str(args.ssl_keyfile.resolve()) if args.ssl_keyfile else None,
    )


if __name__ == "__main__":
    main()
