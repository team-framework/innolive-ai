#!/usr/bin/env python3
"""InnoLive ILF1 metadata-only WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from service.protocol import (
    HEADER,
    MAGIC,
    MAX_JPEG_BYTES,
    VERSION,
    decode_request,
    encode_response,
    recover_sequence,
)
from service.runtime import (
    IMAGE_SIZE,
    InferenceFailure,
    RuntimeConfig,
    RuntimeManager,
    TrackingFailure,
)
from service.tracking import StreamTracker


LOGGER = logging.getLogger("innolive.server")
ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_ENGINE = ROOT / "models" / "best_b1.engine"
DEFAULT_TRACKER = ROOT / "config" / "botsort.yaml"
PROFILE = "B1-640-Q90-W5"
REQUEST_WINDOW = 5
TARGET_FPS = 30
JPEG_QUALITY = 90
MAX_PIXELS = IMAGE_SIZE * IMAGE_SIZE
MAX_STREAMS = 1


@dataclass(frozen=True)
class ServerSettings:
    runtime: RuntimeConfig
    tracker_config: Path = DEFAULT_TRACKER
    max_jpeg_bytes: int = MAX_JPEG_BYTES
    inference_timeout_seconds: float = 1.5

    def __post_init__(self) -> None:
        if self.max_jpeg_bytes < 1:
            raise ValueError("max_jpeg_bytes must be positive")
        if self.inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be positive")

    @property
    def max_message_bytes(self) -> int:
        return HEADER.size + self.max_jpeg_bytes


class ConnectionLimiter:
    def __init__(self) -> None:
        self.active = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> bool:
        async with self._lock:
            if self.active >= MAX_STREAMS:
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
        self.low_confidence_continuations = 0
        self.held_masks = 0
        self._latency: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=300)
        )

    def record_result(self, result: dict[str, Any], response_bytes: int) -> None:
        self.results += 1
        self.metadata_bytes += response_bytes
        self.detections += int(result.get("detections", 0))
        self.tracks += int(result.get("tracks", 0))
        self.low_confidence_continuations += int(
            result.get("low_confidence_continuations", 0)
        )
        self.held_masks += int(result.get("held_tracks", 0))
        for stage, value in result.get("timing_ms", {}).items():
            if isinstance(value, (int, float)):
                self._latency[stage].append(float(value))

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
            "low_confidence_continuations": self.low_confidence_continuations,
            "held_masks": self.held_masks,
            "latency_ms": {
                stage: _percentiles(values) for stage, values in self._latency.items()
            },
        }


def decode_jpeg(jpeg: bytes, settings: ServerSettings) -> np.ndarray:
    if len(jpeg) > settings.max_jpeg_bytes:
        raise ValueError(f"frame exceeds {settings.max_jpeg_bytes} byte limit")
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("frame could not be decoded as JPEG")
    height, width = image.shape[:2]
    if width < 32 or height < 32:
        raise ValueError("decoded dimensions must be at least 32x32")
    if max(width, height) > IMAGE_SIZE:
        raise ValueError(f"decoded frame exceeds long-edge {IMAGE_SIZE} limit")
    if width * height > MAX_PIXELS:
        raise ValueError(f"decoded frame exceeds {MAX_PIXELS} pixel limit")
    return image


def create_app(
    settings: ServerSettings,
    *,
    runtime_factory: Callable[[RuntimeConfig], Any] = RuntimeManager,
    tracker_factory: Callable[..., Any] = StreamTracker,
) -> FastAPI:
    limiter = ConnectionLimiter()
    metrics = ServerMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = None
        app.state.ready = False
        app.state.startup_error = None
        try:
            runtime = await asyncio.to_thread(runtime_factory, settings.runtime)
            app.state.runtime = runtime
            probe = await asyncio.to_thread(
                tracker_factory,
                settings.tracker_config,
                settings.runtime.device,
            )
            probe.reset()
            app.state.ready = bool(runtime.ready)
            if not app.state.ready:
                raise RuntimeError("runtime warm-up did not reach ready state")
        except Exception as error:
            app.state.startup_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("startup validation failed")
            runtime = getattr(app.state, "runtime", None)
            if runtime is not None:
                await asyncio.to_thread(runtime.close)
                app.state.runtime = None
        try:
            yield
        finally:
            app.state.ready = False
            runtime = getattr(app.state, "runtime", None)
            if runtime is not None:
                await asyncio.to_thread(runtime.close)

    app = FastAPI(title="InnoLive ILF1 face metadata server", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/healthz")
    async def health():
        runtime = getattr(app.state, "runtime", None)
        ready = bool(getattr(app.state, "ready", False) and runtime and runtime.ready)
        payload = {
            "status": "ok" if ready else "not_ready",
            "profile": PROFILE,
            "protocol": {"name": "ILF1", "version": VERSION},
            "serving_profile": {
                "engine_batch": 1,
                "image_size": IMAGE_SIZE,
                "max_long_edge": IMAGE_SIZE,
                "jpeg_quality": JPEG_QUALITY,
                "client_window": REQUEST_WINDOW,
                "target_fps": TARGET_FPS,
                "max_streams": MAX_STREAMS,
            },
            "active_streams": limiter.active,
            "metrics": metrics.snapshot(),
        }
        if runtime is not None:
            payload["runtime"] = runtime.health()
        if app.state.startup_error:
            payload["startup_error"] = app.state.startup_error
        return JSONResponse(payload, status_code=200 if ready else 503)

    @app.get("/readyz")
    async def readiness():
        runtime = getattr(app.state, "runtime", None)
        ready = bool(getattr(app.state, "ready", False) and runtime and runtime.ready)
        return JSONResponse(
            {"ready": ready, "profile": PROFILE},
            status_code=200 if ready else 503,
        )

    @app.websocket("/ws")
    async def stream(websocket: WebSocket):
        runtime = getattr(app.state, "runtime", None)
        if not app.state.ready or runtime is None or not runtime.ready:
            await websocket.accept()
            await websocket.close(code=1013, reason="runtime is not ready")
            return
        if not await limiter.enter():
            await websocket.accept()
            await websocket.close(code=1013, reason="stream capacity reached")
            return

        tracker = None
        pending_inference: asyncio.Task | None = None
        try:
            tracker = tracker_factory(
                settings.tracker_config,
                settings.runtime.device,
            )
            last_sequence = -1
            await websocket.accept()
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is None:
                    await websocket.close(
                        code=1003,
                        reason="binary ILF1 requests are required",
                    )
                    break

                received_at = time.perf_counter()
                try:
                    sequence, jpeg = decode_request(
                        payload,
                        max_jpeg_bytes=settings.max_jpeg_bytes,
                    )
                except ValueError as error:
                    sequence = recover_sequence(payload)
                    if sequence is None:
                        await websocket.close(code=1002, reason="truncated ILF1 header")
                        break
                    if payload[:4] != MAGIC:
                        await _send_error(
                            websocket,
                            metrics,
                            sequence,
                            "INVALID_HEADER",
                            str(error),
                            "protocol",
                        )
                        await websocket.close(code=1002, reason="invalid ILF1 header")
                        break
                    if sequence <= last_sequence:
                        await _send_error(
                            websocket,
                            metrics,
                            sequence,
                            "NON_MONOTONIC_SEQUENCE",
                            "sequence must be strictly increasing",
                            "sequence",
                        )
                        await websocket.close(code=1002, reason="sequence regression")
                        break
                    last_sequence = sequence
                    metrics.accepted += 1
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "INVALID_FRAME",
                        str(error),
                        "boundary",
                    )
                    continue

                if sequence <= last_sequence:
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "NON_MONOTONIC_SEQUENCE",
                        "sequence must be strictly increasing",
                        "sequence",
                    )
                    await websocket.close(code=1002, reason="sequence regression")
                    break
                last_sequence = sequence
                metrics.accepted += 1
                metrics.jpeg_bytes += len(jpeg)

                try:
                    image = await asyncio.to_thread(decode_jpeg, jpeg, settings)
                    decoded_at = time.perf_counter()
                except ValueError as error:
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "DECODE_FAILED",
                        str(error),
                        "decode",
                    )
                    continue

                pending_inference = asyncio.create_task(runtime.infer(image, tracker))
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(pending_inference),
                        timeout=settings.inference_timeout_seconds,
                    )
                except TimeoutError:
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "INFERENCE_TIMEOUT",
                        "frame inference exceeded the server timeout",
                        "inference",
                    )
                    try:
                        await pending_inference
                    except Exception:
                        LOGGER.exception("timed-out inference later failed")
                    await websocket.close(code=1011, reason="inference timeout")
                    break
                except TrackingFailure:
                    LOGGER.exception("frame %d tracking failed", sequence)
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "TRACKING_FAILED",
                        "frame tracking failed",
                        "tracking",
                    )
                    await websocket.close(code=1011, reason="tracking failed")
                    break
                except InferenceFailure:
                    LOGGER.exception("frame %d inference failed", sequence)
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "INFERENCE_FAILED",
                        "frame inference failed",
                        "inference",
                    )
                    await websocket.close(code=1011, reason="inference failed")
                    break
                except Exception:
                    LOGGER.exception("frame %d processing failed", sequence)
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "PROCESSING_FAILED",
                        "frame processing failed",
                        "processing",
                    )
                    await websocket.close(code=1011, reason="inference failed")
                    break
                finally:
                    pending_inference = None

                serialize_started = time.perf_counter()
                timing = dict(result.get("timing_ms", {}))
                timing["decode"] = round((decoded_at - received_at) * 1000, 2)
                response = {
                    "type": "result",
                    "seq": sequence,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "objects": result.get("objects", []),
                    "timing_ms": timing,
                    "stats": {
                        key: result[key]
                        for key in (
                            "detections",
                            "raw_detections",
                            "continuation_candidates",
                            "detector_backed_tracks",
                            "low_confidence_continuations",
                            "held_tracks",
                            "tracks",
                            "tracker_frame",
                        )
                        if key in result
                    },
                }
                try:
                    provisional = encode_response(response)
                    timing["serialize"] = round(
                        float(timing.get("serialize", 0))
                        + (time.perf_counter() - serialize_started) * 1000,
                        2,
                    )
                    timing["server_total"] = round(
                        (time.perf_counter() - received_at) * 1000,
                        2,
                    )
                    encoded = encode_response(response)
                    del provisional
                except (TypeError, ValueError):
                    LOGGER.exception("frame %d metadata serialization failed", sequence)
                    await _send_error(
                        websocket,
                        metrics,
                        sequence,
                        "SERIALIZATION_FAILED",
                        "result metadata exceeded the serialization contract",
                        "serialize",
                    )
                    await websocket.close(code=1011, reason="serialization failed")
                    break

                await websocket.send_text(encoded)
                result["timing_ms"] = timing
                metrics.record_result(result, len(encoded.encode("utf-8")))
        except WebSocketDisconnect:
            pass
        finally:
            if pending_inference is not None:
                try:
                    await pending_inference
                except Exception:
                    pass
            if tracker is not None:
                tracker.reset()
            await limiter.leave()

    return app


async def _send_error(
    websocket: WebSocket,
    metrics: ServerMetrics,
    sequence: int,
    code: str,
    message: str,
    stage: str,
) -> None:
    encoded = encode_response(
        {
            "type": "error",
            "seq": sequence,
            "code": code,
            "message": message,
        }
    )
    metrics.errors[stage] += 1
    metrics.metadata_bytes += len(encoded.encode("utf-8"))
    await websocket.send_text(encoded)


def _percentiles(values: deque[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    samples = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(samples, 50)), 2),
        "p95": round(float(np.percentile(samples, 95)), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER)
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
        runtime=RuntimeConfig(args.engine.expanduser().resolve(), args.device),
        tracker_config=args.tracker_config.expanduser().resolve(),
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
