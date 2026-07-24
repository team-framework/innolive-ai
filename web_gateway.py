from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service.grpc_config import channel_options
from service.protocol import (
    MAX_PACKET_BYTES,
    EncodedFrame,
    FrameResult,
    decode_batch,
    encode_result,
)

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "web"
GENERATED_DIR = PROJECT_ROOT / "__generated__"
sys.path.insert(0, str(GENERATED_DIR))

ai_processor_pb2_grpc = importlib.import_module("ai_processor_pb2_grpc")
messages = importlib.import_module("ai_processor_pb2")


@dataclass(slots=True)
class _GatewayFrame:
    frame: EncodedFrame
    sequence: int
    decoded_at: float
    protocol_decode_ms: float = 0.0
    ingress_queue_ms: float = 0.0
    grpc_write_ms: float = 0.0
    grpc_write_done_at: float = 0.0
    grpc_response_at: float = 0.0
    response_queue_ms: float = 0.0

    def timing(self, server_processing_ms: float | None = None) -> dict[str, float]:
        grpc_wait_ms = max(
            0.0,
            self.grpc_response_at
            - (self.grpc_write_done_at or self.decoded_at),
        ) * 1_000
        timing = {
            "protocolDecodeMs": round(self.protocol_decode_ms, 2),
            "ingressQueueMs": round(self.ingress_queue_ms, 2),
            "grpcWriteMs": round(self.grpc_write_ms, 2),
            "grpcWaitMs": round(grpc_wait_ms, 2),
            "responseQueueMs": round(self.response_queue_ms, 2),
        }
        if server_processing_ms is not None:
            timing["grpcResidualMs"] = round(
                max(0.0, grpc_wait_ms - server_processing_ms), 2
            )
        return timing


@asynccontextmanager
async def lifespan(app: FastAPI):
    target = os.getenv("GRPC_TARGET", "127.0.0.1:50051")
    channel = grpc.aio.insecure_channel(target, options=channel_options())
    try:
        await asyncio.wait_for(
            channel.channel_ready(),
            timeout=float(os.getenv("GRPC_CONNECT_TIMEOUT", "10")),
        )
    except TimeoutError as error:
        await channel.close()
        raise RuntimeError(f"gRPC server is unavailable at {target}") from error

    app.state.grpc_target = target
    app.state.grpc_channel = channel
    app.state.grpc_stub = ai_processor_pb2_grpc.AiProcessorStub(channel)
    yield
    await channel.close(grace=2)


app = FastAPI(title="InnoLive gRPC WebSocket gateway", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    state = app.state.grpc_channel.get_state(try_to_connect=True)
    return {
        "status": "ok" if state is grpc.ChannelConnectivity.READY else "degraded",
        "grpcTarget": app.state.grpc_target,
        "grpcState": state.name,
    }


@app.websocket("/ws")
async def process_stream(websocket: WebSocket):
    await websocket.accept()
    call = app.state.grpc_stub.ProcessVideo()
    incoming: asyncio.Queue[list[_GatewayFrame]] = asyncio.Queue(maxsize=2)
    outgoing: asyncio.Queue[tuple[FrameResult, _GatewayFrame, float]] = asyncio.Queue(
        maxsize=2
    )
    sources: dict[int, _GatewayFrame] = {}
    next_sequence = 0

    async def receive_websocket() -> None:
        nonlocal next_sequence
        while True:
            packet = await websocket.receive_bytes()
            decode_started = time.perf_counter()
            frames = decode_batch(packet)
            decode_ms = (time.perf_counter() - decode_started) * 1_000
            traced = []
            decoded_at = time.perf_counter()
            for frame in frames:
                traced_frame = _GatewayFrame(
                    frame=frame,
                    sequence=next_sequence,
                    decoded_at=decoded_at,
                    protocol_decode_ms=decode_ms,
                )
                sources[frame.frame_id] = traced_frame
                traced.append(traced_frame)
                next_sequence += 1
            await incoming.put(traced)

    async def write_grpc() -> None:
        while True:
            traced_frames = await incoming.get()
            batch_size = len(traced_frames)
            for traced in traced_frames:
                write_started = time.perf_counter()
                await call.write(
                    messages.VideoChunk(
                        data=traced.frame.jpeg,
                        timestamp=int(traced.frame.captured_at),
                        frame_id=traced.frame.frame_id,
                        batch_size=batch_size,
                    )
                )
                write_done = time.perf_counter()
                traced.ingress_queue_ms = (write_started - traced.decoded_at) * 1_000
                traced.grpc_write_done_at = write_done
                traced.grpc_write_ms = (write_done - write_started) * 1_000

    async def read_grpc() -> None:
        pending: dict[int, tuple[FrameResult, _GatewayFrame, float]] = {}
        next_to_send = 0
        while True:
            response = await read_response(call)
            traced = sources.pop(response.frame_id, None)
            if traced is None:
                continue
            traced.grpc_response_at = time.perf_counter()
            result = to_frame_result(
                traced.frame,
                response,
                traced.timing(response.processing_ms),
            )
            pending[traced.sequence] = (
                result,
                traced,
                response.processing_ms,
            )
            while next_to_send in pending:
                await outgoing.put(pending.pop(next_to_send))
                next_to_send += 1

    async def send_websocket() -> None:
        while True:
            result, traced, processing_ms = await outgoing.get()
            traced.response_queue_ms = (
                time.perf_counter() - traced.grpc_response_at
            ) * 1_000
            result.timing["gateway"] = traced.timing(processing_ms)
            packet = encode_result(result, processing_ms)
            await websocket.send_bytes(packet)

    tasks = {
        asyncio.create_task(receive_websocket()),
        asyncio.create_task(write_grpc()),
        asyncio.create_task(read_grpc()),
        asyncio.create_task(send_websocket()),
    }
    try:
        done, _ = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            exception = task.exception()
            if exception is not None:
                raise exception
    except WebSocketDisconnect:
        pass
    except grpc.aio.AioRpcError as error:
        await websocket.send_json({"error": error.details() or error.code().name})
        await websocket.close(code=1011)
    except RuntimeError as error:
        await websocket.send_json({"error": str(error)})
        await websocket.close(code=1011)
    except (KeyError, TypeError, ValueError) as error:
        await websocket.send_json({"error": str(error)})
        await websocket.close(code=1003)
    finally:
        call.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def read_response(call):
    response = await call.read()
    if response is grpc.aio.EOF:
        raise RuntimeError("gRPC stream ended before the batch was complete")
    if response.status_message != "success":
        raise RuntimeError(response.status_message)
    return response


def to_frame_result(
    frame: EncodedFrame,
    response,
    gateway_timing: dict[str, float] | None = None,
) -> FrameResult:
    timing = {
        "queueMs": round(response.timing.queue_ms, 2),
        "decodeMs": round(response.timing.decode_ms, 2),
        "inferenceMs": round(response.timing.inference_ms, 2),
        "trackingMs": round(response.timing.tracking_ms, 2),
        "blurEncodeMs": round(response.timing.blur_encode_ms, 2),
        "inferenceBatchSize": response.timing.inference_batch_size,
    }
    if gateway_timing is not None:
        timing["gateway"] = gateway_timing
    return FrameResult(
        frame=frame,
        jpeg=response.data,
        width=response.width,
        height=response.height,
        faces=tuple(
            {
                "bbox": [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2],
                "confidence": round(face.confidence, 4),
                "polygon": [[point.x, point.y] for point in face.polygon],
                "trackId": face.track_id if face.HasField("track_id") else None,
            }
            for face in response.faces
        ),
        timing=timing,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional WebSocket-to-gRPC gateway")
    parser.add_argument("--host", default=os.getenv("HTTP_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=positive_port,
        default=positive_port(os.getenv("HTTP_PORT", "8001")),
    )
    parser.add_argument(
        "--grpc-target",
        default=os.getenv("GRPC_TARGET", "127.0.0.1:50051"),
    )
    return parser.parse_args()


def positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


if __name__ == "__main__":
    import uvicorn

    arguments = parse_args()
    os.environ["GRPC_TARGET"] = arguments.grpc_target
    uvicorn.run(
        "web_gateway:app",
        host=arguments.host,
        port=arguments.port,
        ws_max_queue=2,
        ws_max_size=MAX_PACKET_BYTES,
    )
