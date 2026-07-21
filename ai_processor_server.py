from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

import grpc

from service.face_processor import BATCH_SIZE, Face, FaceProcessorPool, Settings
from service.grpc_config import listen_address, server_options
from service.transform_image import encode_blurred_jpeg

GENERATED_DIR = Path(__file__).resolve().parent / "__generated__"
sys.path.insert(0, str(GENERATED_DIR))

ai_processor_pb2_grpc = importlib.import_module("ai_processor_pb2_grpc")
messages = importlib.import_module("ai_processor_pb2")


class AiProcessorServicer(ai_processor_pb2_grpc.AiProcessorServicer):
    def __init__(
        self,
        processor: FaceProcessorPool,
        encode_workers: int,
        max_stream_inflight: int,
    ):
        if max_stream_inflight < 1:
            raise ValueError("GRPC_STREAM_INFLIGHT must be at least one")
        if encode_workers < 1:
            raise ValueError("encode_workers must be at least one")
        self._processor = processor
        self._encode_pool = ThreadPoolExecutor(
            max_workers=encode_workers,
            thread_name_prefix="jpeg-encode",
        )
        # run_in_executor uses an unbounded work queue.  Without a separate
        # gate, a slow encoder can accumulate every frame accepted by all
        # streams even though inference is bounded.  Keeping at most one
        # encode per worker in flight preserves backpressure and latency.
        self._encode_slots = asyncio.Semaphore(encode_workers)
        self._max_stream_inflight = max_stream_inflight
        self._jpeg_quality = int(os.getenv("AI_JPEG_QUALITY", "85"))

    async def ProcessVideo(self, request_iterator, context) -> AsyncIterator:
        stream = self._processor.open_stream()
        responses: asyncio.Queue[tuple[int, object, float] | None] = asyncio.Queue(
            self._max_stream_inflight
        )
        slots = asyncio.Semaphore(self._max_stream_inflight)
        loop = asyncio.get_running_loop()

        async def complete(
            sequence: int,
            request,
            inference,
            queued_at: float,
        ) -> None:
            try:
                frame = await asyncio.wrap_future(inference)
                await self._encode_slots.acquire()
                encode_started = time.perf_counter()
                try:
                    jpeg = await loop.run_in_executor(
                        self._encode_pool,
                        encode_blurred_jpeg,
                        frame,
                        self._jpeg_quality,
                    )
                    blur_encode_ms = (time.perf_counter() - encode_started) * 1_000
                finally:
                    self._encode_slots.release()
                timing = frame.timing
                response = messages.ProcessedVideoChunk(
                    data=jpeg,
                    status_message="success",
                    timestamp=request.timestamp,
                    frame_id=request.frame_id,
                    width=frame.width,
                    height=frame.height,
                    faces=[self._face_metadata(face) for face in frame.faces],
                    processing_ms=(time.perf_counter() - queued_at) * 1_000,
                    timing=messages.ProcessingTiming(
                        queue_ms=timing.queue_ms,
                        decode_ms=timing.decode_ms,
                        inference_ms=timing.inference_ms,
                        tracking_ms=timing.tracking_ms,
                        blur_encode_ms=blur_encode_ms,
                        inference_batch_size=timing.inference_batch_size,
                    ),
                )
            except Exception as error:
                logging.exception("video frame processing failed")
                response = messages.ProcessedVideoChunk(
                    status_message=f"failed: {error}",
                    timestamp=request.timestamp,
                    frame_id=request.frame_id,
                )

            try:
                await responses.put((sequence, response, time.perf_counter()))
            finally:
                slots.release()

        async def produce() -> None:
            tasks: set[asyncio.Task] = set()
            try:
                sequence = 0
                request_batch = []

                async def schedule(requests) -> None:
                    nonlocal sequence
                    for request in requests:
                        await slots.acquire()
                        queued_at = time.perf_counter()
                        inference = stream.submit(request.data)
                        task = asyncio.create_task(
                            complete(sequence, request, inference, queued_at)
                        )
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
                        sequence += 1

                async for request in request_iterator:
                    expected_size = request.batch_size or 1
                    if not 1 <= expected_size <= BATCH_SIZE:
                        raise ValueError(
                            f"batch_size must be between 1 and {BATCH_SIZE}"
                        )
                    if request_batch and expected_size != request_batch[0].batch_size:
                        raise ValueError("batch_size changed inside a request batch")
                    request_batch.append(request)
                    if len(request_batch) == expected_size:
                        await schedule(request_batch)
                        request_batch = []

                if request_batch:
                    await schedule(request_batch)
                if tasks:
                    await asyncio.gather(*tasks)
                await responses.put(None)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            except Exception:
                logging.exception("gRPC request stream failed")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await responses.put(None)
                raise

        producer = asyncio.create_task(produce())
        pending: dict[int, tuple[object, float]] = {}
        next_sequence = 0
        try:
            while True:
                completed = await responses.get()
                if completed is None:
                    break
                sequence, response, ready_at = completed
                pending[sequence] = (response, ready_at)
                while next_sequence in pending:
                    response, ready_at = pending.pop(next_sequence)
                    response.processing_ms += (
                        time.perf_counter() - ready_at
                    ) * 1_000
                    yield response
                    next_sequence += 1
            await producer
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def AddWhitelist(self, request, context):
        return messages.WhitelistResponse(
            status_message="테스트 성공",
            timestamp=int(time.time()),
        )

    @staticmethod
    def _face_metadata(face: Face):
        x1, y1, x2, y2 = face.bbox
        metadata = messages.FaceMetadata(
            bbox=messages.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=face.confidence,
            polygon=[messages.Point(x=x, y=y) for x, y in face.transport_polygon],
        )
        if face.track_id is not None:
            metadata.track_id = face.track_id
        return metadata

    def close(self) -> None:
        self._encode_pool.shutdown(wait=True, cancel_futures=True)


def positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InnoLive gRPC face anonymizer")
    parser.add_argument("--host", default=os.getenv("GRPC_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=positive_port,
        default=positive_port(os.getenv("GRPC_PORT", "50051")),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=positive_int(
            os.getenv(
                "AI_ENCODE_WORKERS",
                str(min(32, (os.cpu_count() or 4) + 4)),
            )
        ),
        help="number of parallel JPEG encoding workers",
    )
    return parser.parse_args()


async def serve(host: str, port: int, encode_workers: int) -> None:
    processor = FaceProcessorPool(Settings.from_env())
    servicer = AiProcessorServicer(
        processor,
        encode_workers,
        int(os.getenv("GRPC_STREAM_INFLIGHT", "4")),
    )
    server = grpc.aio.server(
        options=server_options(),
        maximum_concurrent_rpcs=int(os.getenv("GRPC_MAX_CONCURRENT_RPCS", "256")),
    )
    ai_processor_pb2_grpc.add_AiProcessorServicer_to_server(servicer, server)
    address = listen_address(host, port)
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"failed to bind gRPC server to {address}")

    await server.start()
    logging.info("gRPC server listening on %s", address)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await asyncio.to_thread(servicer.close)
        await asyncio.to_thread(processor.close)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    arguments = parse_args()
    try:
        asyncio.run(serve(arguments.host, arguments.port, arguments.workers))
    except KeyboardInterrupt:
        pass
