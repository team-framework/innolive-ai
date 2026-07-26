from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import cv2
import grpc
import numpy as np
from grpc_health.v1 import health_pb2, health_pb2_grpc

from ai_processor_server import (
    SERVICE_NAME,
    GrpcServerSettings,
    build_grpc_server,
)
from protos import ai_processor_pb2, ai_processor_pb2_grpc
from service.runtime import InferenceFailure, RuntimeConfig


def _jpeg(width: int = 64, height: int = 36) -> bytes:
    encoded, payload = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError("test JPEG encoding failed")
    return payload.tobytes()


def _request(
    *,
    data: bytes | None = None,
    timestamp: int = 1,
    frame_id: int = 0,
    batch_size: int = 1,
):
    return ai_processor_pb2.VideoChunk(
        data=_jpeg() if data is None else data,
        timestamp=timestamp,
        frame_id=frame_id,
        batch_size=batch_size,
    )


async def _requests(*items: Any) -> AsyncIterator[Any]:
    for item in items:
        yield item


async def _collect(call: Any) -> list[Any]:
    return [response async for response in call]


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout=timeout)


class FakeTracker:
    def __init__(self, runtime: Any):
        self.runtime = runtime
        self.frame_id = 0
        self.reset_called = False
        self.reset_after_inference = False
        self.reset_event = asyncio.Event()

    def reset(self) -> None:
        self.reset_called = True
        settled = getattr(self.runtime, "inference_settled", None)
        self.reset_after_inference = settled is None or settled.is_set()
        self.reset_event.set()


class FakeRuntime:
    ready = True

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], FakeTracker]] = []

    async def infer(
        self,
        image: np.ndarray,
        tracker: FakeTracker,
    ) -> dict[str, Any]:
        self.calls.append((image.shape, tracker))
        tracker.frame_id += 1
        return {
            "objects": [],
            "detections": 0,
            "raw_detections": 0,
            "continuation_candidates": 0,
            "detector_backed_tracks": 0,
            "low_confidence_continuations": 0,
            "held_tracks": 0,
            "tracks": 0,
            "tracker_frame": tracker.frame_id,
            "timing_ms": {
                "queue": 0.1,
                "inference": 1.0,
                "tracking": 0.2,
                "serialize": 0.1,
                "runtime_total": 1.3,
            },
        }


class FailedRuntime(FakeRuntime):
    async def infer(
        self,
        image: np.ndarray,
        tracker: FakeTracker,
    ) -> dict[str, Any]:
        self.calls.append((image.shape, tracker))
        raise InferenceFailure("injected inference failure")


class BlockingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.inference_started = asyncio.Event()
        self.release_inference = asyncio.Event()
        self.inference_settled = asyncio.Event()

    async def infer(
        self,
        image: np.ndarray,
        tracker: FakeTracker,
    ) -> dict[str, Any]:
        self.inference_started.set()
        try:
            await self.release_inference.wait()
            return await super().infer(image, tracker)
        finally:
            self.inference_settled.set()


class LoopbackServer:
    def __init__(
        self,
        runtime: FakeRuntime,
        *,
        inference_timeout_seconds: float = 1.5,
    ):
        self.runtime = runtime
        self.inference_timeout_seconds = inference_timeout_seconds
        self.trackers: list[FakeTracker] = []
        self.bundle = None
        self.channel = None
        self.stub = None
        self.health = None

    def tracker_factory(self, *_args: Any) -> FakeTracker:
        tracker = FakeTracker(self.runtime)
        self.trackers.append(tracker)
        return tracker

    async def __aenter__(self) -> LoopbackServer:
        settings = GrpcServerSettings(
            runtime=RuntimeConfig(engine=Path("unused.engine")),
            tracker_config=Path("unused-botsort.yaml"),
            host="127.0.0.1",
            port=0,
            max_streams=1,
            inference_timeout_seconds=self.inference_timeout_seconds,
            shutdown_grace_seconds=0,
        )
        self.bundle = build_grpc_server(
            settings,
            self.runtime,
            tracker_factory=self.tracker_factory,
        )
        await self.bundle.set_serving(True)
        await self.bundle.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.bundle.bound_port}")
        await asyncio.wait_for(self.channel.channel_ready(), timeout=1.0)
        self.stub = ai_processor_pb2_grpc.AiProcessorStub(self.channel)
        self.health = health_pb2_grpc.HealthStub(self.channel)
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        if isinstance(self.runtime, BlockingRuntime):
            self.runtime.release_inference.set()
        if self.channel is not None:
            await self.channel.close()
        if self.bundle is not None:
            await self.bundle.health_servicer.enter_graceful_shutdown()
            await self.bundle.server.stop(0)
            await self.bundle.servicer.close()


class GrpcLoopbackIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_cannot_silently_share_an_existing_port(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            settings = GrpcServerSettings(
                runtime=RuntimeConfig(engine=Path("unused.engine")),
                tracker_config=Path("unused-botsort.yaml"),
                host="127.0.0.1",
                port=server.bundle.bound_port,
            )
            with self.assertRaises(RuntimeError):
                build_grpc_server(
                    settings,
                    FakeRuntime(),
                    tracker_factory=server.tracker_factory,
                )

    async def test_two_successful_frames_allow_duplicate_zero_frame_id(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            responses = await _collect(
                server.stub.ProcessVideo(
                    _requests(
                        _request(timestamp=11, frame_id=0),
                        _request(timestamp=12, frame_id=0),
                    )
                )
            )

        self.assertEqual(len(responses), 2)
        self.assertEqual(
            [(item.status_message, item.frame_id, item.timestamp) for item in responses],
            [("success", 0, 11), ("success", 0, 12)],
        )
        self.assertTrue(all(item.data == b"" for item in responses))
        self.assertEqual([item.stats.tracker_frame for item in responses], [1, 2])
        self.assertEqual(len(runtime.calls), 2)
        self.assertIs(runtime.calls[0][1], runtime.calls[1][1])

    async def test_invalid_jpeg_returns_error_and_stream_continues(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            responses = await _collect(
                server.stub.ProcessVideo(
                    _requests(
                        _request(data=b"not-a-jpeg", timestamp=21, frame_id=1),
                        _request(timestamp=22, frame_id=2),
                    )
                )
            )

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].status_message, "failed")
        self.assertEqual(responses[0].error_code, "DECODE_FAILED")
        self.assertEqual(responses[0].frame_id, 1)
        self.assertEqual(responses[0].data, b"")
        self.assertEqual(responses[1].status_message, "success")
        self.assertEqual(responses[1].frame_id, 2)
        self.assertEqual(len(runtime.calls), 1)

    async def test_batch_size_greater_than_one_is_rejected(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            responses = await _collect(
                server.stub.ProcessVideo(
                    _requests(_request(batch_size=2, timestamp=31, frame_id=3))
                )
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].status_message, "failed")
        self.assertEqual(responses[0].error_code, "INVALID_BATCH_SIZE")
        self.assertEqual(responses[0].frame_id, 3)
        self.assertEqual(responses[0].data, b"")
        self.assertEqual(runtime.calls, [])

    async def test_inference_failure_returns_terminal_error_and_ends_stream(self):
        runtime = FailedRuntime()
        async with LoopbackServer(runtime) as server:
            responses = await _collect(
                server.stub.ProcessVideo(
                    _requests(
                        _request(timestamp=41, frame_id=4),
                        _request(timestamp=42, frame_id=5),
                    )
                )
            )
            self.assertEqual(server.bundle.servicer.admission.active, 0)
            health = await server.health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
            self.assertEqual(
                health.status,
                health_pb2.HealthCheckResponse.NOT_SERVING,
            )
            rejected = server.stub.ProcessVideo(_requests(_request(timestamp=43, frame_id=6)))
            with self.assertRaises(grpc.aio.AioRpcError) as unavailable:
                await rejected.read()
            self.assertEqual(
                unavailable.exception.code(),
                grpc.StatusCode.UNAVAILABLE,
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].status_message, "failed")
        self.assertEqual(responses[0].error_code, "INFERENCE_FAILED")
        self.assertEqual(responses[0].frame_id, 4)
        self.assertEqual(responses[0].data, b"")
        self.assertEqual(len(runtime.calls), 1)
        self.assertTrue(server.trackers[0].reset_called)

    async def test_capacity_one_does_not_block_health(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            first = server.stub.ProcessVideo()
            await first.write(_request(timestamp=51, frame_id=5))
            first_response = await first.read()
            self.assertEqual(first_response.status_message, "success")
            self.assertEqual(server.bundle.servicer.admission.active, 1)

            health = await server.health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
            self.assertEqual(
                health.status,
                health_pb2.HealthCheckResponse.SERVING,
            )

            second = server.stub.ProcessVideo()
            await second.write(_request(timestamp=52, frame_id=6))
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await second.read()
            self.assertEqual(
                rejected.exception.code(),
                grpc.StatusCode.RESOURCE_EXHAUSTED,
            )

            await first.done_writing()
            self.assertIs(await first.read(), grpc.aio.EOF)
            await _wait_until(lambda: server.bundle.servicer.admission.active == 0)

    async def test_add_whitelist_is_unimplemented(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.AddWhitelist(ai_processor_pb2.FaceData())

        self.assertEqual(
            rejected.exception.code(),
            grpc.StatusCode.UNIMPLEMENTED,
        )

    async def test_cancellation_settles_inference_before_reset_and_release(self):
        runtime = BlockingRuntime()
        async with LoopbackServer(runtime) as server:
            call = server.stub.ProcessVideo()
            await call.write(_request(timestamp=61, frame_id=6))
            await asyncio.wait_for(runtime.inference_started.wait(), timeout=1.0)
            tracker = server.trackers[0]
            self.assertEqual(server.bundle.servicer.admission.active, 1)

            call.cancel()
            self.assertEqual(await call.code(), grpc.StatusCode.CANCELLED)
            await asyncio.sleep(0.05)
            self.assertFalse(runtime.inference_settled.is_set())
            self.assertFalse(tracker.reset_called)
            self.assertEqual(server.bundle.servicer.admission.active, 1)

            runtime.release_inference.set()
            await asyncio.wait_for(tracker.reset_event.wait(), timeout=1.0)
            await _wait_until(lambda: server.bundle.servicer.admission.active == 0)

            self.assertTrue(runtime.inference_settled.is_set())
            self.assertTrue(tracker.reset_called)
            self.assertTrue(tracker.reset_after_inference)

    async def test_timeout_is_terminal_before_gpu_cleanup_finishes(self):
        runtime = BlockingRuntime()
        async with LoopbackServer(
            runtime,
            inference_timeout_seconds=0.02,
        ) as server:
            call = server.stub.ProcessVideo()
            await call.write(_request(timestamp=71, frame_id=7))
            response = await asyncio.wait_for(call.read(), timeout=0.5)

            self.assertEqual(response.status_message, "failed")
            self.assertEqual(response.error_code, "INFERENCE_TIMEOUT")
            self.assertFalse(runtime.inference_settled.is_set())
            self.assertFalse(server.trackers[0].reset_called)
            self.assertEqual(server.bundle.servicer.admission.active, 1)
            health = await server.health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
            self.assertEqual(
                health.status,
                health_pb2.HealthCheckResponse.NOT_SERVING,
            )

            runtime.release_inference.set()
            self.assertIs(await call.read(), grpc.aio.EOF)
            await _wait_until(lambda: server.bundle.servicer.admission.active == 0)
            self.assertTrue(server.trackers[0].reset_after_inference)


if __name__ == "__main__":
    unittest.main()
