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
from service.adaface_model import FaceAlignmentError, FaceCountError, FaceTooSmallError
from service.recognition import SessionRegistry
from service.runtime import InferenceFailure, RuntimeConfig


def _jpeg(width: int = 64, height: int = 36) -> bytes:
    encoded, payload = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError("test JPEG encoding failed")
    return payload.tobytes()


def _image(extension: str, width: int = 64, height: int = 36) -> bytes:
    encoded, payload = cv2.imencode(
        extension,
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError(f"test {extension} encoding failed")
    return payload.tobytes()


def _request(
    *,
    data: bytes | None = None,
    timestamp: int = 1,
    frame_id: int = 0,
    batch_size: int = 1,
    session_id: str = "session-a",
):
    return ai_processor_pb2.VideoChunk(
        data=_jpeg() if data is None else data,
        timestamp=timestamp,
        frame_id=frame_id,
        batch_size=batch_size,
        session_id=session_id,
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


class FakeAdaFace:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        self.error: Exception | None = None
        self.overflow = False
        self.calls = 0

    def submit(self, _image: np.ndarray, *, owner: str):
        del owner
        self.calls += 1
        if self.overflow:
            return None
        future = asyncio.get_running_loop().create_future()
        if self.error is None:
            future.set_result(self.embedding.copy())
        else:
            future.set_exception(self.error)
        return future


class LoopbackServer:
    def __init__(
        self,
        runtime: FakeRuntime,
        *,
        inference_timeout_seconds: float = 1.5,
        adaface: FakeAdaFace | None = None,
        max_sessions: int = 1_024,
    ):
        self.runtime = runtime
        self.inference_timeout_seconds = inference_timeout_seconds
        self.adaface = adaface
        self.sessions = SessionRegistry(max_sessions=max_sessions)
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
            inference_timeout_seconds=self.inference_timeout_seconds,
            shutdown_grace_seconds=0,
        )
        self.bundle = build_grpc_server(
            settings,
            self.runtime,
            sessions=self.sessions,
            adaface=self.adaface,
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
    async def test_session_registry_size_is_bounded_for_unpaginated_listing(self):
        with self.assertRaisesRegex(ValueError, "1..1024"):
            GrpcServerSettings(
                runtime=RuntimeConfig(engine=Path("unused.engine")),
                max_sessions=1_025,
            )

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

    async def test_session_id_cannot_change_within_a_stream(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            call = server.stub.ProcessVideo()
            await call.write(_request(frame_id=1, session_id="session-a"))
            self.assertEqual((await call.read()).status_message, "success")
            await call.write(_request(frame_id=2, session_id="session-b"))
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await call.read()

        self.assertEqual(rejected.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(len(runtime.calls), 1)

    async def test_blank_session_id_is_rejected_before_inference(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            call = server.stub.ProcessVideo(_requests(_request(frame_id=1, session_id="  ")))
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await call.read()

        self.assertEqual(rejected.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(runtime.calls, [])

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

    async def test_process_video_keeps_the_jpeg_only_contract(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            response = await server.stub.ProcessVideo(
                _requests(_request(data=_image(".png"), frame_id=1))
            ).read()
            sessions = server.sessions.list_summaries()

        self.assertEqual(response.error_code, "DECODE_FAILED")
        self.assertEqual(runtime.calls, [])
        self.assertEqual(sessions, ())

    async def test_rejected_first_frame_does_not_consume_a_session_slot(self):
        for request in (
            _request(data=b"not-a-jpeg", session_id="invalid-image"),
            _request(batch_size=2, session_id="invalid-batch"),
        ):
            with self.subTest(session_id=request.session_id):
                async with LoopbackServer(FakeRuntime(), max_sessions=1) as server:
                    rejected = await server.stub.ProcessVideo(_requests(request)).read()
                    accepted = await server.stub.ProcessVideo(
                        _requests(_request(session_id="valid-session"))
                    ).read()

                self.assertEqual(rejected.status_message, "failed")
                self.assertEqual(accepted.status_message, "success")

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
            self.assertEqual(server.bundle.servicer.active_streams, 0)
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

    async def test_many_concurrent_streams_remain_admitted_and_health_stays_serving(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            calls = [server.stub.ProcessVideo() for _ in range(8)]
            await asyncio.gather(
                *(call.write(_request(timestamp=51, frame_id=5)) for call in calls)
            )
            responses = await asyncio.gather(*(call.read() for call in calls))
            self.assertTrue(all(response.status_message == "success" for response in responses))
            self.assertEqual(server.bundle.servicer.active_streams, 8)

            health = await server.health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
            self.assertEqual(
                health.status,
                health_pb2.HealthCheckResponse.SERVING,
            )

            await asyncio.gather(*(call.done_writing() for call in calls))
            endings = await asyncio.gather(*(call.read() for call in calls))
            self.assertTrue(all(ending is grpc.aio.EOF for ending in endings))
            await _wait_until(lambda: server.bundle.servicer.active_streams == 0)

    async def test_add_whitelist_requires_an_available_adaface_model(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.AddWhitelist(
                    ai_processor_pb2.FaceData(data=_jpeg(), session_id="session-a")
                )

        self.assertEqual(
            rejected.exception.code(),
            grpc.StatusCode.FAILED_PRECONDITION,
        )

    async def test_add_whitelist_appends_independent_session_entries(self):
        adaface = FakeAdaFace()
        runtime = FakeRuntime()
        async with LoopbackServer(runtime, adaface=adaface) as server:
            first, second = await asyncio.gather(
                server.stub.AddWhitelist(
                    ai_processor_pb2.FaceData(data=_jpeg(), session_id="session-a")
                ),
                server.stub.AddWhitelist(
                    ai_processor_pb2.FaceData(data=_jpeg(), session_id="session-a")
                ),
            )
            other = await server.stub.AddWhitelist(
                ai_processor_pb2.FaceData(data=_jpeg(), session_id="session-b")
            )
            status_a, status_b, missing = await asyncio.gather(
                server.stub.GetWhitelistStatus(
                    ai_processor_pb2.GetWhitelistStatusRequest(session_id="session-a")
                ),
                server.stub.GetWhitelistStatus(
                    ai_processor_pb2.GetWhitelistStatusRequest(session_id="session-b")
                ),
                server.stub.GetWhitelistStatus(
                    ai_processor_pb2.GetWhitelistStatusRequest(session_id="session-c")
                ),
            )
            session_a = server.sessions.get_or_create("session-a").snapshot()
            session_b = server.sessions.get_or_create("session-b").snapshot()

        self.assertEqual({first.entry_count, second.entry_count}, {1, 2})
        self.assertEqual(len({first.entry_id, second.entry_id, other.entry_id}), 3)
        self.assertEqual((len(session_a.entries), session_a.version), (2, 2))
        self.assertEqual((len(session_b.entries), session_b.version), (1, 1))
        self.assertEqual((status_a.entry_count, status_a.whitelist_version), (2, 2))
        self.assertEqual((status_b.entry_count, status_b.whitelist_version), (1, 1))
        self.assertEqual((missing.entry_count, missing.whitelist_version), (0, 0))
        self.assertEqual(adaface.calls, 3)

    async def test_add_whitelist_accepts_png_and_webp(self):
        adaface = FakeAdaFace()
        async with LoopbackServer(FakeRuntime(), adaface=adaface) as server:
            responses = await asyncio.gather(
                *(
                    server.stub.AddWhitelist(
                        ai_processor_pb2.FaceData(
                            data=_image(extension),
                            session_id="image-session",
                        )
                    )
                    for extension in (".png", ".webp")
                )
            )

        self.assertEqual({response.entry_count for response in responses}, {1, 2})
        self.assertEqual(adaface.calls, 2)

    async def test_create_and_list_sessions_are_atomic_and_session_scoped(self):
        async with LoopbackServer(FakeRuntime()) as server:
            created = await asyncio.gather(
                *(
                    server.stub.CreateSession(ai_processor_pb2.CreateSessionRequest())
                    for _ in range(20)
                )
            )
            first_id = created[0].session_id
            server.sessions.get_or_create(first_id).append(np.asarray([1.0, 0.0]))
            listed = await server.stub.ListSessions(ai_processor_pb2.ListSessionsRequest())

        session_ids = [item.session_id for item in created]
        self.assertEqual(len(session_ids), len(set(session_ids)))
        summaries = {item.session_id: item for item in listed.sessions}
        self.assertEqual(set(summaries), set(session_ids))
        self.assertEqual(
            (summaries[first_id].entry_count, summaries[first_id].whitelist_version),
            (1, 1),
        )
        self.assertTrue(all(item.created_at_unix_ms > 0 for item in listed.sessions))

    async def test_create_session_reports_registry_capacity(self):
        async with LoopbackServer(FakeRuntime(), max_sessions=1) as server:
            await server.stub.CreateSession(ai_processor_pb2.CreateSessionRequest())
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.CreateSession(ai_processor_pb2.CreateSessionRequest())

        self.assertEqual(rejected.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)

    async def test_whitelist_status_rejects_an_empty_session(self):
        async with LoopbackServer(FakeRuntime()) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.GetWhitelistStatus(
                    ai_processor_pb2.GetWhitelistStatusRequest(session_id=" ")
                )

        self.assertEqual(rejected.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    async def test_add_whitelist_rejects_decode_and_face_validation_failures(self):
        adaface = FakeAdaFace()
        runtime = FakeRuntime()
        async with LoopbackServer(runtime, adaface=adaface) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as invalid_image:
                await server.stub.AddWhitelist(
                    ai_processor_pb2.FaceData(data=b"broken", session_id="session-a")
                )
            self.assertEqual(invalid_image.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

            for error in (
                FaceCountError("expected exactly one face, found 0"),
                FaceTooSmallError("face is too small"),
                FaceAlignmentError("alignment failed"),
            ):
                with self.subTest(error=type(error).__name__):
                    adaface.error = error
                    with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                        await server.stub.AddWhitelist(
                            ai_processor_pb2.FaceData(
                                data=_jpeg(),
                                session_id="session-a",
                            )
                        )
                    self.assertEqual(
                        rejected.exception.code(),
                        grpc.StatusCode.INVALID_ARGUMENT,
                    )

    async def test_add_whitelist_queue_overflow_is_bounded(self):
        adaface = FakeAdaFace()
        adaface.overflow = True
        async with LoopbackServer(FakeRuntime(), adaface=adaface) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.AddWhitelist(
                    ai_processor_pb2.FaceData(data=_jpeg(), session_id="session-a")
                )

        self.assertEqual(rejected.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)

    async def test_same_session_concurrent_streams_use_distinct_trackers(self):
        runtime = FakeRuntime()
        async with LoopbackServer(runtime) as server:
            first = server.stub.ProcessVideo()
            second = server.stub.ProcessVideo()
            await asyncio.gather(
                first.write(_request(frame_id=1, session_id="shared")),
                second.write(_request(frame_id=1, session_id="shared")),
            )
            responses = await asyncio.gather(first.read(), second.read())
            await asyncio.gather(first.done_writing(), second.done_writing())
            await asyncio.gather(first.read(), second.read())

        self.assertTrue(all(response.status_message == "success" for response in responses))
        self.assertEqual(len(server.trackers), 2)
        self.assertIsNot(server.trackers[0], server.trackers[1])

    async def test_cancellation_settles_inference_before_reset_and_release(self):
        runtime = BlockingRuntime()
        async with LoopbackServer(runtime) as server:
            call = server.stub.ProcessVideo()
            await call.write(_request(timestamp=61, frame_id=6))
            await asyncio.wait_for(runtime.inference_started.wait(), timeout=1.0)
            tracker = server.trackers[0]
            self.assertEqual(server.bundle.servicer.active_streams, 1)

            call.cancel()
            self.assertEqual(await call.code(), grpc.StatusCode.CANCELLED)
            await asyncio.sleep(0.05)
            self.assertFalse(runtime.inference_settled.is_set())
            self.assertFalse(tracker.reset_called)
            self.assertEqual(server.bundle.servicer.active_streams, 1)

            runtime.release_inference.set()
            await asyncio.wait_for(tracker.reset_event.wait(), timeout=1.0)
            await _wait_until(lambda: server.bundle.servicer.active_streams == 0)

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
            self.assertEqual(server.bundle.servicer.active_streams, 1)
            health = await server.health.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME))
            self.assertEqual(
                health.status,
                health_pb2.HealthCheckResponse.NOT_SERVING,
            )

            runtime.release_inference.set()
            self.assertIs(await call.read(), grpc.aio.EOF)
            await _wait_until(lambda: server.bundle.servicer.active_streams == 0)
            self.assertTrue(server.trackers[0].reset_after_inference)


if __name__ == "__main__":
    unittest.main()
