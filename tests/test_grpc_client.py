from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

import cv2
import grpc
import numpy as np
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from grpc_client import (
    VideoFrame,
    VideoFrameError,
    VideoProcessorClient,
    VideoProtocolError,
    VideoRpcError,
)
from protos import ai_processor_pb2, ai_processor_pb2_grpc
from service.grpc_config import server_options


def _jpeg(value: int) -> bytes:
    encoded, payload = cv2.imencode(
        ".jpg",
        np.full((32, 48, 3), value, dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError("test JPEG encoding failed")
    return payload.tobytes()


def _image(extension: str, value: int) -> bytes:
    encoded, payload = cv2.imencode(
        extension,
        np.full((32, 48, 3), value, dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError(f"test {extension} encoding failed")
    return payload.tobytes()


class ClientContractServicer(ai_processor_pb2_grpc.AiProcessorServicer):
    def __init__(self, mode: str = "success") -> None:
        self.mode = mode
        self.requests: list[Any] = []
        self.whitelist_requests: list[Any] = []
        self.whitelist_counts: dict[str, int] = {}
        self.sessions: list[str] = []

    async def ProcessVideo(self, request_iterator, context) -> AsyncIterator[Any]:
        if self.mode == "early_eof":
            return

        batch: list[Any] = []
        async for request in request_iterator:
            self.requests.append(request)
            if self.mode != "success":
                yield self._special_response(request)
                return
            batch.append(request)
            if len(batch) == 5:
                for item in batch:
                    yield self._success(item)
                batch.clear()
        for item in batch:
            yield self._success(item)

    async def AddWhitelist(self, request, context):
        if self.mode == "whitelist_error":
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "whitelist full")
        self.whitelist_requests.append(request)
        count = self.whitelist_counts.get(request.session_id, 0) + 1
        self.whitelist_counts[request.session_id] = count
        return ai_processor_pb2.WhitelistResponse(
            status_message="success",
            entry_id=f"entry-{len(self.whitelist_requests)}",
            entry_count=count,
            whitelist_version=count,
        )

    async def GetWhitelistStatus(self, request, context):
        del context
        count = self.whitelist_counts.get(request.session_id, 0)
        return ai_processor_pb2.GetWhitelistStatusResponse(
            session_id=request.session_id,
            entry_count=count,
            whitelist_version=count,
        )

    async def CreateSession(self, request, context):
        del request
        del context
        session_id = f"session-generated-{len(self.sessions) + 1}"
        self.sessions.append(session_id)
        return ai_processor_pb2.SessionInfo(
            session_id=session_id,
            created_at_unix_ms=len(self.sessions),
        )

    async def ListSessions(self, request, context):
        del request
        del context
        return ai_processor_pb2.ListSessionsResponse(
            sessions=[
                ai_processor_pb2.SessionInfo(
                    session_id=session_id,
                    entry_count=self.whitelist_counts.get(session_id, 0),
                    whitelist_version=self.whitelist_counts.get(session_id, 0),
                    created_at_unix_ms=index,
                )
                for index, session_id in enumerate(self.sessions, start=1)
            ]
        )

    @staticmethod
    def _success(request):
        return ai_processor_pb2.ProcessedVideoChunk(
            timestamp=request.timestamp,
            status_message="success",
            width=48,
            height=32,
            frame_id=request.frame_id,
            processing_ms=1.0,
            timing=ai_processor_pb2.ProcessingTiming(
                inference_batch_size=1,
                server_total_ms=1.0,
            ),
        )

    def _special_response(self, request):
        response = self._success(request)
        if self.mode == "pixel_echo":
            response.data = request.data
        elif self.mode == "reorder":
            response.frame_id += 1
        elif self.mode == "frame_error":
            response.status_message = "failed"
            response.error_code = "INFERENCE_FAILED"
            response.error_message = "frame inference failed"
        else:
            raise AssertionError(f"unknown test mode: {self.mode}")
        return response


class ClientLoopback:
    def __init__(self, mode: str = "success") -> None:
        self.servicer = ClientContractServicer(mode)
        self.server = grpc.aio.server(options=server_options())
        self.health = health.aio.HealthServicer()
        ai_processor_pb2_grpc.add_AiProcessorServicer_to_server(
            self.servicer,
            self.server,
        )
        health_pb2_grpc.add_HealthServicer_to_server(self.health, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")

    async def __aenter__(self) -> ClientLoopback:
        await self.health.set(
            "AiProcessor",
            health_pb2.HealthCheckResponse.SERVING,
        )
        await self.server.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.server.stop(0)


class GrpcClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_health_service_is_exposed_by_the_client(self):
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            self.assertTrue(await client.is_serving())

    async def test_client_creates_and_lists_server_generated_sessions(self):
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            first, second = await asyncio.gather(
                client.create_session(),
                client.create_session(),
            )
            listed = await client.list_sessions()

        self.assertNotEqual(first.session_id, second.session_id)
        self.assertCountEqual(
            [item.session_id for item in listed],
            [first.session_id, second.session_id],
        )

    async def test_add_whitelist_sends_session_and_complete_jpeg(self):
        face = _jpeg(42)
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            response = await client.add_whitelist(
                face,
                session_id="client-session",
            )

        self.assertEqual(response.entry_id, "entry-1")
        self.assertEqual(len(loopback.servicer.whitelist_requests), 1)
        request = loopback.servicer.whitelist_requests[0]
        self.assertEqual(request.session_id, "client-session")
        self.assertEqual(request.data, face)

    async def test_add_whitelist_accepts_png_and_webp(self):
        images = [_image(".png", 20), _image(".webp", 30)]
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            responses = await client.add_whitelist_many(
                images,
                session_id="image-session",
            )

        self.assertEqual([response.entry_count for response in responses], [1, 2])
        self.assertEqual(
            [request.data for request in loopback.servicer.whitelist_requests],
            images,
        )

    async def test_session_bound_client_hides_repeated_session_arguments(self):
        face = _jpeg(42)
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            session = client.for_session("bound-session")
            added = await session.add_whitelist(face)
            results = [result async for result in session.process_jpegs([face])]

        self.assertEqual(added.entry_count, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(loopback.servicer.whitelist_requests[0].session_id, "bound-session")
        self.assertEqual(loopback.servicer.requests[0].session_id, "bound-session")

    async def test_multiple_enrollments_and_status_remain_session_scoped(self):
        faces = [_jpeg(10), _jpeg(20)]
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            session_a = client.for_session("session-a")
            session_b = client.for_session("session-b")
            added = await session_a.add_whitelist_many(faces)
            status_a, status_b = await asyncio.gather(
                session_a.get_whitelist_status(),
                session_b.get_whitelist_status(),
            )

        self.assertEqual([response.entry_count for response in added], [1, 2])
        self.assertEqual((status_a.session_id, status_a.entry_count), ("session-a", 2))
        self.assertEqual((status_b.session_id, status_b.entry_count), ("session-b", 0))

    async def test_rpc_error_preserves_method_status_and_details(self):
        async with (
            ClientLoopback("whitelist_error") as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            with self.assertRaises(VideoRpcError) as rejected:
                await client.add_whitelist(_jpeg(1), session_id="client-session")

        self.assertEqual(rejected.exception.method, "AddWhitelist")
        self.assertEqual(rejected.exception.code, grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertEqual(rejected.exception.details, "whitelist full")

    async def test_easy_jpeg_api_assigns_ids_and_keeps_exact_sources_at_w5(self):
        jpegs = [_jpeg(value) for value in range(7)]
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            results = [
                result
                async for result in client.process_jpegs(
                    jpegs,
                    session_id="client-session",
                    start_frame_id=10,
                )
            ]
            max_inflight = client.max_inflight_observed

        self.assertEqual(
            [result.response.frame_id for result in results],
            list(range(10, 17)),
        )
        self.assertEqual(
            [result.source_jpeg for result in results],
            jpegs,
        )
        self.assertTrue(all(result.response.timestamp > 0 for result in results))
        self.assertEqual(max_inflight, 5)
        self.assertTrue(all(result.response.data == b"" for result in results))
        self.assertTrue(
            all(request.session_id == "client-session" for request in loopback.servicer.requests)
        )

    async def test_server_frame_error_is_typed_and_cancels_the_stream(self):
        async with (
            ClientLoopback("frame_error") as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            with self.assertRaises(VideoFrameError) as failed:
                async for _ in client.process_jpegs(
                    [_jpeg(1), _jpeg(2)], session_id="client-session"
                ):
                    pass

        self.assertEqual(failed.exception.frame_id, 1)
        self.assertIn("INFERENCE_FAILED", failed.exception.detail)

    async def test_demo_mode_can_yield_a_typed_frame_error_response(self):
        async with (
            ClientLoopback("frame_error") as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            results = [
                result
                async for result in client.process_jpegs(
                    [_jpeg(1)],
                    session_id="client-session",
                    raise_frame_errors=False,
                )
            ]

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].response.error_code, "INFERENCE_FAILED")

    async def test_pixel_echo_and_reordered_response_fail_closed(self):
        for mode in ("pixel_echo", "reorder"):
            with self.subTest(mode=mode):
                async with ClientLoopback(mode) as loopback:
                    async with VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client:
                        with self.assertRaises(VideoProtocolError):
                            async for _ in client.process_jpegs(
                                [_jpeg(3)], session_id="client-session"
                            ):
                                pass

    async def test_early_eof_is_not_treated_as_success(self):
        async with (
            ClientLoopback("early_eof") as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):
            with self.assertRaises(VideoProtocolError):
                async for _ in client.process_jpegs([_jpeg(4)], session_id="client-session"):
                    pass

    async def test_explicit_consumer_close_releases_async_source(self):
        source_closed = asyncio.Event()

        async def frames() -> AsyncIterator[VideoFrame]:
            try:
                for frame_id in range(1, 20):
                    yield VideoFrame(_jpeg(frame_id), frame_id, frame_id)
            finally:
                source_closed.set()

        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
            aclosing(client.process_video(frames(), session_id="client-session")) as results,
        ):
            await anext(results)

        await asyncio.wait_for(source_closed.wait(), timeout=1.0)

    async def test_one_channel_supports_concurrent_session_streams(self):
        async with (
            ClientLoopback() as loopback,
            VideoProcessorClient(f"127.0.0.1:{loopback.port}") as client,
        ):

            async def collect(session_id: str, value: int):
                return [
                    result
                    async for result in client.process_jpegs(
                        [_jpeg(value), _jpeg(value + 1)],
                        session_id=session_id,
                    )
                ]

            first, second = await asyncio.gather(
                collect("session-a", 10),
                collect("session-b", 20),
            )

        self.assertEqual([len(first), len(second)], [2, 2])
        self.assertEqual(
            sorted(request.session_id for request in loopback.servicer.requests),
            ["session-a", "session-a", "session-b", "session-b"],
        )


if __name__ == "__main__":
    unittest.main()
