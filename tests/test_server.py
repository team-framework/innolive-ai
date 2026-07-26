from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import grpc
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from grpc_client import VideoResult, VideoRpcError
from protos import ai_processor_pb2
from server import ServerSettings, create_app, parse_args
from service.protocol import VERSION, decode_response, decode_result, encode_request


def jpeg(width: int = 64, height: int = 36, *, value: int = 0) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        np.full((height, width, 3), value, dtype=np.uint8),
    )
    if not success:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


class FakeGrpcClient:
    def __init__(self, *, serving: bool = True, mode: str = "success") -> None:
        self.serving = serving
        self.mode = mode
        self.closed = False
        self.received: list[Any] = []
        self.stream_sessions: list[str] = []
        self.whitelist_images: list[tuple[str, bytes]] = []
        self.whitelist_entry_ids: dict[str, list[str]] = {}
        self.whitelist_counts: dict[str, int] = {}
        self.whitelist_versions: dict[str, int] = {}
        self.sessions: dict[str, int] = {}
        self.active_stream_counts: dict[str, int] = {}
        self.created_sessions = 0
        self.processed_jpeg = jpeg(value=192)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        self.closed = True

    async def is_serving(self, service: str, *, timeout: float) -> bool:
        self.health_request = (service, timeout)
        return self.serving

    async def process_video(
        self,
        frames,
        *,
        session_id: str,
        window: int,
        raise_frame_errors: bool,
    ) -> AsyncIterator[VideoResult]:
        self.stream_options = (session_id, window, raise_frame_errors)
        self.stream_sessions.append(session_id)
        async for frame in frames:
            self.received.append(frame)
            response = self._response(frame)
            yield VideoResult(source_jpeg=frame.data, response=response)
            if response.error_code == "INFERENCE_FAILED":
                return

    async def add_whitelist(self, image: bytes, *, session_id: str):
        if self.mode == "whitelist_error":
            raise VideoRpcError(
                "AddWhitelist",
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "whitelist full",
            )
        self.whitelist_images.append((session_id, image))
        self._ensure_session(session_id)
        entry_id = f"entry-{len(self.whitelist_images)}"
        entry_ids = self.whitelist_entry_ids.setdefault(session_id, [])
        entry_ids.append(entry_id)
        count = len(entry_ids)
        version = self.whitelist_versions.get(session_id, 0) + 1
        self.whitelist_counts[session_id] = count
        self.whitelist_versions[session_id] = version
        return ai_processor_pb2.WhitelistResponse(
            entry_id=entry_id,
            entry_count=count,
            whitelist_version=version,
        )

    async def get_whitelist_status(self, session_id: str):
        count = self.whitelist_counts.get(session_id, 0)
        return ai_processor_pb2.GetWhitelistStatusResponse(
            session_id=session_id,
            entry_count=count,
            whitelist_version=self.whitelist_versions.get(session_id, 0),
            entry_ids=self.whitelist_entry_ids.get(session_id, ()),
        )

    async def delete_whitelist(self, entry_id: str, *, session_id: str):
        entry_ids = self.whitelist_entry_ids.get(session_id, [])
        if entry_id not in entry_ids:
            raise VideoRpcError(
                "DeleteWhitelist",
                grpc.StatusCode.NOT_FOUND,
                "whitelist entry does not exist",
            )
        entry_ids.remove(entry_id)
        version = self.whitelist_versions.get(session_id, 0) + 1
        self.whitelist_counts[session_id] = len(entry_ids)
        self.whitelist_versions[session_id] = version
        return ai_processor_pb2.WhitelistResponse(
            entry_id=entry_id,
            entry_count=len(entry_ids),
            whitelist_version=version,
        )

    async def create_session(self):
        if self.mode == "session_error":
            raise VideoRpcError(
                "CreateSession",
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "session registry full",
            )
        self.created_sessions += 1
        session_id = f"session-generated-{self.created_sessions}"
        self._ensure_session(session_id)
        return self._session_info(session_id)

    async def list_sessions(self):
        if self.mode == "session_error":
            raise VideoRpcError(
                "ListSessions",
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "session registry full",
            )
        newest_first = sorted(
            self.sessions,
            key=lambda session_id: self.sessions[session_id],
            reverse=True,
        )
        return tuple(self._session_info(session_id) for session_id in newest_first)

    async def delete_session(self, session_id: str):
        if session_id not in self.sessions:
            raise VideoRpcError(
                "DeleteSession",
                grpc.StatusCode.NOT_FOUND,
                "session does not exist",
            )
        if self.active_stream_counts.get(session_id, 0):
            raise VideoRpcError(
                "DeleteSession",
                grpc.StatusCode.FAILED_PRECONDITION,
                "session has active video streams",
            )
        del self.sessions[session_id]
        self.whitelist_entry_ids.pop(session_id, None)
        self.whitelist_counts.pop(session_id, None)
        self.whitelist_versions.pop(session_id, None)

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = 1_000 + len(self.sessions)

    def _session_info(self, session_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            session_id=session_id,
            entry_count=self.whitelist_counts.get(session_id, 0),
            whitelist_version=self.whitelist_versions.get(session_id, 0),
            created_at_unix_ms=self.sessions[session_id],
            active_stream_count=self.active_stream_counts.get(session_id, 0),
        )

    def _response(self, frame):
        if self.mode == "decode_then_success" and b"broken" in frame.data:
            return ai_processor_pb2.ProcessedVideoChunk(
                timestamp=frame.timestamp,
                status_message="failed",
                frame_id=frame.frame_id,
                error_code="DECODE_FAILED",
                error_message="frame could not be decoded",
            )
        if self.mode == "fatal":
            return ai_processor_pb2.ProcessedVideoChunk(
                timestamp=frame.timestamp,
                status_message="failed",
                frame_id=frame.frame_id,
                error_code="INFERENCE_FAILED",
                error_message="injected inference failure",
            )
        if self.mode == "invalid_data":
            processed_jpeg = b"not-a-jpeg"
        elif self.mode == "missing_data":
            processed_jpeg = b""
        else:
            processed_jpeg = self.processed_jpeg
        return ai_processor_pb2.ProcessedVideoChunk(
            data=processed_jpeg,
            timestamp=frame.timestamp,
            status_message="success",
            width=64,
            height=36,
            frame_id=frame.frame_id,
            faces=[
                ai_processor_pb2.FaceMetadata(
                    bbox=ai_processor_pb2.BoundingBox(x1=1, y1=2, x2=30, y2=32),
                    confidence=0.9,
                    polygon=[
                        ai_processor_pb2.Point(x=1, y=2),
                        ai_processor_pb2.Point(x=30, y=2),
                        ai_processor_pb2.Point(x=30, y=32),
                    ],
                    track_id=1,
                    source="detected",
                    class_name="face",
                    whitelisted=True,
                )
            ],
            timing=ai_processor_pb2.ProcessingTiming(
                queue_ms=0.1,
                decode_ms=0.2,
                inference_ms=1.0,
                tracking_ms=0.3,
                blur_encode_ms=0.4,
                serialize_ms=0.1,
                server_total_ms=1.7,
                runtime_total_ms=1.4,
                inference_batch_size=1,
            ),
            stats=ai_processor_pb2.FrameStats(tracker_frame=frame.frame_id),
        )


def app(client: FakeGrpcClient | None = None):
    fake = client or FakeGrpcClient()
    application = create_app(
        ServerSettings(
            session_id="demo-session",
            grpc_target="test-grpc:50051",
            max_jpeg_bytes=4096,
        ),
        client_factory=lambda *_args, **_kwargs: fake,
    )
    return application, fake


class GrpcDemoGatewayTests(unittest.TestCase):
    def test_browser_gateway_no_longer_requires_a_manual_session_argument(self):
        with patch("sys.argv", ["server.py", "--host", "127.0.0.1", "--port", "8002"]):
            arguments = parse_args()

        self.assertEqual(arguments.session_id, "demo-session")

    def test_health_identifies_the_real_grpc_route(self):
        application, fake = app()
        with TestClient(application) as client:
            health = client.get("/healthz")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            health.json()["transport_path"], ["browser-websocket", "grpc-bidi-ProcessVideo"]
        )
        self.assertEqual(health.json()["grpc"]["target"], "test-grpc:50051")
        self.assertTrue(health.json()["grpc"]["serving"])
        self.assertEqual(health.json()["protocol"], {"name": "ILF1", "version": VERSION})
        self.assertEqual(
            health.json()["enrollment_content_types"],
            ["image/jpeg", "image/png", "image/webp"],
        )
        self.assertTrue(fake.closed)

    def test_result_returns_server_processed_jpeg_in_a_binary_envelope(self):
        application, fake = app()
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            source_jpeg = jpeg()
            websocket.send_bytes(encode_request(7, source_jpeg))
            result, result_jpeg = decode_result(websocket.receive_bytes())

        self.assertEqual((result["type"], result["seq"]), ("result", 7))
        self.assertEqual((result["width"], result["height"]), (64, 36))
        self.assertEqual(result["transport"], "grpc")
        self.assertIn("grpc_round_trip", result["timing_ms"])
        self.assertEqual(result["timing_ms"]["blur_encode"], 0.4)
        self.assertNotIn("jpeg", result)
        self.assertNotIn("data", result)
        self.assertEqual(result_jpeg, fake.processed_jpeg)
        self.assertNotEqual(result_jpeg, source_jpeg)
        self.assertTrue(result["objects"][0]["whitelisted"])
        self.assertEqual(result["session_id"], "demo-session")
        self.assertEqual(fake.stream_options, ("demo-session", 5, False))
        self.assertEqual(fake.received[0].frame_id, 7)

    def test_multiple_enrollments_and_status_are_isolated_by_session(self):
        application, fake = app()
        face = jpeg()
        with TestClient(application) as client:
            first = client.post(
                "/api/whitelist?session_id=session-a",
                content=face,
                headers={"content-type": "image/jpeg"},
            )
            second = client.post(
                "/api/whitelist?session_id=session-a",
                content=face,
                headers={"content-type": "image/jpeg"},
            )
            status_a = client.get("/api/whitelist?session_id=session-a")
            status_b = client.get("/api/whitelist?session_id=session-b")

        self.assertEqual((first.status_code, second.status_code), (201, 201))
        self.assertEqual(first.json()["entry_count"], 1)
        self.assertEqual(second.json()["entry_count"], 2)
        self.assertEqual(status_a.json()["entry_count"], 2)
        self.assertEqual(status_a.json()["entry_ids"], ["entry-1", "entry-2"])
        self.assertEqual(status_a.headers["cache-control"], "no-store")
        self.assertEqual(status_b.json()["entry_count"], 0)
        self.assertEqual(status_b.json()["entry_ids"], [])
        self.assertEqual(fake.whitelist_images, [("session-a", face), ("session-a", face)])

    def test_whitelist_api_deletes_one_entry_and_preserves_grpc_status(self):
        application, fake = app()
        face = jpeg()
        with TestClient(application) as client:
            first = client.post("/api/whitelist?session_id=session-a", content=face)
            second = client.post("/api/whitelist?session_id=session-a", content=face)
            fake.active_stream_counts["session-a"] = 1
            deleted = client.delete("/api/whitelist?session_id=session-a&entry_id=entry-1")
            status = client.get("/api/whitelist?session_id=session-a")
            missing = client.delete("/api/whitelist?session_id=session-a&entry_id=entry-1")
            invalid = client.delete("/api/whitelist?session_id=session-a")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(status.json()["entry_count"], 1)
        self.assertEqual(status.json()["whitelist_version"], 3)
        self.assertEqual(status.json()["entry_ids"], ["entry-2"])
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "NOT_FOUND")
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["error"]["code"], "INVALID_ARGUMENT")

    def test_session_api_creates_unique_sessions_and_lists_independent_state(self):
        application, _ = app()
        face = jpeg()
        with TestClient(application) as client:
            first = client.post("/api/sessions")
            second = client.post("/api/sessions")
            first_id = first.json()["session_id"]
            second_id = second.json()["session_id"]
            client.post(f"/api/whitelist?session_id={first_id}", content=face)
            client.post(f"/api/whitelist?session_id={first_id}", content=face)
            client.post(f"/api/whitelist?session_id={second_id}", content=face)
            listed = client.get("/api/sessions")

        self.assertEqual((first.status_code, second.status_code), (201, 201))
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first.json()["entry_count"], 0)
        self.assertEqual(first.json()["whitelist_version"], 0)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.headers["cache-control"], "no-store")
        sessions = {item["session_id"]: item for item in listed.json()["sessions"]}
        self.assertEqual(sessions[first_id]["entry_count"], 2)
        self.assertEqual(sessions[first_id]["whitelist_version"], 2)
        self.assertEqual(sessions[second_id]["entry_count"], 1)
        self.assertEqual(sessions[second_id]["whitelist_version"], 1)
        self.assertIsInstance(sessions[first_id]["created_at_unix_ms"], int)

    def test_session_api_preserves_grpc_status(self):
        application, _ = app(FakeGrpcClient(mode="session_error"))
        with TestClient(application) as client:
            created = client.post("/api/sessions")
            listed = client.get("/api/sessions")

        self.assertEqual(created.status_code, 429)
        self.assertEqual(created.json()["error"]["code"], "RESOURCE_EXHAUSTED")
        self.assertEqual(listed.status_code, 429)
        self.assertEqual(listed.json()["error"]["code"], "RESOURCE_EXHAUSTED")

    def test_session_api_deletes_only_idle_existing_sessions(self):
        application, fake = app()
        with TestClient(application) as client:
            session_id = client.post("/api/sessions").json()["session_id"]
            fake.active_stream_counts[session_id] = 1
            active = client.delete(f"/api/sessions/{session_id}")
            fake.active_stream_counts[session_id] = 0
            deleted = client.delete(f"/api/sessions/{session_id}")
            missing = client.delete(f"/api/sessions/{session_id}")

        self.assertEqual(active.status_code, 409)
        self.assertEqual(active.json()["error"]["code"], "FAILED_PRECONDITION")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "NOT_FOUND")

    def test_session_delete_preserves_an_encoded_session_path(self):
        application, _ = app()
        with TestClient(application) as client:
            created = client.post(
                "/api/whitelist?session_id=session%2Fwith%20spaces",
                content=jpeg(),
            )
            deleted = client.delete("/api/sessions/session%2Fwith%20spaces")

        self.assertEqual(created.status_code, 201)
        self.assertEqual(deleted.status_code, 204)

    def test_whitelist_api_validates_session_and_body_limit(self):
        application, _ = app()
        with TestClient(application) as client:
            missing = client.get("/api/whitelist")
            oversized = client.post(
                "/api/whitelist?session_id=session-a",
                content=b"x" * 4097,
            )

        self.assertEqual(missing.status_code, 400)
        self.assertEqual(missing.json()["error"]["code"], "INVALID_ARGUMENT")
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(oversized.json()["error"]["code"], "PAYLOAD_TOO_LARGE")

    def test_whitelist_api_preserves_grpc_status(self):
        application, _ = app(FakeGrpcClient(mode="whitelist_error"))
        with TestClient(application) as client:
            response = client.post(
                "/api/whitelist?session_id=session-a",
                content=jpeg(),
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "RESOURCE_EXHAUSTED")

    def test_websocket_query_selects_an_independent_session(self):
        application, fake = app()
        with TestClient(application) as client, ExitStack() as stack:
            first = stack.enter_context(client.websocket_connect("/ws?session_id=session-a"))
            second = stack.enter_context(client.websocket_connect("/ws?session_id=session-b"))
            first.send_bytes(encode_request(1, jpeg()))
            second.send_bytes(encode_request(1, jpeg()))
            first_result, _ = decode_result(first.receive_bytes())
            second_result, _ = decode_result(second.receive_bytes())

        self.assertEqual(first_result["session_id"], "session-a")
        self.assertEqual(second_result["session_id"], "session-b")
        self.assertCountEqual(fake.stream_sessions, ["session-a", "session-b"])

    def test_grpc_decode_error_is_forwarded_and_stream_can_continue(self):
        application, _ = app(FakeGrpcClient(mode="decode_then_success"))
        invalid = b"\xff\xd8broken\xff\xd9"
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(1, invalid))
            error = decode_response(websocket.receive_text())
            websocket.send_bytes(encode_request(2, jpeg()))
            result, result_jpeg = decode_result(websocket.receive_bytes())

        self.assertEqual(
            (error["type"], error["seq"], error["code"]),
            ("error", 1, "DECODE_FAILED"),
        )
        self.assertEqual((result["type"], result["seq"]), ("result", 2))
        self.assertTrue(result_jpeg.startswith(b"\xff\xd8"))

    def test_duplicate_sequence_is_rejected_before_grpc(self):
        application, fake = app()
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(3, jpeg()))
            result, _ = decode_result(websocket.receive_bytes())
            self.assertEqual(result["type"], "result")
            websocket.send_bytes(encode_request(3, jpeg()))
            error = decode_response(websocket.receive_text())

        self.assertEqual(error["code"], "NON_MONOTONIC_SEQUENCE")
        self.assertEqual([frame.frame_id for frame in fake.received], [3])

    def test_unknown_header_returns_sequence_then_closes(self):
        application, _ = app()
        payload = b"OLD1\x00\x00\x00\x09" + jpeg()
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(payload)
            error = decode_response(websocket.receive_text())
            with self.assertRaises(WebSocketDisconnect):
                websocket.receive_text()

        self.assertEqual((error["seq"], error["code"]), (9, "INVALID_HEADER"))

    def test_fatal_grpc_frame_error_is_forwarded_then_closes(self):
        application, _ = app(FakeGrpcClient(mode="fatal"))
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(5, jpeg()))
            error = decode_response(websocket.receive_text())
            with self.assertRaises(WebSocketDisconnect):
                websocket.receive_text()

        self.assertEqual((error["seq"], error["code"]), (5, "INFERENCE_FAILED"))

    def test_missing_or_invalid_server_processed_jpeg_fails_closed(self):
        for mode in ("missing_data", "invalid_data"):
            with self.subTest(mode=mode):
                application, _ = app(FakeGrpcClient(mode=mode))
                with (
                    TestClient(application) as client,
                    client.websocket_connect("/ws") as websocket,
                ):
                    websocket.send_bytes(encode_request(6, jpeg()))
                    error = decode_response(websocket.receive_text())
                    with self.assertRaises(WebSocketDisconnect):
                        websocket.receive_text()

                self.assertEqual(
                    (error["type"], error["seq"], error["code"]),
                    ("error", 6, "GRPC_STREAM_FAILED"),
                )

    def test_many_browser_streams_share_the_grpc_client_without_admission_limit(self):
        application, fake = app()
        with TestClient(application) as client, ExitStack() as stack:
            websockets = [stack.enter_context(client.websocket_connect("/ws")) for _ in range(8)]
            for websocket in websockets:
                websocket.send_bytes(encode_request(1, jpeg()))
            results = [decode_result(websocket.receive_bytes())[0] for websocket in websockets]

        self.assertTrue(all(result["type"] == "result" for result in results))
        self.assertEqual(len(fake.received), 8)

    def test_not_serving_grpc_backend_rejects_health_and_stream(self):
        application, _ = app(FakeGrpcClient(serving=False))
        with TestClient(application) as client:
            health = client.get("/healthz")
            self.assertEqual(health.status_code, 503)
            self.assertFalse(health.json()["grpc"]["serving"])
            with (
                client.websocket_connect("/ws") as websocket,
                self.assertRaises(WebSocketDisconnect) as closed,
            ):
                websocket.receive_text()
        self.assertEqual(closed.exception.code, 1013)


if __name__ == "__main__":
    unittest.main()
