from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from grpc_client import VideoResult
from protos import ai_processor_pb2
from server import ServerSettings, create_app
from service.protocol import decode_response, encode_request


def jpeg(width: int = 64, height: int = 36) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
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
        async for frame in frames:
            self.received.append(frame)
            response = self._response(frame)
            yield VideoResult(source_jpeg=frame.data, response=response)
            if response.error_code == "INFERENCE_FAILED":
                return

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
        return ai_processor_pb2.ProcessedVideoChunk(
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
                serialize_ms=0.1,
                server_total_ms=1.7,
                runtime_total_ms=1.4,
                inference_batch_size=1,
            ),
            stats=ai_processor_pb2.FrameStats(tracker_frame=frame.frame_id),
        )


def app(client: FakeGrpcClient | None = None, *, max_streams: int = 4):
    fake = client or FakeGrpcClient()
    application = create_app(
        ServerSettings(
            session_id="demo-session",
            grpc_target="test-grpc:50051",
            max_jpeg_bytes=4096,
            max_streams=max_streams,
        ),
        client_factory=lambda *_args, **_kwargs: fake,
    )
    return application, fake


class GrpcDemoGatewayTests(unittest.TestCase):
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
        self.assertTrue(fake.closed)

    def test_result_passes_through_process_video_as_metadata_only(self):
        application, fake = app()
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(7, jpeg()))
            result = decode_response(websocket.receive_text())

        self.assertEqual((result["type"], result["seq"]), ("result", 7))
        self.assertEqual((result["width"], result["height"]), (64, 36))
        self.assertEqual(result["transport"], "grpc")
        self.assertIn("grpc_round_trip", result["timing_ms"])
        self.assertNotIn("jpeg", result)
        self.assertNotIn("data", result)
        self.assertTrue(result["objects"][0]["whitelisted"])
        self.assertEqual(fake.stream_options, ("demo-session", 5, False))
        self.assertEqual(fake.received[0].frame_id, 7)

    def test_grpc_decode_error_is_forwarded_and_stream_can_continue(self):
        application, _ = app(FakeGrpcClient(mode="decode_then_success"))
        invalid = b"\xff\xd8broken\xff\xd9"
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(1, invalid))
            error = decode_response(websocket.receive_text())
            websocket.send_bytes(encode_request(2, jpeg()))
            result = decode_response(websocket.receive_text())

        self.assertEqual(
            (error["type"], error["seq"], error["code"]),
            ("error", 1, "DECODE_FAILED"),
        )
        self.assertEqual((result["type"], result["seq"]), ("result", 2))

    def test_duplicate_sequence_is_rejected_before_grpc(self):
        application, fake = app()
        with TestClient(application) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(3, jpeg()))
            self.assertEqual(decode_response(websocket.receive_text())["type"], "result")
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

    def test_second_browser_stream_is_rejected(self):
        application, _ = app(max_streams=1)
        with (
            TestClient(application) as client,
            client.websocket_connect("/ws"),
            client.websocket_connect("/ws") as second,
            self.assertRaises(WebSocketDisconnect) as closed,
        ):
            second.receive_text()
        self.assertEqual(closed.exception.code, 1013)

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
