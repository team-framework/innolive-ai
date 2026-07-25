from __future__ import annotations

import unittest
import asyncio
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server import ServerSettings, create_app, decode_jpeg
from service.protocol import decode_response, encode_request
from service.runtime import InferenceFailure, RuntimeConfig, TrackingFailure


def jpeg(width: int = 64, height: int = 36) -> bytes:
    success, encoded = cv2.imencode(
        ".jpg",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    if not success:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


class FakeTracker:
    frame_id = 1

    def __init__(self, *_args):
        self.reset_called = False

    def reset(self):
        self.reset_called = True


class FakeRuntime:
    ready = True

    def __init__(self, _config):
        self.closed = False

    async def infer(self, _image, _tracker):
        return {
            "objects": [],
            "detections": 0,
            "raw_detections": 0,
            "continuation_candidates": 0,
            "detector_backed_tracks": 0,
            "low_confidence_continuations": 0,
            "held_tracks": 0,
            "tracks": 0,
            "tracker_frame": 1,
            "timing_ms": {
                "queue": 0.1,
                "inference": 1.0,
                "tracking": 0.2,
                "serialize": 0.1,
                "runtime_total": 1.3,
            },
        }

    def health(self):
        return {"ready": self.ready, "runtime_instances": 1, "scheduler": "serialized_b1"}

    def close(self):
        self.ready = False
        self.closed = True


def settings(**overrides) -> ServerSettings:
    values = {
        "runtime": RuntimeConfig(Path("unused.engine")),
        "max_jpeg_bytes": 4096,
    }
    values.update(overrides)
    return ServerSettings(**values)


def app(runtime_factory=FakeRuntime, **setting_overrides):
    return create_app(
        settings(**setting_overrides),
        runtime_factory=runtime_factory,
        tracker_factory=FakeTracker,
    )


class InputBoundaryTests(unittest.TestCase):
    def test_accepts_portrait_and_landscape_at_long_edge_limit(self):
        configured = settings(max_jpeg_bytes=100_000)
        self.assertEqual(decode_jpeg(jpeg(640, 360), configured).shape, (360, 640, 3))
        self.assertEqual(decode_jpeg(jpeg(360, 640), configured).shape, (640, 360, 3))

    def test_rejects_oversized_dimensions_and_invalid_jpeg(self):
        configured = settings(max_jpeg_bytes=100_000)
        with self.assertRaisesRegex(ValueError, "long-edge"):
            decode_jpeg(jpeg(641, 360), configured)
        with self.assertRaisesRegex(ValueError, "decoded"):
            decode_jpeg(b"not jpeg", configured)


class WebSocketContractTests(unittest.TestCase):
    def test_result_is_metadata_only_and_matches_sequence(self):
        with TestClient(app()) as client:
            self.assertEqual(client.get("/readyz").status_code, 200)
            with client.websocket_connect("/ws") as websocket:
                websocket.send_bytes(encode_request(7, jpeg()))
                result = decode_response(websocket.receive_text())
        self.assertEqual(result["type"], "result")
        self.assertEqual(result["seq"], 7)
        self.assertEqual(result["width"], 64)
        self.assertEqual(result["height"], 36)
        self.assertNotIn("jpeg", result)
        self.assertNotIn("data", result)

    def test_bad_jpeg_gets_one_terminal_error_and_stream_can_continue(self):
        invalid = b"\xff\xd8broken\xff\xd9"
        with TestClient(app()) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(1, invalid))
            error = decode_response(websocket.receive_text())
            websocket.send_bytes(encode_request(2, jpeg()))
            result = decode_response(websocket.receive_text())
        self.assertEqual((error["type"], error["seq"], error["code"]), ("error", 1, "DECODE_FAILED"))
        self.assertEqual((result["type"], result["seq"]), ("result", 2))

    def test_duplicate_sequence_is_terminal_error(self):
        with TestClient(app()) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(3, jpeg()))
            self.assertEqual(decode_response(websocket.receive_text())["type"], "result")
            websocket.send_bytes(encode_request(3, jpeg()))
            error = decode_response(websocket.receive_text())
        self.assertEqual(error["code"], "NON_MONOTONIC_SEQUENCE")

    def test_unknown_header_returns_recoverable_sequence_then_closes(self):
        payload = b"OLD1\x00\x00\x00\x09" + jpeg()
        with TestClient(app()) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(payload)
            error = decode_response(websocket.receive_text())
            with self.assertRaises(WebSocketDisconnect):
                websocket.receive_text()
        self.assertEqual((error["seq"], error["code"]), (9, "INVALID_HEADER"))

    def test_inference_failure_returns_terminal_error_without_partial_result(self):
        class FailedInference(FakeRuntime):
            async def infer(self, _image, _tracker):
                raise InferenceFailure("injected failure")

        with TestClient(app(FailedInference)) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(5, jpeg()))
            error = decode_response(websocket.receive_text())
        self.assertEqual((error["seq"], error["code"]), (5, "INFERENCE_FAILED"))
        self.assertNotIn("objects", error)

    def test_tracking_failure_has_its_own_terminal_stage(self):
        class FailedTracking(FakeRuntime):
            async def infer(self, _image, _tracker):
                raise TrackingFailure("injected failure")

        with TestClient(app(FailedTracking)) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(8, jpeg()))
            error = decode_response(websocket.receive_text())
        self.assertEqual((error["seq"], error["code"]), (8, "TRACKING_FAILED"))

    def test_serialization_failure_returns_error_instead_of_partial_metadata(self):
        class OversizedMetadata(FakeRuntime):
            async def infer(self, image, tracker):
                result = await super().infer(image, tracker)
                result["objects"] = [{"mask_polygon": "x" * (600 * 1024)}]
                return result

        with TestClient(app(OversizedMetadata)) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(10, jpeg()))
            error = decode_response(websocket.receive_text())
        self.assertEqual((error["seq"], error["code"]), (10, "SERIALIZATION_FAILED"))

    def test_timeout_returns_terminal_error(self):
        class SlowInference(FakeRuntime):
            async def infer(self, _image, _tracker):
                await asyncio.sleep(0.02)
                return await super().infer(_image, _tracker)

        configured = app(SlowInference, inference_timeout_seconds=0.001)
        with TestClient(configured) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_bytes(encode_request(6, jpeg()))
            error = decode_response(websocket.receive_text())
        self.assertEqual((error["seq"], error["code"]), (6, "INFERENCE_TIMEOUT"))

    def test_second_stream_is_rejected_by_admission_control(self):
        with TestClient(app()) as client, client.websocket_connect("/ws"):
            with client.websocket_connect("/ws") as second:
                with self.assertRaises(WebSocketDisconnect) as closed:
                    second.receive_text()
        self.assertEqual(closed.exception.code, 1013)

    def test_readiness_failure_rejects_health_and_stream(self):
        def failed_runtime(_config):
            raise RuntimeError("warm-up failed")

        with TestClient(app(failed_runtime)) as client:
            health = client.get("/healthz")
            self.assertEqual(health.status_code, 503)
            self.assertEqual(health.json()["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
