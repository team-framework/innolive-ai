import asyncio
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

try:
    import av
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError

    from service.webrtc import LatestFrameSlot, ProcessedVideoTrack, WebRtcGateway, pack_frame
    from service.webrtc_encoder import RealtimeH264Encoder, configure_encoder
except ImportError as error:
    raise unittest.SkipTest("install requirements-webrtc.txt for WebRTC tests") from error

from fastapi import FastAPI
from fastapi.testclient import TestClient

from grpc_client import VideoFrameError, VideoResult
from protos import ai_processor_pb2 as pb
from server import ServerSettings


class Source(MediaStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.frames = asyncio.Queue()

    async def recv(self):
        return await self.frames.get()


class Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False
        self.requests = []

    async def process_video(self, frames, **kwargs):
        try:
            async for frame in frames:
                self.requests.append(frame)
                if self.fail:
                    raise VideoFrameError(frame.frame_id, "test failure")
                result = pb.ProcessedVideoChunk(
                    data=frame.data,
                    width=frame.width,
                    height=frame.height,
                    timestamp=frame.timestamp,
                    frame_id=frame.frame_id,
                )
                yield VideoResult(frame.data, result)
        finally:
            self.closed = True
            await frames.aclose()


class WebRtcTests(unittest.IsolatedAsyncioTestCase):
    def frame(self, width=640, height=480, pts=3000):
        frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), np.uint8), format="bgr24")
        frame.pts, frame.time_base = pts, Fraction(1, 90_000)
        return frame

    async def test_latest_slot_drops_pending_input_and_close_wakes_waiter(self):
        slot = LatestFrameSlot()
        for i in range(10):
            slot.put(i)
        self.assertEqual(await slot.get(), 9)
        self.assertEqual(slot.dropped, 9)
        waiter = asyncio.create_task(slot.get())
        slot.close()
        with self.assertRaises(MediaStreamError):
            await waiter

    def test_raw_conversion_bounds_size_strides_and_preserves_timestamp(self):
        packed = pack_frame(self.frame(1280, 720), 1, -1, "face_swap")
        self.assertEqual((packed.width, packed.height), (640, 360))
        self.assertEqual(len(packed.data), 640 * 360 * 3 // 2)
        self.assertEqual(packed.timestamp, 3000)
        self.assertEqual(pack_frame(self.frame(642, 480, 1), 2, 3000, "blur").timestamp, 3001)

    async def test_processed_track_uses_raw_stream_and_releases_client(self):
        source, client, reports = Source(), Client(), []
        track = ProcessedVideoTrack(
            source, client, "test", "face_swap", reports.append, lambda: None
        )
        source.frames.put_nowait(self.frame())
        result = await asyncio.wait_for(track.recv(), 2)
        self.assertEqual(result.format.name, "yuv420p")
        self.assertEqual(result.pts, 3000)
        self.assertEqual(client.requests[0].pix_fmt, "yuv420p")
        self.assertEqual(reports[0]["processed"], 1)
        await track.close()
        self.assertTrue(client.closed)
        self.assertTrue(track.drain_task.done())

    async def test_failure_does_not_return_unprocessed_source(self):
        source, client, reports, failures = Source(), Client(fail=True), [], []
        track = ProcessedVideoTrack(
            source, client, "test", "face_swap", reports.append, lambda: failures.append(True)
        )
        source.frames.put_nowait(self.frame())
        with self.assertRaises(MediaStreamError):
            await track.recv()
        self.assertEqual(reports[-1]["type"], "error")
        self.assertEqual(failures, [True])
        await track.close()

    def test_h264_packetizes_each_frame_with_no_b_frames(self):
        encoder = RealtimeH264Encoder("libx264")
        for i in range(3):
            packets, timestamp = encoder.encode(self.frame(64, 64, i * 3000), force_keyframe=i == 0)
            self.assertTrue(packets)
            self.assertEqual(timestamp, i * 3000)
        self.assertEqual(encoder.codec.max_b_frames, 0)

    def test_auto_encoder_requires_a_successful_probe_and_explicit_nvenc_fails(self):
        def probe(name):
            if name == "h264_nvenc":
                raise RuntimeError("driver unavailable")

        with patch("service.webrtc_encoder.probe_encoder", side_effect=probe):
            self.assertEqual(configure_encoder("auto"), "libx264")
            with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
                configure_encoder("nvenc")

    def test_offer_validation_capacity_and_failed_negotiation_cleanup(self):
        gateway = WebRtcGateway(ServerSettings(webrtc=True))
        app = FastAPI()
        app.state.ready = True
        app.state.grpc_client = SimpleNamespace(get_whitelist_status=AsyncMock())
        app.add_api_route("/offer", gateway.offer, methods=["POST"])
        with TestClient(app) as client:
            self.assertEqual(client.post("/offer", json=[]).status_code, 400)
            self.assertEqual(client.post("/offer", content=b"x" * 70_000).status_code, 413)
            self.assertEqual(
                client.post(
                    "/offer", json={"sdp": "v=0\r\n", "type": "offer", "session_id": "test"}
                ).status_code,
                400,
            )
            self.assertEqual(gateway.peers, {})
            gateway.peers = {"one": {}, "two": {}}
            self.assertEqual(client.post("/offer", json={}).status_code, 429)

    def test_settings_reject_invalid_ice_and_capacity(self):
        for kwargs in (
            {"webrtc_max_peers": 0},
            {"webrtc_encoder": "unknown"},
            {"webrtc_ice_servers": ({"urls": "https://invalid"},)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ServerSettings(**kwargs)
