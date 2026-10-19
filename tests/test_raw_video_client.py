import unittest

import grpc

from grpc_client import VideoFrame, VideoProcessorClient, VideoProtocolError, _validate_frame
from protos import ai_processor_pb2 as pb
from protos import ai_processor_pb2_grpc as rpc


class EchoRaw(rpc.AiProcessorServicer):
    async def ProcessVideo(self, requests, context):
        async for frame in requests:
            yield pb.ProcessedVideoChunk(
                data=frame.data,
                width=frame.width,
                height=frame.height,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                status_message="success",
            )


class RawVideoClientTests(unittest.IsolatedAsyncioTestCase):
    def frame(self, **overrides):
        values = dict(
            data=bytes(48 * 32 * 3 // 2),
            timestamp=3000,
            frame_id=1,
            width=48,
            height=32,
            pix_fmt="yuv420p",
        )
        return VideoFrame(**(values | overrides))

    async def test_raw_roundtrip_uses_existing_proto_and_keeps_order(self):
        server = grpc.aio.server()
        rpc.add_AiProcessorServicer_to_server(EchoRaw(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with VideoProcessorClient(f"127.0.0.1:{port}") as client:
                frames = [self.frame(frame_id=i, timestamp=i * 3000) for i in range(1, 4)]
                results = [
                    r async for r in client.process_video(frames, session_id="raw-test", window=1)
                ]
                self.assertEqual([r.response.frame_id for r in results], [1, 2, 3])
                self.assertTrue(all(r.processed_data == frames[0].data for r in results))
        finally:
            await server.stop(0)

    def test_raw_rejects_bad_format_dimensions_and_payload(self):
        for values in (
            {"pix_fmt": "rgb24"},
            {"width": 49},
            {"height": 0},
            {"height": 2000},
            {"width": True},
            {"data": b"short"},
            {"pix_fmt": ""},
        ):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                _validate_frame(self.frame(**values), 0)

    def test_raw_rejects_wrong_size_and_truncated_return(self):
        source = self.frame()
        for width, data in ((32, source.data), (48, source.data[:-1])):
            result = pb.ProcessedVideoChunk(
                data=data,
                width=width,
                height=32,
                timestamp=3000,
                frame_id=1,
                status_message="success",
            )
            with self.assertRaises(VideoProtocolError):
                VideoProcessorClient._validate_response(source, result, raise_frame_errors=True)
