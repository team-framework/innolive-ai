import json
import struct
import unittest

from service.protocol import EncodedFrame, FrameResult, decode_batch, encode_result


def request_packet(frames):
    header = json.dumps(
        {
            "v": 1,
            "frames": [
                {
                    "id": frame.frame_id,
                    "capturedAt": frame.captured_at,
                    "size": len(frame.jpeg),
                }
                for frame in frames
            ],
        }
    ).encode()
    return b"".join(
        (struct.pack(">I", len(header)), header, *(frame.jpeg for frame in frames))
    )


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.frames = [
            EncodedFrame(index, 1000.0 + index, bytes([index]) * 8)
            for index in range(4)
        ]

    def test_decodes_four_jpegs_without_reordering(self):
        self.assertEqual(decode_batch(request_packet(self.frames)), self.frames)

    def test_decodes_single_frame_for_low_latency_mode(self):
        self.assertEqual(decode_batch(request_packet(self.frames[:1])), self.frames[:1])

    def test_rejects_empty_packet(self):
        with self.assertRaisesRegex(ValueError, "1 to 4"):
            decode_batch(request_packet([]))

    def test_rejects_trailing_payload(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            decode_batch(request_packet(self.frames) + b"extra")

    def test_rejects_non_object_header(self):
        header = b"[]"
        with self.assertRaisesRegex(ValueError, "must be an object"):
            decode_batch(struct.pack(">I", len(header)) + header)

    def test_response_keeps_jpeg_and_metadata(self):
        frame = self.frames[0]
        timing = {
            "inferenceMs": 4.2,
            "inferenceBatchSize": 1,
            "gateway": {"grpcResidualMs": 0.7},
        }
        result = FrameResult(
            frame,
            frame.jpeg,
            640,
            480,
            ({"trackId": 1},),
            timing,
        )
        packet = encode_result(result, 12.345)
        header_size = struct.unpack_from(">I", packet)[0]
        header = json.loads(packet[4 : 4 + header_size])
        payload = packet[4 + header_size :]

        self.assertEqual(header["processingMs"], 12.35)
        self.assertEqual(header["frames"][0]["faces"], [{"trackId": 1}])
        self.assertEqual(header["frames"][0]["timing"], timing)
        self.assertEqual(payload, frame.jpeg)


if __name__ == "__main__":
    unittest.main()
