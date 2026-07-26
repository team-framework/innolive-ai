from __future__ import annotations

import json
import struct
import unittest

from service.protocol import (
    HEADER,
    MAGIC,
    MAX_JPEG_BYTES,
    MAX_RESPONSE_BYTES,
    RESULT_HEADER,
    RESULT_MAGIC,
    VERSION,
    decode_request,
    decode_response,
    decode_result,
    encode_request,
    encode_response,
    encode_result,
    recover_sequence,
)

JPEG = b"\xff\xd8test\xff\xd9"


class ProtocolTests(unittest.TestCase):
    def test_request_round_trip_is_big_endian(self):
        payload = encode_request(0x01020304, JPEG)
        self.assertEqual(payload[: HEADER.size], b"ILF1\x01\x02\x03\x04")
        self.assertEqual(decode_request(payload), (0x01020304, JPEG))

    def test_rejects_unknown_magic(self):
        payload = struct.pack("!4sI", b"OLD1", 7) + JPEG
        with self.assertRaisesRegex(ValueError, "magic"):
            decode_request(payload)
        self.assertEqual(recover_sequence(payload), 7)

    def test_rejects_truncated_header_without_recoverable_sequence(self):
        with self.assertRaisesRegex(ValueError, "header"):
            decode_request(b"ILF1")
        self.assertIsNone(recover_sequence(b"ILF1"))

    def test_rejects_empty_incomplete_and_oversized_jpeg(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_request(HEADER.pack(MAGIC, 1))
        with self.assertRaisesRegex(ValueError, "complete JPEG"):
            decode_request(HEADER.pack(MAGIC, 1) + b"\xff\xd8open")
        with self.assertRaisesRegex(ValueError, "byte limit"):
            decode_request(HEADER.pack(MAGIC, 1) + JPEG, max_jpeg_bytes=4)

    def test_result_round_trip_is_binary_and_keeps_the_jpeg_separate(self):
        expected = {"type": "result", "seq": 0x01020304, "objects": [{"track_id": 2}]}
        payload = encode_result(expected, JPEG)

        magic, sequence, metadata_length = RESULT_HEADER.unpack_from(payload)
        metadata, result_jpeg = decode_result(payload)

        self.assertEqual((magic, sequence), (RESULT_MAGIC, 0x01020304))
        self.assertGreater(metadata_length, 0)
        self.assertEqual(metadata, {**expected, "v": VERSION})
        self.assertEqual(result_jpeg, JPEG)

    def test_error_terminal_round_trip_remains_json_text(self):
        expected = {"type": "error", "seq": 4, "code": "FAILED"}
        encoded = encode_response(expected)

        self.assertIsInstance(encoded, str)
        self.assertEqual(decode_response(encoded), {**expected, "v": VERSION})

    def test_response_requires_terminal_type_and_sequence(self):
        with self.assertRaisesRegex(ValueError, "type"):
            encode_response({"type": "frame", "seq": 1})
        with self.assertRaisesRegex(ValueError, "type"):
            encode_response({"type": "result", "seq": 1})
        with self.assertRaisesRegex(ValueError, "type"):
            encode_result({"type": "error", "seq": 1}, JPEG)
        with self.assertRaisesRegex(ValueError, "seq"):
            encode_response({"type": "error", "code": "FAILED"})
        with self.assertRaisesRegex(ValueError, "invalid seq"):
            decode_response(json.dumps({"v": VERSION, "type": "error"}))

    def test_response_size_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            encode_response({"type": "error", "seq": 1, "value": "x" * MAX_RESPONSE_BYTES})

    def test_result_rejects_invalid_envelopes(self):
        metadata = json.dumps(
            {"v": VERSION, "type": "result", "seq": 7},
            separators=(",", ":"),
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "header"):
            decode_result(b"ILR1")
        with self.assertRaisesRegex(ValueError, "magic"):
            decode_result(RESULT_HEADER.pack(b"OLD1", 7, len(metadata)) + metadata + JPEG)
        with self.assertRaisesRegex(ValueError, "metadata.*byte limit"):
            decode_result(RESULT_HEADER.pack(RESULT_MAGIC, 7, MAX_RESPONSE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_result(RESULT_HEADER.pack(RESULT_MAGIC, 7, len(metadata)) + metadata[:-1])

    def test_result_rejects_mismatched_sequence_and_invalid_jpeg(self):
        metadata = json.dumps(
            {"v": VERSION, "type": "result", "seq": 8},
            separators=(",", ":"),
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "do not match"):
            decode_result(RESULT_HEADER.pack(RESULT_MAGIC, 7, len(metadata)) + metadata + JPEG)
        with self.assertRaisesRegex(ValueError, "complete JPEG"):
            encode_result({"type": "result", "seq": 7}, b"not-a-jpeg")
        with self.assertRaisesRegex(ValueError, "byte limit"):
            decode_result(encode_result({"type": "result", "seq": 7}, JPEG), max_jpeg_bytes=4)

    def test_public_limits_are_fixed(self):
        self.assertEqual(VERSION, 2)
        self.assertEqual(MAX_JPEG_BYTES, 4 * 1024 * 1024)
        self.assertEqual(MAX_RESPONSE_BYTES, 512 * 1024)


if __name__ == "__main__":
    unittest.main()
