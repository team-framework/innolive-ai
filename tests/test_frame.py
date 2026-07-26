from __future__ import annotations

import unittest
from unittest.mock import patch

from service.frame import FrameLimits, decode_jpeg


def _jpeg_header(width: int, height: int) -> bytes:
    return b"".join(
        (
            b"\xff\xd8",
            b"\xff\xc0",
            (11).to_bytes(2, "big"),
            b"\x08",
            height.to_bytes(2, "big"),
            width.to_bytes(2, "big"),
            b"\x01\x01\x11\x00",
            b"\xff\xd9",
        )
    )


class FrameBoundaryTests(unittest.TestCase):
    def test_declared_oversized_dimensions_are_rejected_before_decode(self):
        with (
            patch("service.frame.cv2.imdecode") as decoder,
            self.assertRaisesRegex(ValueError, "long-edge 640"),
        ):
            decode_jpeg(_jpeg_header(640, 641))

        decoder.assert_not_called()

    def test_missing_or_truncated_frame_header_is_rejected(self):
        malformed = b"\xff\xd8\xff\xe0\x00\x10too-short\xff\xd9"
        with self.assertRaises(ValueError):
            decode_jpeg(malformed)

    def test_frame_limit_configuration_cannot_disable_dimension_safety(self):
        with self.assertRaises(ValueError):
            FrameLimits(min_dimension=641, max_long_edge=640)
        with self.assertRaises(ValueError):
            FrameLimits(max_pixels=0)


if __name__ == "__main__":
    unittest.main()
