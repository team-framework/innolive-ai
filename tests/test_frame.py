from __future__ import annotations

import unittest
import zlib
from unittest.mock import patch

import cv2
import numpy as np

from service.frame import FrameLimits, decode_image, decode_jpeg


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


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return b"".join(
        (
            len(payload).to_bytes(4, "big"),
            chunk_type,
            payload,
            checksum.to_bytes(4, "big"),
        )
    )


def _png_header(width: int, height: int, *extra_chunks: bytes) -> bytes:
    header = b"".join(
        (
            width.to_bytes(4, "big"),
            height.to_bytes(4, "big"),
            b"\x08\x02\x00\x00\x00",
        )
    )
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            *extra_chunks,
            _png_chunk(b"IEND", b""),
        )
    )


class FrameBoundaryTests(unittest.TestCase):
    def test_enrollment_decoder_accepts_png_and_webp(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        for extension in (".png", ".webp"):
            with self.subTest(extension=extension):
                success, encoded = cv2.imencode(extension, image)
                self.assertTrue(success)
                decoded = decode_image(encoded.tobytes())
                self.assertEqual(decoded.shape, image.shape)

    def test_png_dimensions_are_rejected_before_decode(self):
        oversized = _png_header(1921, 640)
        with (
            patch("service.frame.cv2.imdecode") as decoder,
            self.assertRaisesRegex(ValueError, "long-edge 1920"),
        ):
            decode_image(oversized)

        decoder.assert_not_called()

    def test_animated_png_and_webp_are_rejected_before_decode(self):
        animated_png = _png_header(64, 48, _png_chunk(b"acTL", b"\x00" * 8))
        vp8x = bytes((0x02, 0, 0, 0, 63, 0, 0, 47, 0, 0))
        webp_body = b"WEBP" + b"VP8X" + len(vp8x).to_bytes(4, "little") + vp8x
        animated_webp = b"RIFF" + len(webp_body).to_bytes(4, "little") + webp_body

        for image in (animated_png, animated_webp):
            with (
                self.subTest(magic=image[:12]),
                patch("service.frame.cv2.imdecode") as decoder,
                self.assertRaisesRegex(ValueError, "animated"),
            ):
                decode_image(image)
            decoder.assert_not_called()

    def test_png_alpha_and_grayscale_are_normalized_to_three_channels(self):
        for image in (
            np.zeros((48, 64), dtype=np.uint8),
            np.zeros((48, 64, 4), dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape):
                success, encoded = cv2.imencode(".png", image)
                self.assertTrue(success)
                decoded = decode_image(encoded.tobytes())
                self.assertEqual(decoded.shape, (48, 64, 3))

    def test_declared_oversized_dimensions_are_rejected_before_decode(self):
        with (
            patch("service.frame.cv2.imdecode") as decoder,
            self.assertRaisesRegex(ValueError, "long-edge 1920"),
        ):
            decode_jpeg(_jpeg_header(640, 1921))

        decoder.assert_not_called()

    def test_fhd_dimensions_are_accepted(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)
        decoded = decode_jpeg(encoded.tobytes())
        self.assertEqual(decoded.shape, (1080, 1920, 3))

    def test_pixel_area_over_the_limit_is_rejected_before_decode(self):
        # A caller may cap area below max_long_edge**2 (e.g. FHD area). A frame
        # within the long edge but over that area is rejected before cv2 runs.
        limits = FrameLimits(max_long_edge=1920, max_pixels=1920 * 1080)
        with (
            patch("service.frame.cv2.imdecode") as decoder,
            self.assertRaisesRegex(ValueError, "pixel limit"),
        ):
            decode_jpeg(_jpeg_header(1920, 1200), limits)

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
