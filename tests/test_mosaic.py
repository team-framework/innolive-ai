from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from service.mosaic import mosaic_jpeg


def _decode(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("test JPEG decode failed")
    return image


class MosaicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.image[:, ::2] = 255

    def test_non_whitelisted_masks_are_unioned_and_blurred_once(self):
        objects = [
            {
                "whitelisted": False,
                "mask_polygon": [[20, 20], [90, 20], [90, 100], [20, 100]],
            },
            {
                "whitelisted": None,
                "mask_polygon": [[70, 20], [140, 20], [140, 100], [70, 100]],
            },
        ]

        with patch("service.mosaic.cv2.GaussianBlur", wraps=cv2.GaussianBlur) as blur:
            output = _decode(mosaic_jpeg(self.image, objects))

        self.assertEqual(blur.call_count, 1)
        for left, right in ((30, 60), (100, 130)):
            difference = np.abs(
                output[40:80, left:right].astype(np.int16)
                - self.image[40:80, left:right].astype(np.int16)
            )
            self.assertGreater(float(difference.mean()), 20)

    def test_whitelisted_mask_never_subtracts_from_protected_union(self):
        objects = [
            {
                "whitelisted": True,
                "mask_polygon": [[40, 30], [120, 30], [120, 90], [40, 90]],
            },
            {
                "whitelisted": False,
                "mask_polygon": [[70, 40], [140, 40], [140, 100], [70, 100]],
            },
        ]

        output = _decode(mosaic_jpeg(self.image, objects))

        protected_difference = np.abs(
            output[55:85, 80:110].astype(np.int16) - self.image[55:85, 80:110].astype(np.int16)
        )
        self.assertGreater(float(protected_difference.mean()), 20)

    def test_no_protected_faces_skips_blur_and_returns_jpeg(self):
        with patch("service.mosaic.cv2.GaussianBlur", wraps=cv2.GaussianBlur) as blur:
            payload = mosaic_jpeg(
                self.image,
                [
                    {
                        "whitelisted": True,
                        "mask_polygon": [[20, 20], [80, 20], [80, 80], [20, 80]],
                    }
                ],
            )

        self.assertEqual(blur.call_count, 0)
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        self.assertTrue(payload.endswith(b"\xff\xd9"))

    def test_invalid_protected_polygon_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid mask polygon"):
            mosaic_jpeg(
                self.image,
                [{"whitelisted": False, "mask_polygon": [[1, 1], [2, 2]]}],
            )


if __name__ == "__main__":
    unittest.main()
