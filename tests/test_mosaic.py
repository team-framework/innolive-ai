from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import service.mosaic as mosaic
from service.frame import raw_yuv420p_size
from service.mosaic import _feathered_mask, mosaic_jpeg, mosaic_yuv420p


def _decode(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("test JPEG decode failed")
    return image


def _full_frame_feathered_mask(mask: np.ndarray) -> np.ndarray:
    """Reference: the previous whole-frame feathering."""
    if not np.any(mask):
        return mask
    kernel_size = mosaic.MASK_FEATHER_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask, kernel)
    distance = cv2.distanceTransform(expanded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    feathered = np.rint(np.minimum(distance / mosaic.MASK_FEATHER_RADIUS, 1.0) * 255.0).astype(
        np.uint8
    )
    feathered[mask != 0] = 255
    return feathered


def _full_frame_mosaic(image, blend_mask, blur_radius, pixel_size):
    """Reference: the previous whole-frame protected-region bounds."""
    rows, columns = np.nonzero(blend_mask)
    if not rows.size:
        return image
    padding = math.ceil(blur_radius * 3)
    top = max(0, int(rows.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(image.shape[1], int(columns.max()) + padding + 1)
    region = image[top:bottom, left:right]
    reduced = cv2.resize(
        region,
        (
            max(1, math.ceil(region.shape[1] / pixel_size)),
            max(1, math.ceil(region.shape[0] / pixel_size)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    sigma = blur_radius / pixel_size
    blurred = cv2.resize(
        cv2.GaussianBlur(reduced, (0, 0), sigma, sigma),
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    output = image.copy()
    alpha = blend_mask[top:bottom, left:right, None].astype(np.uint32)
    output[top:bottom, left:right] = (
        region.astype(np.uint32) * (255 - alpha) + blurred.astype(np.uint32) * alpha + 127
    ) // 255
    return output


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

    def test_number_plate_is_blurred_even_if_whitelisted_metadata_is_wrong(self):
        output = _decode(
            mosaic_jpeg(
                self.image,
                [
                    {
                        "class_id": 1,
                        "class_name": "number_plate",
                        "whitelisted": True,
                        "mask_polygon": [[20, 20], [120, 20], [120, 90], [20, 90]],
                    }
                ],
            )
        )

        difference = np.abs(
            output[40:70, 40:100].astype(np.int16) - self.image[40:70, 40:100].astype(np.int16)
        )
        self.assertGreater(float(difference.mean()), 20)

    def test_feathered_mask_keeps_face_opaque_and_softens_only_its_outer_edge(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[25:56, 25:56] = 255

        feathered = _feathered_mask(mask)

        self.assertTrue(np.all(feathered[mask != 0] == 255))
        self.assertGreater(int(feathered[40, 23]), 0)
        self.assertLess(int(feathered[40, 23]), 255)
        self.assertGreater(int(feathered[40, 21]), 0)
        self.assertGreater(int(feathered[40, 20]), 0)
        self.assertLess(int(feathered[40, 20]), 255)
        self.assertEqual(int(feathered[40, 16]), 0)

    def test_feathered_mask_matches_full_frame_computation(self):
        rng = np.random.default_rng(52)
        for height, width in ((720, 1280), (1280, 720), (120, 160), (9, 9)):
            for _ in range(40):
                mask = np.zeros((height, width), dtype=np.uint8)
                for _ in range(int(rng.integers(1, 4))):
                    center = (
                        int(rng.integers(-20, width + 20)),
                        int(rng.integers(-20, height + 20)),
                    )
                    axes = (
                        int(rng.integers(0, max(1, width // 4))),
                        int(rng.integers(0, max(1, height // 4))),
                    )
                    cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
                with self.subTest(size=(height, width), pixels=int(np.count_nonzero(mask))):
                    np.testing.assert_array_equal(
                        _feathered_mask(mask), _full_frame_feathered_mask(mask)
                    )

    def test_feathered_mask_skips_pixels_far_from_the_protected_region(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[300:420, 600:690] = 255

        with patch("service.mosaic.cv2.distanceTransform", wraps=cv2.distanceTransform) as distance:
            _feathered_mask(mask)

        margin = mosaic.MASK_FEATHER_RADIUS + 1
        self.assertEqual(distance.call_args.args[0].shape, (120 + 2 * margin, 90 + 2 * margin))

    def test_mosaic_matches_full_frame_mask_computation(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        for objects in (
            [{"mask_polygon": [[300, 120], [380, 130], [370, 240], [290, 230]]}],
            [{"mask_polygon": [[0, 0], [60, 0], [60, 80], [0, 80]]}],
            [
                {"mask_polygon": [[20, 200], [90, 210], [80, 330], [10, 320]]},
                {"mask_polygon": [[560, 20], [639, 30], [630, 140], [550, 130]]},
            ],
            [],
        ):
            mask = mosaic._protected_mask(image.shape[:2], objects)
            expected = _full_frame_mosaic(
                image, _full_frame_feathered_mask(mask), mosaic.DEFAULT_BLUR_RADIUS, 2
            )
            with self.subTest(objects=len(objects)):
                self.assertEqual(mosaic_yuv420p(image, objects), mosaic._encode_yuv420p(expected))

    def test_protected_blur_tapers_into_the_scene_outside_the_face_mask(self):
        output = _decode(
            mosaic_jpeg(
                self.image,
                [
                    {
                        "whitelisted": False,
                        "mask_polygon": [[40, 30], [120, 30], [120, 90], [40, 90]],
                    }
                ],
            )
        )

        outer_edge_difference = np.abs(
            output[45:75, 38:40].astype(np.int16) - self.image[45:75, 38:40].astype(np.int16)
        )
        self.assertGreater(float(outer_edge_difference.mean()), 2)

        scene_difference = np.abs(
            output[45:75, 28:32].astype(np.int16) - self.image[45:75, 28:32].astype(np.int16)
        )
        self.assertLess(float(scene_difference.mean()), 5)

    def test_invalid_protected_polygon_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "invalid mask polygon"):
            mosaic_jpeg(
                self.image,
                [{"whitelisted": False, "mask_polygon": [[1, 1], [2, 2]]}],
            )


class MosaicParamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        for row in range(0, 120, 16):
            for column in range(0, 160, 16):
                if ((row // 16) + (column // 16)) % 2:
                    self.image[row : row + 16, column : column + 16] = 255
        self.objects = [
            {
                "whitelisted": False,
                "mask_polygon": [[40, 30], [120, 30], [120, 90], [40, 90]],
            }
        ]

    def test_defaults_are_stronger_than_the_legacy_sigma(self):
        self.assertEqual(mosaic.DEFAULT_BLUR_RADIUS, 24.0)
        self.assertEqual(mosaic.DEFAULT_PIXEL_SIZE, 2)

    def test_custom_strength_changes_the_output(self):
        weak = _decode(
            mosaic_jpeg(
                self.image,
                self.objects,
                blur_radius=8.0,
                pixel_size=1,
            )
        )
        default = _decode(mosaic_jpeg(self.image, self.objects))

        original = self.image[40:80, 40:120].astype(np.int16)
        weak_diff = float(np.abs(weak[40:80, 40:120].astype(np.int16) - original).mean())
        default_diff = float(np.abs(default[40:80, 40:120].astype(np.int16) - original).mean())
        # Both settings alter the protected region ...
        self.assertGreater(weak_diff, 20)
        self.assertGreater(default_diff, 20)
        # ... and the stronger default flattens the pattern more than the weak one.
        weak_std = float(weak[40:80, 40:120].astype(np.int16).std())
        default_std = float(default[40:80, 40:120].astype(np.int16).std())
        self.assertGreater(default_diff, weak_diff)
        self.assertLess(default_std, weak_std)

    def test_pixel_size_one_keeps_pure_gaussian_blur(self):
        with patch("service.mosaic.cv2.GaussianBlur", wraps=cv2.GaussianBlur) as blur:
            output = _decode(mosaic_jpeg(self.image, self.objects, blur_radius=12.0, pixel_size=1))

        self.assertEqual(blur.call_count, 1)
        args, _ = blur.call_args
        self.assertAlmostEqual(float(args[2]), 12.0)
        difference = np.abs(
            output[40:80, 40:120].astype(np.int16) - self.image[40:80, 40:120].astype(np.int16)
        )
        self.assertGreater(float(difference.mean()), 20)

    def test_invalid_strength_fails_closed(self):
        cases = (
            ({"blur_radius": 0.0}, "blur_radius"),
            ({"blur_radius": mosaic.MAX_BLUR_RADIUS + 1}, "blur_radius"),
            ({"blur_radius": float("nan")}, "blur_radius"),
            ({"blur_radius": float("inf")}, "blur_radius"),
            ({"pixel_size": 0}, "pixel_size"),
            ({"pixel_size": mosaic.MAX_PIXEL_SIZE + 1}, "pixel_size"),
            ({"pixel_size": 2.5}, "pixel_size"),
        )
        for kwargs, expected in cases:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, expected):
                mosaic_jpeg(self.image, self.objects, **kwargs)


class MosaicRawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.image[:, ::2] = 255

    def test_raw_output_has_expected_yuv420p_size(self):
        payload = mosaic_yuv420p(
            self.image,
            [{"whitelisted": False, "mask_polygon": [[20, 20], [90, 20], [90, 100], [20, 100]]}],
        )
        self.assertEqual(len(payload), raw_yuv420p_size(160, 120))

    def test_raw_output_rejects_odd_dimensions(self):
        odd = np.zeros((120, 161, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            mosaic_yuv420p(odd, [])


if __name__ == "__main__":
    unittest.main()
