from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from ai_processor_server import AiProcessorServicer
from service.mosaic import mosaic_bgr
from service.runtime import RuntimeManager, _limit_polygon_points


def legacy_objects(polygons, tracks, width, height):
    """Freeze the scalar coordinate/rounding contract used before vectorized clipping."""
    objects = []
    for row in tracks:
        points = []
        if polygons is not None and int(row[-1]) < len(polygons):
            polygon = np.asarray(polygons[int(row[-1])], dtype=np.float32)
            if len(polygon):
                points = [
                    [
                        round(float(np.clip(x, 0, width - 1)), 1),
                        round(float(np.clip(y, 0, height - 1)), 1),
                    ]
                    for x, y in _limit_polygon_points(polygon)
                ]
        class_id = int(row[6])
        objects.append(
            {
                "track_id": int(row[4]),
                "class_id": class_id,
                "class_name": {0: "face", 1: "number_plate"}[class_id],
                "confidence": round(float(row[5]), 4),
                "bbox": [
                    round(float(np.clip(float(value), 0, bound)), 1)
                    for value, bound in zip(row[:4], (width - 1, height - 1) * 2, strict=True)
                ],
                "mask_polygon": points,
                "mask_area_px": round(
                    float(cv2.contourArea(np.asarray(points, dtype=np.float32)))
                    if len(points) >= 3
                    else 0.0,
                    1,
                ),
            }
        )
    return objects


class RuntimeObjectParityTests(unittest.TestCase):
    def setUp(self):
        self.runtime = RuntimeManager.__new__(RuntimeManager)
        self.runtime.names = {0: "face", 1: "number_plate"}

    def compare(self, polygons, tracks, width, height, *, render=False):
        prediction = SimpleNamespace(
            masks=None if polygons is None else SimpleNamespace(xy=polygons),
            boxes=[None] * len(tracks),
        )
        original = copy.deepcopy(polygons)
        expected = legacy_objects(polygons, tracks, width, height)
        actual = self.runtime._objects(prediction, tracks, width, height)
        self.assertEqual(actual, expected)
        if polygons is not None:
            for before, after in zip(original, polygons, strict=True):
                np.testing.assert_array_equal(before, after)
        for old, new in zip(expected, actual, strict=True):
            if len(old["mask_polygon"]) >= 3:
                self.assertEqual(
                    AiProcessorServicer._face_metadata(old).SerializeToString(deterministic=True),
                    AiProcessorServicer._face_metadata(new).SerializeToString(deterministic=True),
                )
        if render:
            image = np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)
            np.testing.assert_array_equal(mosaic_bgr(image, expected), mosaic_bgr(image, actual))

    def test_random_contours_preserve_metadata_bytes_and_mosaic_pixels(self):
        rng = np.random.default_rng(42)
        for width, height in ((99, 67), (640, 480), (1920, 1080)):
            for count in (1, 8, 32):
                with self.subTest(width=width, height=height, count=count):
                    polygons, tracks = [], []
                    for index in range(count):
                        angles = np.linspace(0, 2 * np.pi, (3, 64, 129)[index % 3], endpoint=False)
                        center = rng.uniform([0, 0], [width - 1, height - 1])
                        radius = rng.uniform([10, 10], [width / 3, height / 3])
                        polygon = (
                            center + np.column_stack((np.cos(angles), np.sin(angles))) * radius
                        )
                        polygons.append(polygon.astype(np.float32))
                        tracks.append(
                            [
                                *np.min(polygon, axis=0),
                                *np.max(polygon, axis=0),
                                index + 1,
                                0.98765,
                                index % 2,
                                index,
                            ]
                        )
                    self.compare(
                        polygons, np.asarray(tracks, dtype=np.float32), width, height, render=True
                    )

    def test_decimal_ties_nearby_floats_and_frame_edges_keep_python_rounding(self):
        values = np.asarray([-0.0, -0.05, 0.05, 1.05, 1.15, 2.55, 98.95, 99.05], np.float32)
        values = np.concatenate(
            (
                values,
                np.nextafter(values, np.float32(-np.inf)),
                np.nextafter(values, np.float32(np.inf)),
            )
        )
        polygon = np.column_stack((values, values[::-1]))
        tracks = np.asarray([[-2, -3, 100, 101, 1, 0.9, 0, 0]], np.float32)
        self.compare([polygon], tracks, 100, 100)

    def test_missing_empty_and_noncontiguous_masks_keep_the_contract(self):
        tracks = np.asarray([[1, 2, 10, 12, 1, 0.8, 0, 0]], np.float32)
        for polygons in (None, [], [np.empty((0, 2), np.float32)]):
            self.compare(polygons, tracks, 64, 64)
        coordinates = np.asarray(
            [[1.05, 0, 2.55, 0], [10.15, 0, 2.55, 0], [10.15, 0, 12.05, 0]], np.float32
        )[:, ::2]
        self.assertFalse(coordinates.flags.c_contiguous)
        self.compare([coordinates], tracks, 64, 64, render=True)


if __name__ == "__main__":
    unittest.main()
