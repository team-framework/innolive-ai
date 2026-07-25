from __future__ import annotations

import unittest

import numpy as np

from scripts.benchmark_stream import (
    _contains_pixels,
    health_url,
    percentile,
    resize_long_edge,
)


class BenchmarkContractTests(unittest.TestCase):
    def test_resize_is_orientation_neutral_and_does_not_upscale(self):
        landscape = np.zeros((1080, 1920, 3), dtype=np.uint8)
        portrait = np.zeros((1920, 1080, 3), dtype=np.uint8)
        small = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(resize_long_edge(landscape).shape, (360, 640, 3))
        self.assertEqual(resize_long_edge(portrait).shape, (640, 360, 3))
        self.assertEqual(resize_long_edge(small).shape, (240, 320, 3))

    def test_metadata_guard_rejects_pixel_fields_recursively(self):
        self.assertFalse(_contains_pixels({"objects": [{"mask_polygon": []}]}))
        self.assertTrue(_contains_pixels({"objects": [{"jpeg": "no"}]}))
        self.assertTrue(_contains_pixels({"stats": {"pixels": 1}}))

    def test_percentile_and_health_url(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 95), 4)
        self.assertEqual(health_url("ws://localhost:8001/ws"), "http://localhost:8001/healthz")
        self.assertEqual(health_url("wss://example.test/ws"), "https://example.test/healthz")


if __name__ == "__main__":
    unittest.main()
