from __future__ import annotations

import unittest

import numpy as np
from ultralytics.engine.results import Boxes

from service.tracking import (
    ACTIVATION_CONFIDENCE,
    CONTINUATION_CONFIDENCE,
    DETECTOR_CONFIDENCE,
    HOLD_CONFIDENCE_DECAY,
    MAX_MASK_HOLD_FRAMES,
    StreamTracker,
)


def boxes(*rows: tuple[float, float, float, float, float, float]) -> Boxes:
    values = np.asarray(rows, dtype=np.float32).reshape((-1, 6))
    return Boxes(values, orig_shape=(640, 640)).cpu().numpy()


class TrackingTests(unittest.TestCase):
    def test_fixed_threshold_contract(self):
        self.assertEqual(DETECTOR_CONFIDENCE, 0.01)
        self.assertEqual(CONTINUATION_CONFIDENCE, 0.05)
        self.assertEqual(ACTIVATION_CONFIDENCE, 0.25)
        self.assertEqual(MAX_MASK_HOLD_FRAMES, 1)
        self.assertEqual(HOLD_CONFIDENCE_DECAY, 0.90)

    def test_each_connection_has_an_independent_id_counter(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        first = StreamTracker().update(boxes((20, 20, 100, 100, 0.95, 0)), image)
        second = StreamTracker().update(boxes((200, 200, 280, 280, 0.95, 0)), image)
        self.assertEqual(int(first[0, 4]), 1)
        self.assertEqual(int(second[0, 4]), 1)

    def test_detector_mask_is_held_for_exactly_one_missing_frame(self):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        tracker = StreamTracker()
        tracked = tracker.update(boxes((20, 20, 100, 100, 0.90, 0)), image)
        track_id = int(tracked[0, 4])
        detected, _ = tracker.stabilize(
            [
                {
                    "track_id": track_id,
                    "class_id": 0,
                    "class_name": "face",
                    "confidence": 0.90,
                    "bbox": [20, 20, 100, 100],
                    "mask_polygon": [[20, 20], [100, 20], [100, 100], [20, 100]],
                    "mask_area_px": 6400,
                }
            ],
            640,
            640,
        )
        self.assertEqual(detected[0]["source"], "detected")

        empty = boxes()
        tracker.update(empty, image)
        held, metrics = tracker.stabilize([], 640, 640)
        self.assertEqual(metrics["held_tracks"], 1)
        self.assertEqual(held[0]["source"], "held")
        self.assertEqual(held[0]["hold_frames"], 1)
        self.assertAlmostEqual(held[0]["confidence"], 0.81)

        tracker.update(empty, image)
        expired, metrics = tracker.stabilize([], 640, 640)
        self.assertEqual(expired, [])
        self.assertEqual(metrics["held_tracks"], 0)


if __name__ == "__main__":
    unittest.main()
