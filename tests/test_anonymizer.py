from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from service.anonymizer import (
    DetectionBatch,
    HeadAnonymizerStream,
    _TrackedMask,
    resolve_model_path,
)


class ResolveModelPathTest(unittest.TestCase):
    def test_prefers_only_a_fresh_tensorrt_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            pt_path = Path(directory) / "best.pt"
            engine_path = Path(directory) / "best.engine"
            pt_path.touch()
            engine_path.touch()

            os.utime(pt_path, (100, 100))
            os.utime(engine_path, (200, 200))
            self.assertEqual(resolve_model_path(pt_path), engine_path.resolve())

            os.utime(pt_path, (300, 300))
            self.assertEqual(resolve_model_path(pt_path), pt_path.resolve())

    def test_missing_engine_falls_back_to_pt(self):
        with tempfile.TemporaryDirectory() as directory:
            engine_path = Path(directory) / "best.engine"
            pt_path = engine_path.with_suffix(".pt")
            pt_path.touch()
            self.assertEqual(resolve_model_path(engine_path), pt_path.resolve())


class MaskWarpTest(unittest.TestCase):
    def test_warp_mask_follows_predicted_box(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        warped = HeadAnonymizerStream._warp_mask(
            mask,
            np.asarray([10, 10, 20, 20], dtype=np.float32),
            np.asarray([20, 15, 40, 35], dtype=np.float32),
            (50, 50),
        )
        self.assertGreater(float(warped[20:30, 25:35].mean()), 0.9)
        self.assertEqual(float(warped[:10, :10].max()), 0.0)

    def test_lost_track_mask_is_held_for_configured_frames(self):
        stream = object.__new__(HeadAnonymizerStream)
        stream.config = SimpleNamespace(
            mask_hold_frames=2,
            unconfirmed_hold_frames=2,
            temporal_decay=0.70,
        )
        stream.runtime = SimpleNamespace(
            _tracker_args=SimpleNamespace(new_track_thresh=0.25)
        )
        cached_mask = np.ones((20, 20), dtype=np.float32)
        bbox = np.asarray([2, 2, 18, 18], dtype=np.float32)
        stream._tracked_masks = {7: _TrackedMask(cached_mask, bbox)}
        stream._unconfirmed_masks = []
        stream.tracker = SimpleNamespace(
            lost_stracks=[SimpleNamespace(track_id=7, xyxy=bbox)]
        )
        detections = DetectionBatch(
            boxes=SimpleNamespace(
                conf=np.empty(0, dtype=np.float32),
                xyxy=np.empty((0, 4), dtype=np.float32),
            ),
            masks=np.empty((0, 20, 20), dtype=np.float32),
        )
        no_tracks = np.empty((0, 8), dtype=np.float32)

        self.assertEqual(len(stream._stable_masks(detections, no_tracks, (20, 20))), 1)
        self.assertEqual(len(stream._stable_masks(detections, no_tracks, (20, 20))), 1)
        self.assertEqual(len(stream._stable_masks(detections, no_tracks, (20, 20))), 0)
        self.assertNotIn(7, stream._tracked_masks)

    def test_unconfirmed_mask_bridges_bot_sort_confirmation_gap(self):
        stream = object.__new__(HeadAnonymizerStream)
        stream.config = SimpleNamespace(
            unconfirmed_hold_frames=2,
            temporal_decay=0.70,
        )
        stream._unconfirmed_masks = []
        mask = np.ones((20, 20), dtype=np.float32)
        bbox = np.asarray([2, 2, 18, 18], dtype=np.float32)

        rendered: list[np.ndarray] = []
        stream._hold_unconfirmed_masks(
            [_TrackedMask(mask, bbox)], [], rendered, (20, 20)
        )
        self.assertEqual(len(rendered), 1)

        for expected in (1, 1, 0):
            rendered = []
            stream._hold_unconfirmed_masks([], [], rendered, (20, 20))
            self.assertEqual(len(rendered), expected)


if __name__ == "__main__":
    unittest.main()
