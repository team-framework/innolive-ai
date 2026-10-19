from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
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


def moving_scene(frames: int = 40):
    """Textured frames panning like a hand-held camera, with one face box following the scene."""
    rng = np.random.default_rng(54)
    texture = rng.integers(0, 256, (900, 900), dtype=np.uint8)
    texture = np.repeat(cv2.GaussianBlur(texture, (0, 0), 2.0)[:, :, None], 3, axis=2)
    for index in range(frames):
        dx = round(60 + 25 * np.sin(index / 6))
        dy = round(60 + 15 * np.cos(index / 9))
        image = np.ascontiguousarray(texture[dy : dy + 640, dx : dx + 640])
        x, y = 200 - dx, 220 - dy
        detections = (
            boxes((x + 100, y + 80, x + 220, y + 230, 0.9, 0)) if index % 7 != 3 else boxes()
        )
        yield image, detections


class PrecomputedMotionTests(unittest.TestCase):
    def test_precomputed_motion_matches_sequential_update(self):
        sequential, precomputed = StreamTracker(), StreamTracker()
        self.assertTrue(precomputed.motion_precomputable)
        moved = False
        for image, detections in moving_scene():
            expected = sequential.update(detections, image)
            motion = precomputed.estimate_motion(image)
            moved = moved or not np.allclose(motion, np.eye(2, 3))
            actual = precomputed.update(detections, image, motion=motion)
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(moved, "test scene must exercise a non-identity GMC warp")

    def test_failed_precomputed_motion_falls_back_to_identity_like_the_base_tracker(self):
        sequential, precomputed = StreamTracker(), StreamTracker()
        for index, (image, detections) in enumerate(moving_scene(12)):
            if index == 6:
                with patch.object(
                    sequential._tracker.gmc, "apply", side_effect=RuntimeError("gmc")
                ):
                    expected = sequential.update(detections, image)
                actual = precomputed.update(detections, image, motion=RuntimeError("gmc"))
            else:
                expected = sequential.update(detections, image)
                actual = precomputed.update(
                    detections, image, motion=precomputed.estimate_motion(image)
                )
            np.testing.assert_array_equal(actual, expected)

    def test_detection_dependent_gmc_is_not_precomputed(self):
        self.assertFalse(StreamTracker(gmc_method="orb").motion_precomputable)
        self.assertTrue(StreamTracker(gmc_method="none").motion_precomputable)

    def test_update_without_motion_keeps_the_base_gmc_path(self):
        tracker = StreamTracker()
        image, detections = next(moving_scene(1))
        with patch.object(tracker._tracker.gmc, "apply", wraps=tracker._tracker.gmc.apply) as apply:
            tracker.update(detections, image)
        self.assertEqual(apply.call_count, 1)

    def test_gmc_downscale_follows_the_first_frame_long_edge(self):
        for (height, width), expected in (
            ((480, 640), 2),
            ((720, 1280), 2),
            ((1080, 1920), 3),
            ((1920, 1080), 3),
        ):
            with self.subTest(size=(width, height)):
                image = np.zeros((height, width, 3), dtype=np.uint8)
                precomputed, sequential = StreamTracker(), StreamTracker()
                precomputed.estimate_motion(image)
                sequential.update(boxes(), image)
                self.assertEqual(precomputed._tracker.gmc.downscale, expected)
                self.assertEqual(sequential._tracker.gmc.downscale, expected)
                # 스트림 중간 프레임은 첫 프레임 배율을 유지한다.
                precomputed.estimate_motion(np.zeros((360, 640, 3), dtype=np.uint8))
                self.assertEqual(precomputed._tracker.gmc.downscale, expected)

    def test_fhd_gmc_restores_full_resolution_translation(self):
        texture = cv2.GaussianBlur(
            np.random.default_rng(7).integers(0, 256, (1200, 2040), dtype=np.uint8), (0, 0), 2
        )
        tracker = StreamTracker()
        first = cv2.cvtColor(texture[60:1140, 60:1980], cv2.COLOR_GRAY2BGR)
        second = cv2.cvtColor(texture[51:1131, 72:1992], cv2.COLOR_GRAY2BGR)
        tracker.estimate_motion(first)
        warp = tracker.estimate_motion(second)
        self.assertEqual(tracker._tracker.gmc.downscale, 3)
        self.assertIsInstance(warp, np.ndarray)
        np.testing.assert_allclose(warp[:, 2], (-12, 9), atol=1.5)


if __name__ == "__main__":
    unittest.main()
