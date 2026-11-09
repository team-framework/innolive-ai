"""RF-DETR model-independent decoding and tracker lifecycle contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from service.detection import Boxes
from service.rfdetr import RESOLUTION, decode, load_checkpoint_model
from service.tracking import StreamTracker


class RFDETRDecodeTests(unittest.TestCase):
    def outputs(self):
        boxes = np.asarray([[[0.5, 0.5, 0.4, 0.4], [0.25, 0.25, 0.2, 0.2]]], np.float32)
        logits = np.asarray([[[4, -20, 30], [-20, 3, 30]]], np.float32)
        masks = np.full((1, 2, RESOLUTION // 4, RESOLUTION // 4), -10, np.float32)
        masks[0, 0, 47:109, 47:109] = 10
        masks[0, 1, 23:55, 23:55] = 10
        return boxes, logits, masks

    def test_background_is_excluded_and_public_plate_id_preserved(self):
        result = decode(*self.outputs(), (1080, 1920, 3), threshold=0.25)
        np.testing.assert_array_equal(result.boxes.cls, [0, 1])
        np.testing.assert_allclose(result.boxes.xyxy[0], [576, 324, 1344, 756], atol=0.001)
        self.assertEqual(len(result.masks.xy), 2)
        self.assertGreater(result.masks.xy[0][:, 0].max(), 1300)

    def test_no_foreground_returns_empty_shapes(self):
        boxes, logits, masks = self.outputs()
        logits[..., :2] = -20
        result = decode(boxes, logits, masks, (480, 640, 3), threshold=0.25)
        self.assertEqual(result.boxes.data.shape, (0, 6))
        self.assertEqual(result.masks.xy, [])

    def test_ties_have_stable_query_class_order_and_count_is_bounded(self):
        boxes, logits, masks = self.outputs()
        logits[..., :2] = 1
        result = decode(boxes, logits, masks, (624, 624, 3), max_detections=3)
        np.testing.assert_array_equal(result.boxes.cls, [0, 1, 0])

    def test_malformed_mask_contract_is_rejected(self):
        boxes, logits, masks = self.outputs()
        with self.assertRaisesRegex(ValueError, "output shapes"):
            decode(boxes, logits, masks[:, :, :10], (480, 640, 3))

    def test_disconnected_mask_components_are_all_protected(self):
        boxes, logits, masks = self.outputs()
        masks[0, 0, 10:15, 10:15] = 10
        result = decode(boxes, logits, masks, (624, 624, 3), threshold=0.25)
        self.assertLess(result.masks.xy[0][:, 0].min(), 60)
        self.assertGreater(result.masks.xy[0][:, 0].max(), 420)

    def test_lazy_contours_own_planes_after_next_inference_reuses_buffers(self):
        boxes, logits, masks = self.outputs()
        result = decode(boxes, logits, masks, (624, 624, 3), threshold=0.25)
        self.assertEqual(result.masks.xy.cache, {})
        masks.fill(-10)
        self.assertGreater(len(result.masks.xy[0]), 3)
        self.assertEqual(len(result.masks.xy.cache), 1)


class RFDETRLoadTests(unittest.TestCase):
    def test_checkpoint_load_preserves_process_precision_on_success_and_failure(self):
        import torch

        previous = torch.get_float32_matmul_precision()
        try:
            for original in ("highest", "medium"):
                for failure in (False, True):
                    with self.subTest(original=original, failure=failure):
                        torch.set_float32_matmul_precision(original)

                        def load(*args, failure=failure, **kwargs):
                            if failure:
                                raise ValueError("invalid checkpoint")
                            return "model"

                        def import_rfdetr(name):
                            torch.set_float32_matmul_precision("high")
                            return SimpleNamespace(RFDETR=SimpleNamespace(from_checkpoint=load))

                        with patch("service.rfdetr.import_module", side_effect=import_rfdetr):
                            if failure:
                                with self.assertRaisesRegex(ValueError, "invalid checkpoint"):
                                    load_checkpoint_model("model.pth", "cpu")
                            else:
                                self.assertEqual(load_checkpoint_model("model.pth", "cpu"), "model")
                        self.assertEqual(torch.get_float32_matmul_precision(), original)
        finally:
            torch.set_float32_matmul_precision(previous)


class IndependentBoTSORTTests(unittest.TestCase):
    image = np.zeros((160, 160, 3), np.uint8)

    def test_low_confidence_continues_but_cannot_create_id(self):
        tracker = StreamTracker(gmc_method="none")
        self.assertEqual(len(tracker.update(Boxes([[10, 10, 90, 90, 0.1, 0]]), self.image)), 0)
        self.assertEqual(len(tracker.update(Boxes([[10, 10, 90, 90, 0.9, 0]]), self.image)), 0)
        first = tracker.update(Boxes([[10, 10, 90, 90, 0.9, 0]]), self.image)
        continued = tracker.update(Boxes([[12, 10, 92, 90, 0.1, 0]]), self.image)
        self.assertEqual(first[0, 4], continued[0, 4])
        self.assertAlmostEqual(float(continued[0, 5]), 0.1, places=5)

    def test_plate_never_inherits_face_identity(self):
        tracker = StreamTracker(gmc_method="none")
        face = tracker.update(Boxes([[10, 10, 90, 90, 0.9, 0]]), self.image)
        tracker.update(Boxes([[10, 10, 90, 90, 0.9, 1]]), self.image)
        plate = tracker.update(Boxes([[10, 10, 90, 90, 0.9, 1]]), self.image)
        self.assertNotEqual(face[0, 4], plate[0, 4])
        self.assertEqual(plate[0, 6], 1)

    def test_expired_id_is_not_reactivated_and_reset_restarts_stream(self):
        tracker = StreamTracker(gmc_method="none")
        row = Boxes([[10, 10, 90, 90, 0.9, 0]])
        old = tracker.update(row, self.image)[0, 4]
        for _ in range(32):
            tracker.update(Boxes([]), self.image)
        self.assertFalse(tracker.live_track_ids)
        tracker.update(row, self.image)
        new = tracker.update(row, self.image)[0, 4]
        self.assertNotEqual(old, new)
        tracker.reset()
        self.assertEqual(tracker.update(row, self.image)[0, 4], 1)

    def test_reset_refits_camera_motion_resolution(self):
        tracker = StreamTracker()
        tracker.estimate_motion(np.zeros((1080, 1920, 3), np.uint8))
        self.assertEqual(tracker._tracker.gmc.downscale, 3)
        tracker.reset()
        tracker.estimate_motion(np.zeros((1080, 1920, 3), np.uint8))
        self.assertEqual(tracker._tracker.gmc.downscale, 3)
