from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from service.face_metadata import (
    CLASSES,
    MetadataExtractor,
    crop_metadata_face,
    resize_metadata_crop,
)


class MetadataCropTests(unittest.TestCase):
    def test_margin_uses_original_pixels_on_all_four_sides(self):
        image = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
        crop = crop_metadata_face(image, np.array([30, 20, 70, 40]))
        np.testing.assert_array_equal(crop, image[17:43, 24:76])

    def test_margin_is_clipped_at_image_edges(self):
        image = np.arange(40 * 50 * 3, dtype=np.uint8).reshape(40, 50, 3)
        crop = crop_metadata_face(image, np.array([0, 0, 10, 20]))
        np.testing.assert_array_equal(crop, image[:23, :12])

    def test_invalid_boxes_are_rejected(self):
        image = np.zeros((40, 50, 3), np.uint8)
        for box in ([1, 1, 1, 10], [10, 0, 1, 10], [0, 0, np.nan, 10], [0, 0, 1], [80, 0, 90, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                crop_metadata_face(image, np.array(box))

    def test_tall_crop_preserves_width_and_replicates_side_edges(self):
        crop = np.zeros((8, 4, 3), np.uint8)
        crop[:, 0] = (10, 20, 30)
        crop[:, -1] = (90, 100, 110)
        result = resize_metadata_crop(crop, 224)
        self.assertEqual(result.shape, (224, 224, 3))
        np.testing.assert_array_equal(result[:, :56], np.broadcast_to(crop[0, 0], (224, 56, 3)))
        np.testing.assert_array_equal(result[:, 168:], np.broadcast_to(crop[0, -1], (224, 56, 3)))
        expected = cv2.resize(crop, (112, 224), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(result[:, 56:168], expected)

    def test_downscale_uses_area_and_odd_padding_is_centered(self):
        crop = np.zeros((5, 10, 3), np.uint8)
        crop[:, ::2] = 255
        result = resize_metadata_crop(crop, 5)
        expected = cv2.resize(crop, (5, 2), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(result[1:3], expected)
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[3:], np.broadcast_to(expected[-1], (2, 5, 3)))

    def test_empty_wrong_dtype_and_wrong_channels_are_rejected(self):
        for crop in (
            np.zeros((0, 4, 3), np.uint8),
            np.zeros((4, 4, 3), np.float32),
            np.zeros((4, 4), np.uint8),
            np.zeros((4, 4, 4), np.uint8),
        ):
            with self.subTest(shape=crop.shape), self.assertRaises(ValueError):
                resize_metadata_crop(crop, 224)


class MetadataModelTests(unittest.TestCase):
    def setUp(self):
        # A tiny backbone isolates head behavior; the real model is checked by source parity.
        features = nn.Sequential(nn.Conv2d(3, 960, 1, stride=224))
        state = {f"features.{name}": value for name, value in features.state_dict().items()}
        for name, classes in CLASSES.items():
            state[f"heads.{name}.0.weight"] = torch.zeros(128, 960)
            state[f"heads.{name}.0.bias"] = torch.full((128,), -1.0)
            state[f"heads.{name}.3.weight"] = torch.zeros(len(classes), 128)
            state[f"heads.{name}.3.weight"][0] = 1 / 128
            state[f"heads.{name}.3.bias"] = torch.zeros(len(classes))
        checkpoint = {
            "config": {
                "architecture": "torchvision.mobilenet_v3_large.features + GAP",
                "targets": list(CLASSES),
                "classes": CLASSES,
                "input_size": 224,
            },
            "model": state,
        }
        with (
            patch("torch.load", return_value=checkpoint),
            patch(
                "torchvision.models.mobilenet_v3_large",
                return_value=SimpleNamespace(features=features),
            ),
        ):
            self.extractor = MetadataExtractor(Path("fixture.pt"))

    def test_negative_hidden_values_use_hardswish_even_when_strict_loading_succeeds(self):
        with torch.inference_mode():
            outputs = self.extractor.model(torch.zeros(1, 3, 224, 224))
        for values in outputs.values():
            self.assertAlmostEqual(float(values[0, 0]), -1 / 3, places=6)
            self.assertTrue(torch.equal(values[0, 1:], torch.zeros_like(values[0, 1:])))

    def test_inputs_match_rgb_imagenet_normalization_in_channels_last_batch(self):
        crop = np.full((10, 5, 3), (0, 0, 255), np.uint8)
        inputs = self.extractor.prepare_inputs([crop, crop])
        self.assertEqual(tuple(inputs.shape), (2, 3, 224, 224))
        self.assertTrue(inputs.is_contiguous(memory_format=torch.channels_last))
        expected = (
            torch.tensor([1.0, 0.0, 0.0]) - self.extractor.mean[:, 0, 0]
        ) / self.extractor.std[:, 0, 0]
        torch.testing.assert_close(inputs[0, :, 0, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)

    def test_empty_prediction_does_not_run_model(self):
        self.assertEqual(self.extractor.predict([]), [])


if __name__ == "__main__":
    unittest.main()
