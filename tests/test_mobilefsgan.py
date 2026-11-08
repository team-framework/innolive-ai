from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from threading import RLock
from unittest.mock import Mock, patch

import cv2
import numpy as np

from service.face_anonymization import FaceAnonymizationConfig, FaceAnonymizationRuntime
from service.mobilefsgan import (
    ALPHA,
    TEMPLATE,
    MobileFSGANFrame,
    MobileFSGANRenderer,
    alignment,
    crop_face,
    image_tensor,
    paste_back,
    validate_bundle,
)
from service.yunet_tensorrt import decode


class MobileFSGANTests(unittest.TestCase):
    def test_yunet_vector_decoder_preserves_score_coordinates_and_nms(self):
        raw = {}
        for stride in (8, 16, 32):
            cells = (32 // stride) ** 2
            raw[f"cls_{stride}"] = np.zeros((1, cells, 1), np.float32)
            raw[f"obj_{stride}"] = np.ones((1, cells, 1), np.float32)
            raw[f"bbox_{stride}"] = np.zeros((1, cells, 4), np.float32)
            raw[f"kps_{stride}"] = np.zeros((1, cells, 10), np.float32)
        self.assertEqual(decode(raw, 32, 32).shape, (0, 15))
        raw["cls_8"][0, 5, 0] = 0.81
        faces = decode(raw, 32, 32)
        self.assertEqual(faces.shape, (1, 15))
        np.testing.assert_allclose(faces[0, :4], [4, 4, 8, 8])
        np.testing.assert_allclose(faces[0, 4:14], [8] * 10)
        self.assertAlmostEqual(float(faces[0, 14]), 0.9, places=6)
        with self.assertRaisesRegex(ValueError, "size"):
            decode(raw, 64, 32)

    def test_source_tensor_keeps_bundle_rgb_range_and_256_crop(self):
        crop = np.full((256, 256, 3), [0, 127, 255], np.uint8)
        values = image_tensor(crop)
        self.assertEqual(values.shape, (3, 256, 256))
        np.testing.assert_allclose(values[:, 0, 0], [1, np.float32(127 / 127.5 - 1), -1], atol=1e-7)
        self.assertTrue(values.flags.c_contiguous)

    def test_similarity_and_reflected_crop_match_identity(self):
        matrix = alignment(TEMPLATE)
        np.testing.assert_allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-5)
        image = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        np.testing.assert_array_equal(crop_face(image, matrix), image)
        for points in (np.zeros((5, 2)), np.full((5, 2), np.nan), np.zeros((4, 2))):
            with self.assertRaises(ValueError):
                alignment(points)

    def test_roi_blend_matches_full_frame_reference_and_preserves_outside_polygon(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        generated = rng.uniform(-1, 1, (3, 256, 256)).astype(np.float32)
        matrix = np.array([[0.8, 0.0, -40], [0, 0.8, -30]], np.float32)
        polygon = [[60, 40], [330, 75], [380, 320], [80, 350]]
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.array(polygon)], 1)
        inverse = cv2.invertAffineTransform(matrix)
        fake = (
            np.rint((generated.transpose(1, 2, 0) + 1) * 127.5)
            .clip(0, 255)
            .astype(np.uint8)[:, :, ::-1]
        )
        placed = cv2.warpAffine(fake, inverse, (640, 480))
        alpha = (cv2.warpAffine(ALPHA, inverse, (640, 480)) * mask)[:, :, None]
        expected = np.rint(placed * alpha + image * (1 - alpha)).clip(0, 255).astype(np.uint8)
        actual = image.copy()
        paste_back(actual, generated, matrix, polygon)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[mask == 0], image[mask == 0])

    def test_bad_output_never_partially_mutates_frame(self):
        image = np.full((256, 256, 3), 80, np.uint8)
        for generated, matrix, polygon in (
            (np.full((3, 256, 256), np.nan), np.eye(2, 3), [[0, 0], [255, 0], [255, 255]]),
            (np.zeros((3, 256, 256)), np.eye(2, 3), [[np.nan, 0], [255, 0], [255, 255]]),
            (np.zeros((3, 256, 256)), np.zeros((2, 3)), [[0, 0], [255, 0], [255, 255]]),
            (np.zeros((3, 256, 256)), np.eye(2, 3), [[400, 400], [500, 400], [500, 500]]),
        ):
            with self.assertRaises(ValueError):
                paste_back(image, generated, matrix, polygon)
            self.assertTrue(np.all(image == 80))

    def test_full_frame_landmarks_are_shared_and_associated_by_iou(self):
        boxes = np.array([[10, 10, 100, 100, 1], [200, 10, 300, 100, 1]], np.float32)
        points = np.arange(20).reshape(2, 5, 2)
        detector = Mock()
        detector.detect.return_value = boxes, points
        frame = MobileFSGANFrame(np.zeros((480, 640, 3), np.uint8), detector)
        frame.prime([boxes[0, :4], boxes[1, :4]])
        np.testing.assert_array_equal(frame.landmarks(boxes[1, :4]), points[1])
        np.testing.assert_array_equal(frame.landmarks(boxes[0, :4]), points[0])
        self.assertEqual(detector.detect.call_count, 1)

    def test_distinct_identities_batched_with_independent_failure_and_chunking(self):
        renderer = MobileFSGANRenderer.__new__(MobileFSGANRenderer)
        renderer.lock, renderer.max_batch = RLock(), 2
        renderer.source_latent = lambda p: (
            np.full(512, p, np.float32)
            if p != -1
            else (_ for _ in ()).throw(ValueError("bad source"))
        )
        renderer.g = Mock()
        renderer.g.numpy.side_effect = lambda values: np.broadcast_to(
            values["source_z"][:, :1, None, None], (len(values["source_z"]), 3, 256, 256)
        ).copy()
        results = renderer.render_crops([np.zeros((256, 256, 3), np.uint8)] * 5, [1, -1, 2, 3, 4])
        self.assertIsInstance(results[1], ValueError)
        for i, expected in ((0, 1), (2, 2), (3, 3), (4, 4)):
            self.assertTrue(np.all(results[i] == expected))
        self.assertEqual(
            [len(c.args[0]["source_z"]) for c in renderer.g.numpy.call_args_list], [1, 2, 1]
        )

    def test_configuration_routes_cuda_model_and_keeps_other_backends(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            FaceAnonymizationConfig(swap_backend="mobilefsgan")
        config = FaceAnonymizationConfig(
            swap_backend="mobilefsgan", swap_provider="cuda", mobilefsgan_max_batch=4
        )
        with patch("service.mobilefsgan.MobileFSGANRenderer") as factory:
            runtime = FaceAnonymizationRuntime(config)
            runtime._renderer()
            self.assertIs(runtime.renderer, factory.return_value)
            self.assertEqual(factory.call_args.kwargs["max_batch"], 4)
        self.assertEqual(FaceAnonymizationConfig().swap_backend, "inswapper")

    def test_manifest_rejects_changed_asset_engine_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = {}
            for name in ("onnx/C_fp32.onnx", "onnx/G_fp32.onnx", "detector/yunet.onnx"):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(name.encode())
                assets[name] = hashlib.sha256(name.encode()).hexdigest()
            (root / "model_manifest.json").write_text(json.dumps({"asset_sha256": assets}))
            plans = {}
            for name in ("C", "G"):
                (root / f"{name}_fp32.plan").write_bytes(name.encode())
                plans[f"{name}_fp32.plan"] = hashlib.sha256(name.encode()).hexdigest()
            metadata = dict(
                tensorrt="11.4",
                gpu="test",
                compute_capability=[8, 6],
                precision="fp32",
                model_onnx_sha256={n: assets[f"onnx/{n}_fp32.onnx"] for n in ("C", "G")},
                engines=plans,
            )
            (root / "engine_manifest.json").write_text(json.dumps(metadata))
            kwargs = dict(version="11.4", gpu="test", capability=(8, 6))
            validate_bundle(root, root, **kwargs)
            with self.assertRaisesRegex(ValueError, "environment"):
                validate_bundle(root, root, **{**kwargs, "version": "11.1"})
            (root / "G_fp32.plan").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "engine hash"):
                validate_bundle(root, root, **kwargs)
            (root / "onnx/G_fp32.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "asset hash"):
                validate_bundle(root, root, **kwargs)
