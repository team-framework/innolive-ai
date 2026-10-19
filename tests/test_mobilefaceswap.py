from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from service.face_anonymization import FaceAnonymizationConfig, FaceAnonymizationRuntime
from service.face_metadata import FaceAttributes
from service.face_presets import FacePreset
from service.mobilefaceswap import (
    REFERENCE,
    MobileFaceSwapFrame,
    MobileFaceSwapRenderer,
    ScrfdLandmarks,
    alignment,
    image_tensor,
    load_weights,
    paste_back,
    source_tensor,
)


class MobileFaceSwapTests(unittest.TestCase):
    def test_alignment_recovers_full_affine_including_nonuniform_scale(self):
        expected = np.array([[1.5, 0.1, 5], [-0.2, 2, 8]], np.float32)
        points = cv2.transform(REFERENCE[None], cv2.invertAffineTransform(expected))[0]
        np.testing.assert_allclose(alignment(points), expected, atol=1e-5)
        for invalid in (np.zeros((5, 2)), np.full((5, 2), np.nan), np.zeros((4, 2))):
            with self.assertRaises(ValueError):
                alignment(invalid)

    def test_rgb_and_imagenet_source_normalization(self):
        image = np.zeros((224, 224, 3), np.uint8)
        image[:] = (0, 127, 255)
        np.testing.assert_allclose(image_tensor(image)[0, :, 0, 0], [1, 127 / 255, 0])
        expected = (np.array([1, 127 / 255, 0]) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        actual = source_tensor(image)
        self.assertEqual(actual.shape, (1, 3, 112, 112))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual[0, :, 0, 0], expected, rtol=1e-6)

    def test_paste_back_cannot_change_pixels_outside_segmentation(self):
        image = np.full((224, 224, 3), 32, np.uint8)
        generated = np.ones((1, 3, 224, 224), np.float32)
        mask = np.ones((1, 1, 224, 224), np.float32)
        matrix = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        polygon = [[70, 70], [150, 70], [150, 150], [70, 150]]
        result = paste_back(image, generated, mask, matrix, polygon)
        self.assertEqual(result[100, 100].tolist(), [255, 255, 255])
        np.testing.assert_array_equal(result[:70], image[:70])
        np.testing.assert_array_equal(result[:, 151:], image[:, 151:])
        with self.assertRaises(ValueError):
            paste_back(image, generated, mask * 0, matrix, polygon)
        with self.assertRaises(ValueError):
            paste_back(image, generated * np.nan, mask, matrix, polygon)

    def test_model_backend_is_opt_in_and_lazy(self):
        with (
            patch("service.mobilefaceswap.MobileFaceSwapRenderer") as mobile,
            patch("service.face_anonymization.InSwapperRenderer") as inswapper,
        ):
            default = FaceAnonymizationRuntime(FaceAnonymizationConfig())
            selected = FaceAnonymizationRuntime(
                FaceAnonymizationConfig(swap_backend="mobilefaceswap")
            )
            mobile.assert_not_called()
            inswapper.assert_not_called()
            selected._renderer()
            selected._renderer()
            mobile.assert_called_once()
            inswapper.assert_not_called()
            default._renderer()
            inswapper.assert_called_once()

    def test_renderer_load_failure_is_sticky_for_blur_fallback(self):
        runtime = FaceAnonymizationRuntime(FaceAnonymizationConfig(swap_backend="mobilefaceswap"))
        with (
            patch("service.mobilefaceswap.MobileFaceSwapRenderer", side_effect=FileNotFoundError),
            self.assertRaises(FileNotFoundError),
            self.assertLogs("innolive.face_swap", level="ERROR"),
        ):
            runtime._renderer()
        self.assertEqual(runtime.renderer_error, "swapper_unavailable")
        with self.assertRaisesRegex(RuntimeError, "swapper_unavailable"):
            runtime._renderer()

    def test_rejects_unsupported_provider_and_mismatched_checkpoints(self):
        with self.assertRaises(ValueError):
            FaceAnonymizationConfig(swap_backend="inswapper", swap_provider="mps")
        with self.assertRaises(ValueError):
            FaceAnonymizationConfig(swap_backend="unknown")
        with self.assertRaises(ValueError):
            MobileFaceSwapRenderer(Path("/missing"), provider="mps")
        model = Mock()
        model.state_dict.return_value = {"layer": np.zeros((2, 3))}
        paddle = Mock()
        for weights in ({}, {"layer": np.zeros((3, 2))}):
            paddle.load.return_value = weights
            with self.assertRaises(ValueError):
                load_weights(model, Path("weights.pdparams"), paddle)
        model.set_state_dict.assert_not_called()

    def test_fp16_is_explicit_and_requires_mobilefaceswap_cuda(self):
        self.assertEqual(FaceAnonymizationConfig().mobilefaceswap_precision, "fp32")
        for backend, provider in (
            ("inswapper", "cuda"),
            ("mobilefaceswap", "cpu"),
            ("mobilefaceswap", "mps"),
        ):
            with self.assertRaisesRegex(ValueError, "fp16 requires"):
                FaceAnonymizationConfig(
                    swap_backend=backend, swap_provider=provider, mobilefaceswap_precision="fp16"
                )
        config = FaceAnonymizationConfig(
            swap_backend="mobilefaceswap", swap_provider="cuda", mobilefaceswap_precision="fp16"
        )
        with patch("service.mobilefaceswap_torch.TorchMobileFaceSwapRenderer") as renderer:
            FaceAnonymizationRuntime(config)._renderer()
            renderer.assert_called_once_with(config.mobilefaceswap_dir, "cuda", precision="fp16")

    def test_source_hash_failure_prevents_detection_and_model_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            path.write_bytes(b"changed image")
            preset = FacePreset(FaceAttributes("female", "20s", "off", "none"), 1, path, "0" * 64)
            renderer = MobileFaceSwapRenderer.__new__(MobileFaceSwapRenderer)
            renderer.sources = OrderedDict()
            renderer.detector = Mock()
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                renderer.source_model(preset)
            renderer.detector.detect.assert_not_called()

    def test_target_landmarks_must_match_the_requested_track(self):
        renderer = MobileFaceSwapRenderer.__new__(MobileFaceSwapRenderer)
        renderer.detector = Mock()
        renderer.detector.detect.return_value = (
            np.array([[0, 0, 10, 10, 0.9]], np.float32),
            np.zeros((1, 5, 2), np.float32),
        )
        with self.assertRaisesRegex(ValueError, "matching the tracked bbox"):
            renderer.swap(np.zeros((224, 224, 3), np.uint8), {"bbox": [80, 80, 140, 140]}, Mock())

    def test_smaller_scrfd_image_keeps_static_canvas_and_restores_coordinates(self):
        detector = ScrfdLandmarks.__new__(ScrfdLandmarks)
        detector.input_name = "input"
        detector.centers = [np.array([[8, 8]]) for _ in range(3)]
        outputs = [np.array([0.9]), np.array([0.0]), np.array([0.0])]
        outputs += [np.ones((1, 4)) for _ in range(3)]
        outputs += [np.ones((1, 5, 2)) for _ in range(3)]
        detector.session = Mock()
        detector.session.run.return_value = outputs
        image = np.full((480, 640, 3), 255, np.uint8)
        boxes, points = detector.detect(image, input_size=320)
        np.testing.assert_allclose(boxes[0, :4], [0, 0, 32, 32])
        np.testing.assert_allclose(points[0], np.full((5, 2), 32))
        blob = detector.session.run.call_args.args[1]["input"]
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertGreater(blob[0, 0, 239, 319], 0)
        self.assertLess(blob[0, 0, 240, 320], 0)

    def test_roi_uses_smaller_scale_without_full_frame_work_on_success(self):
        detector = Mock()
        detector.detect.return_value = (
            np.array([[18, 18, 78, 78, 0.9]], np.float32),
            (REFERENCE / 3)[None],
        )
        frame = MobileFaceSwapFrame(np.zeros((224, 224, 3), np.uint8), detector)
        points = frame.landmarks([80, 80, 140, 140])
        np.testing.assert_allclose(points, REFERENCE / 3 + [62, 62])
        detector.detect.assert_called_once()
        self.assertEqual(detector.detect.call_args.kwargs, {"input_size": 320})

    def test_full_frame_fallback_is_cached_only_for_current_original_frame(self):
        empty = (np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32))
        detected = (np.array([[80, 80, 140, 140, 0.9]], np.float32), REFERENCE[None])
        detector = Mock()
        detector.detect.side_effect = [empty, empty, detected, empty, empty, empty, detected]
        original = np.zeros((224, 224, 3), np.uint8)
        frame = MobileFaceSwapFrame(original, detector)
        np.testing.assert_array_equal(frame.landmarks([80, 80, 140, 140]), REFERENCE)
        np.testing.assert_array_equal(frame.landmarks([80, 80, 140, 140]), REFERENCE)
        self.assertEqual(detector.detect.call_count, 4)
        self.assertIs(detector.detect.call_args_list[1].args[0], original)
        self.assertEqual(detector.detect.call_args_list[1].kwargs, {"input_size": 640})
        self.assertEqual(detector.detect.call_args_list[2].kwargs, {"input_size": 320})
        # New frames must not inherit old coordinates, even if track IDs repeat.
        next_frame = MobileFaceSwapFrame(original.copy(), detector)
        next_frame.landmarks([80, 80, 140, 140])
        self.assertEqual(detector.detect.call_count, 7)

    def test_rendering_reads_original_pixels_and_composites_onto_current_output(self):
        original = np.full((224, 224, 3), 25, np.uint8)
        composed = np.full_like(original, 100)
        target = Mock(image=original)
        target.landmarks.return_value = REFERENCE
        renderer = MobileFaceSwapRenderer.__new__(MobileFaceSwapRenderer)
        renderer.render_crop = Mock(
            return_value=(np.ones((1, 3, 224, 224)), np.ones((1, 1, 224, 224)))
        )
        item = {
            "bbox": [70, 70, 150, 150],
            "mask_polygon": [[70, 70], [150, 70], [150, 150], [70, 150]],
        }
        output = renderer.swap(composed, item, Mock(), target=target)
        np.testing.assert_array_equal(renderer.render_crop.call_args.args[0], original)
        np.testing.assert_array_equal(output[:70], composed[:70])


class MobileFaceSwapBatchTests(unittest.TestCase):
    def test_batch_restores_spatial_batch_anchor_order_and_roi_scale(self):
        detector = ScrfdLandmarks.__new__(ScrfdLandmarks)
        detector.max_batch = 2
        detector.centers = [np.array([[8, 8], [8, 8]]) for _ in range(3)]
        scores = np.array([[[[0.9], [0]], [[0.8], [0]]]], np.float32).reshape(-1, 1)
        empty = np.zeros_like(scores)
        # HWNC: spatial index, batch index, two anchors, output channels.
        distance = np.array([[[[1] * 4] * 2, [[2] * 4] * 2]], np.float32).reshape(-1, 4)
        offsets = np.array([[[[1] * 10] * 2, [[3] * 10] * 2]], np.float32).reshape(-1, 10)
        detector.session = Mock()
        detector.session.run_batch.return_value = (
            [scores, empty, empty] + [distance] * 3 + [offsets] * 3
        )
        results = detector.detect_many(
            [np.zeros((320, 320, 3), np.uint8), np.zeros((160, 160, 3), np.uint8)]
        )
        np.testing.assert_allclose(results[0][1], np.full((1, 5, 2), 16))
        np.testing.assert_allclose(results[1][1], np.full((1, 5, 2), 16))
        np.testing.assert_allclose(results[0][0][0, :4], [0, 0, 16, 16])
        np.testing.assert_allclose(results[1][0][0, :4], [-4, -4, 12, 12])
        self.assertEqual(detector.session.run_batch.call_count, 1)

    def test_prime_keeps_tracks_separate_and_retries_missing_landmarks(self):
        detector = Mock(max_batch=8)
        empty = (np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32))
        detector.detect_many.return_value = [
            (np.array([[18, 18, 78, 78, 0.9]], np.float32), (REFERENCE / 3)[None]),
            empty,
        ]
        detector.detect.return_value = (
            np.array([[150, 80, 210, 140, 0.9]], np.float32),
            REFERENCE[None],
        )
        frame = MobileFaceSwapFrame(np.zeros((224, 224, 3), np.uint8), detector)
        frame.prime([[80, 80, 140, 140], [150, 80, 210, 140]])
        np.testing.assert_allclose(frame.landmarks([80, 80, 140, 140]), REFERENCE / 3 + [62, 62])
        detector.detect.assert_not_called()
        np.testing.assert_allclose(frame.landmarks([150, 80, 210, 140]), REFERENCE)
        detector.detect.assert_called_once()
        self.assertEqual(detector.detect.call_args.kwargs, {"input_size": 640})
        self.assertFalse(MobileFaceSwapFrame(frame.image, detector).roi_points)

    def test_roi_paste_handles_clipped_and_overlapping_faces(self):
        image = np.full((240, 320, 3), 32, np.uint8)
        original = image.copy()
        generated = np.ones((1, 3, 224, 224), np.float32)
        mask = np.ones((1, 1, 224, 224), np.float32)
        matrix = np.array([[1, 0, 40], [0, 1, 20]], np.float32)
        polygon = [[-20, -10], [80, -10], [80, 100], [-20, 100]]
        result = paste_back(image, generated, mask, matrix, polygon)
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(result[:, 81:], image[:, 81:])
        np.testing.assert_array_equal(result[101:], image[101:])
        self.assertEqual(int(result[40, 40, 0]), 255)
        with self.assertRaises(ValueError):
            paste_back(image, generated, mask, matrix, [[400, 400], [500, 400], [500, 500]])

    def test_source_conditioning_keeps_original_onnx_detector(self):
        renderer = MobileFaceSwapRenderer.__new__(MobileFaceSwapRenderer)
        renderer.detector = Mock(accelerated=True)
        renderer.conditioning_detector = None
        renderer.landmarks_path = Path("original.onnx")
        renderer.landmarks_provider = "cuda"
        image = np.zeros((224, 224, 3), np.uint8)
        with patch("service.mobilefaceswap.ScrfdLandmarks") as detector:
            renderer.source_landmarks(image)
            renderer.source_landmarks(image)
            detector.assert_called_once_with(Path("original.onnx"), "cuda", allow_tensorrt=False)
            self.assertEqual(detector.return_value.detect.call_count, 2)
            renderer.detector.detect.assert_not_called()
