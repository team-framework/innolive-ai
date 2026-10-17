from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from service.runtime import (
    IMAGE_SIZE,
    RuntimeConfig,
    RuntimeManager,
    detector_mask_pad,
    select_runtime,
    validate_engine,
)


class RuntimeContractTests(unittest.TestCase):
    def engine(self, root: Path, **overrides) -> Path:
        engine = root / "best_b1.engine"
        engine.write_bytes(b"engine")
        manifest = {
            "schema_version": 1,
            "standard_profile": "B1-640-Q90-W5",
            "precision": "fp16",
            "dynamic": False,
            "batch": 1,
            "image_size": 640,
            "class_names": {"0": "face", "1": "number_plate"},
            "source_checkpoint": "best.pt",
            "source_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
        }
        manifest.update(overrides)
        engine.with_suffix(".engine.json").write_text(json.dumps(manifest))
        (root / "best.pt").write_bytes(b"checkpoint")
        return engine

    def test_accepts_exact_standard_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.engine(Path(temporary))
            self.assertEqual(validate_engine(RuntimeConfig(engine=engine))["batch"], 1)

    def test_rejects_b4_dynamic_non_fp16_and_wrong_size(self):
        cases = (
            ({"batch": 4}, "batch"),
            ({"dynamic": True}, "dynamic"),
            ({"precision": "int8"}, "precision"),
            ({"image_size": 960}, "image_size"),
        )
        for override, message in cases:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temporary:
                engine = self.engine(Path(temporary), **override)
                with self.assertRaisesRegex(RuntimeError, message):
                    validate_engine(RuntimeConfig(engine=engine))

    def test_rejects_hash_mismatch_and_pytorch_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.engine(Path(temporary))
            engine.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                validate_engine(RuntimeConfig(engine=engine))
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(RuntimeError, "TensorRT"):
                validate_engine(RuntimeConfig(engine=checkpoint))

    def test_rejects_checkpoint_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = self.engine(Path(temporary))
            (Path(temporary) / "best.pt").write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "checkpoint hash"):
                validate_engine(RuntimeConfig(engine=engine))

    def test_auto_falls_back_to_pytorch_when_tensorrt_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = RuntimeConfig(checkpoint=checkpoint, device="cpu")
            with patch("service.runtime._tensorrt_supported", return_value=False):
                selection = select_runtime(config)
        self.assertEqual(selection.backend, "pytorch")
        self.assertEqual(selection.artifact, checkpoint.resolve())
        self.assertEqual(selection.device, "cpu")

    def test_explicit_tensorrt_never_silently_falls_back(self):
        config = RuntimeConfig(backend="tensorrt")
        with (
            patch("service.runtime._tensorrt_supported", return_value=False),
            self.assertRaisesRegex(RuntimeError, "TensorRT requires"),
        ):
            select_runtime(config)

    def test_normalizes_backend_and_device_options(self):
        config = RuntimeConfig(backend=" PyTorch ", device=" MPS ")
        self.assertEqual((config.backend, config.device), ("pytorch", "mps"))

    def test_prediction_uses_640_and_both_model_classes(self):
        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def predict(self, **kwargs):
                self.kwargs = kwargs
                return ["prediction"]

        model = FakeModel()
        runtime = RuntimeManager.__new__(RuntimeManager)
        runtime._model = model
        runtime.device = "cpu"

        self.assertEqual(runtime._predict(np.zeros((64, 64, 3), dtype=np.uint8)), "prediction")
        self.assertEqual(model.kwargs["imgsz"], IMAGE_SIZE)
        self.assertEqual(model.kwargs["classes"], [0, 1])
        self.assertIs(model.kwargs["retina_masks"], False)

    def test_mask_pad_is_one_detector_mask_pixel_in_output_pixels(self):
        cases = {(1920, 1080): 3, (1080, 1920): 3, (1280, 720): 2, (640, 360): 1, (64, 64): 1}
        for (height, width), expected in cases.items():
            with self.subTest(height=height, width=width):
                self.assertEqual(detector_mask_pad(height, width), expected)
        for height, width in ((0, 640), (640, 0)):
            with self.subTest(height=height, width=width), self.assertRaises(ValueError):
                detector_mask_pad(height, width)


class _FakePrediction:
    class boxes:
        """Mimics the ultralytics ``prediction.boxes.cpu().numpy()`` chain."""

        @staticmethod
        def cpu():
            return _FakePrediction.boxes

        @staticmethod
        def numpy():
            return "boxes"


class _MotionTracker:
    motion_precomputable = True

    def __init__(self):
        self.events: list[str] = []
        self.motion_thread: str | None = None
        self.received = "unset"

    def estimate_motion(self, image):
        import threading
        import time

        self.motion_thread = threading.current_thread().name
        time.sleep(0.05)
        self.events.append("motion")
        return "warp"

    stop_after_update = False

    def update(self, boxes, image, *, motion="unset"):
        self.received = motion
        if self.stop_after_update:
            raise RuntimeError("stop after the tracker received the warp")
        return np.zeros((0, 8), dtype=np.float32)


class ParallelMotionTests(unittest.TestCase):
    def _runtime(self, predict):
        from concurrent.futures import ThreadPoolExecutor

        runtime = RuntimeManager.__new__(RuntimeManager)
        runtime.backend = "tensorrt"
        runtime._predict = predict
        runtime._motion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="b1-motion")
        self.addCleanup(runtime._motion_executor.shutdown, wait=True)
        return runtime

    def test_motion_runs_beside_inference_and_reaches_the_tracker(self):
        from service.runtime import TrackingFailure

        tracker = _MotionTracker()
        tracker.stop_after_update = True

        def predict(image):
            tracker.events.append("predict")
            return _FakePrediction()

        runtime = self._runtime(predict)
        with self.assertRaises(TrackingFailure):
            runtime._infer_sync(np.zeros((8, 8, 3), dtype=np.uint8), tracker)
        self.assertEqual(tracker.received, "warp")
        self.assertTrue(tracker.motion_thread.startswith("b1-motion"))
        self.assertEqual(tracker.events, ["predict", "motion"])

    def test_inference_failure_waits_for_motion_before_raising(self):
        from service.runtime import InferenceFailure

        tracker = _MotionTracker()

        def predict(image):
            raise RuntimeError("engine")

        runtime = self._runtime(predict)
        with self.assertRaises(InferenceFailure):
            runtime._infer_sync(np.zeros((8, 8, 3), dtype=np.uint8), tracker)
        self.assertEqual(tracker.events, ["motion"])
        self.assertEqual(tracker.received, "unset")


if __name__ == "__main__":
    unittest.main()
