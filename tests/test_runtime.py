from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.engine.results import Boxes

from service.detection import EXPECTED_CLASS_NAMES
from service.runtime import (
    IMAGE_SIZE,
    RuntimeConfig,
    RuntimeManager,
    select_runtime,
    validate_engine,
)
from service.tracking import StreamTracker


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

    def test_raw_detections_keep_low_confidence_plates_without_changing_tracks(self):
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        prediction = SimpleNamespace(
            boxes=Boxes(
                torch.tensor([[20, 20, 80, 80, 0.9, 0], [500, 200, 630, 240, 0.03, 1]]),
                image.shape[:2],
            ),
            masks=SimpleNamespace(
                xy=[
                    np.array([[20, 20], [80, 20], [80, 80], [20, 80]]),
                    np.array([[500, 200], [630, 200], [630, 240], [500, 240]]),
                ]
            ),
        )
        runtime = RuntimeManager.__new__(RuntimeManager)
        runtime.names = EXPECTED_CLASS_NAMES
        with patch.object(runtime, "_predict", return_value=prediction):
            normal = runtime._infer_sync(image, StreamTracker(device="cpu"))
            diagnostic = runtime._infer_sync(
                image, StreamTracker(device="cpu"), include_raw_detections=True
            )
            prediction.masks.xy[1] = np.array([[500, 200], [630, 200]])
            box_only = runtime._infer_sync(
                image, StreamTracker(device="cpu"), include_raw_detections=True
            )
        self.assertEqual(normal["objects"], diagnostic["objects"])
        self.assertEqual([item["class_name"] for item in normal["objects"]], ["face"])
        self.assertEqual(normal["raw_objects"], [])
        plate = diagnostic["raw_objects"][1]
        self.assertEqual(plate["class_name"], "number_plate")
        self.assertEqual(plate["confidence"], 0.03)
        self.assertEqual(plate["bbox"], [500.0, 200.0, 630.0, 240.0])
        self.assertEqual(plate["mask_area_px"], 5200.0)
        self.assertNotIn("track_id", plate)
        self.assertEqual(box_only["raw_objects"][1]["mask_polygon"], [])
        self.assertEqual(box_only["raw_objects"][1]["bbox"], plate["bbox"])


if __name__ == "__main__":
    unittest.main()
