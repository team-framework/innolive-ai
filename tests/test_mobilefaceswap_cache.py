import hashlib
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

import torch

from service.face_metadata import FaceAttributes
from service.face_presets import FacePreset
from service.mobilefaceswap_torch import (
    ConditionedModelCache,
    CudaGraphModel,
    MobileFaceSwapUNet,
    TorchMobileFaceSwapRenderer,
    fuse_model,
)


class ConditionedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("MobileFaceSwap_224.pdparams", "arcface.pdparams"):
            (self.root / name).write_bytes(name.encode())
        (self.root / "landmarks").mkdir()
        (self.root / "landmarks/scrfd_10g_bnkps.onnx").write_bytes(b"landmark weights")
        image = self.root / "source.png"
        image.write_bytes(b"source image bytes")
        self.preset = FacePreset(
            FaceAttributes("female", "20s", "off", "none"),
            1,
            image,
            hashlib.sha256(image.read_bytes()).hexdigest(),
        )
        self.cache = ConditionedModelCache(self.root / "cache", self.root)
        self.model = fuse_model(MobileFaceSwapUNet().eval()).requires_grad_(False)

    def test_cached_model_preserves_output_and_bypasses_source_conditioning(self):
        self.cache.save(self.preset, self.model)
        renderer = object.__new__(TorchMobileFaceSwapRenderer)
        renderer.disk_cache = self.cache
        renderer.torch_sources = OrderedDict()
        renderer.device, renderer.dtype = torch.device("cpu"), torch.float32
        renderer.source_model = Mock(side_effect=AssertionError("must not condition on cache hit"))
        loaded = renderer.torch_source_model(self.preset)
        renderer.source_model.assert_not_called()
        values = torch.linspace(0, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
        with torch.inference_mode():
            expected, actual = self.model(values), loaded(values)
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_cache_is_invalidated_by_model_or_source_changes(self):
        self.cache.save(self.preset, self.model)
        (self.root / "arcface.pdparams").write_bytes(b"different model")
        changed = ConditionedModelCache(self.root / "cache", self.root)
        self.assertNotEqual(changed.fingerprint, self.cache.fingerprint)
        self.assertIsNone(changed.load(self.preset))
        self.preset.image.write_bytes(b"different source")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.cache.load(self.preset)

    def test_corrupt_or_nonfinite_cache_is_rejected(self):
        self.cache.save(self.preset, self.model)
        path = self.cache.directory / f"{self.cache.key(self.preset)}.pt"
        payload = torch.load(path, weights_only=True)
        next(iter(payload["weights"].values())).fill_(float("nan"))
        torch.save(payload, path)
        with self.assertLogs("innolive.mobilefaceswap", level="WARNING"):
            self.assertIsNone(self.cache.load(self.preset))
        path.write_bytes(b"invalid model archive")
        with self.assertLogs("innolive.mobilefaceswap", level="WARNING"):
            self.assertIsNone(self.cache.load(self.preset))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaGraphTests(unittest.TestCase):
    def test_graph_preserves_output_and_isolates_source_models(self):
        torch.manual_seed(41)
        models = [
            fuse_model(MobileFaceSwapUNet().eval()).cuda().half().requires_grad_(False)
            for _ in range(2)
        ]
        models = [m.to(memory_format=torch.channels_last) for m in models]
        graphs = [CudaGraphModel(m, torch.device("cuda"), torch.float16) for m in models]
        with torch.inference_mode():
            for index in (0, 1, 0, 1):
                values = torch.rand(
                    (1, 3, 224, 224), device="cuda", dtype=torch.float16
                ).contiguous(memory_format=torch.channels_last)
                expected = models[index](values)
                actual = graphs[index](values)
                for a, b in zip(expected, actual, strict=True):
                    torch.testing.assert_close(a.float(), b, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ScrfdCudaPreprocessingTests(unittest.TestCase):
    def test_gpu_pixels_match_original_blob_and_reject_stale_engine(self):
        import json
        import os

        import cv2
        import numpy as np

        from service.scrfd_tensorrt import ScrfdTensorRT

        engine = os.environ.get("MOBILEFACESWAP_TEST_ENGINE")
        onnx = os.environ.get("MOBILEFACESWAP_TEST_ONNX")
        if not engine or not onnx:
            self.skipTest("set explicit SCRFD test artifact paths")
        runner = ScrfdTensorRT(Path(engine), Path(onnx))
        rng = np.random.default_rng(42)
        for size in (320, 640):
            pixels = rng.integers(0, 256, (2, size, size, 3), dtype=np.uint8)
            canvases = np.zeros((2, 640, 640, 3), np.uint8)
            canvases[:, :size, :size] = pixels
            blob = cv2.dnn.blobFromImages(canvases, 1 / 128, (640, 640), (127.5,) * 3, swapRB=True)
            expected = [v.copy() for v in runner.run_batch(blob)]
            actual = runner.run_images(pixels)
            for a, b in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(a, b)
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "stale.engine"
            link.symlink_to(Path(engine).resolve())
            metadata = json.loads(Path(engine + ".json").read_text())
            metadata["onnx_sha256"] = "wrong-model"
            link.with_suffix(".engine.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                ScrfdTensorRT(link, Path(onnx))

    def test_capture_can_coexist_with_other_thread_device_synchronization(self):
        import threading

        from service.cuda_capture import CUDA_CAPTURE_LOCK

        stopped = threading.Event()
        entered = threading.Event()
        errors = []

        def synchronize():
            try:
                while not stopped.is_set():
                    with CUDA_CAPTURE_LOCK:
                        torch.cuda.synchronize()
                    entered.set()
                    stopped.wait(0.001)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=synchronize)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            for _ in range(3):
                model = fuse_model(MobileFaceSwapUNet().eval()).cuda().half().requires_grad_(False)
                CudaGraphModel(model, torch.device("cuda"), torch.float16)
        finally:
            stopped.set()
            worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
