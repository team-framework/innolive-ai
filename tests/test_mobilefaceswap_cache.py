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
