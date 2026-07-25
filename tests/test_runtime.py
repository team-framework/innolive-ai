from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from service.runtime import RuntimeConfig, validate_engine


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
            "class_names": {"0": "face"},
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


if __name__ == "__main__":
    unittest.main()
