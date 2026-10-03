from __future__ import annotations

import csv
import hashlib
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from scripts.import_face_presets import dataset_catalog, install_dataset
from service.face_metadata import FaceAttributes
from service.face_presets import FacePreset, PresetCatalog
from service.inswapper import InSwapperRenderer

FIELDS = (
    "filename",
    "person_key",
    "local_id",
    "gender",
    "age_group",
    "glasses",
    "expression",
    "status",
    "image_path",
    "width",
    "height",
    "sha256",
)


class PresetImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture.name)
        (cls.root / "images").mkdir()
        rows = []
        for index, (gender, age, glasses, exp, identity) in enumerate(
            itertools.product(range(2), range(4), range(2), range(7), range(5))
        ):
            name = f"g{gender}_a{age}_gl{glasses}_e{exp}_id{identity}.png"
            image = np.zeros((24, 24, 3), np.uint8)
            image[:, :, 0] = index % 256
            image[:, :, 1] = index // 256
            success, encoded = cv2.imencode(".png", image)
            assert success
            data = encoded.tobytes()
            (cls.root / "images" / name).write_bytes(data)
            rows.append(
                dict(
                    zip(
                        FIELDS,
                        (
                            name,
                            f"g{gender}_a{age}_id{identity}",
                            identity,
                            ("female", "male")[gender],
                            ("10s", "20s", "30s", "over40s")[age],
                            ("off", "on")[glasses],
                            ("none", "anger", "disgust", "fear", "happy", "sad", "surprise")[exp],
                            "generated",
                            f"images/{name}",
                            24,
                            24,
                            hashlib.sha256(data).hexdigest(),
                        ),
                        strict=True,
                    )
                )
            )
        cls.write_rows(cls.root, rows)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    @staticmethod
    def write_rows(root, rows):
        with (root / "dataset_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source"
        shutil.copytree(self.root, self.source)
        with (self.source / "dataset_manifest.csv").open(encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        self.output = Path(self.temporary.name) / "installed" / "face_presets.json"

    def test_complete_dataset_maps_zero_based_ids_without_changing_originals(self):
        before = (self.source / "dataset_manifest.csv").read_bytes()
        provenance = install_dataset(self.source, self.output)
        self.assertEqual(provenance["image_count"], 560)
        self.assertEqual(provenance["identity_count"], 40)
        catalog = PresetCatalog(self.output)
        self.assertEqual(len(catalog.presets), 560)
        for slot in range(1, 6):
            preset = catalog.match(FaceAttributes("female", "10s", "off", "none"), slot)
            self.assertEqual(preset.image.name, f"g0_a0_gl0_e0_id{slot - 1}.png")
            self.assertEqual(hashlib.sha256(preset.image.read_bytes()).hexdigest(), preset.sha256)
        self.assertEqual(before, (self.source / "dataset_manifest.csv").read_bytes())
        self.assertNotIn(str(self.source), self.output.read_text())
        self.assertEqual(install_dataset(self.source, self.output), provenance)

    def test_missing_or_duplicate_combinations_are_rejected(self):
        for rows in (self.rows[:-1], [self.rows[0], self.rows[0], *self.rows[2:]]):
            with self.subTest(count=len(rows)):
                self.write_rows(self.source, rows)
                with self.assertRaises(ValueError):
                    dataset_catalog(self.source)

    def test_mislabeled_filename_and_invalid_identity_are_rejected(self):
        for mutation in ({"gender": "male"}, {"local_id": "5"}, {"glasses": "unknown"}):
            rows = [dict(row) for row in self.rows]
            rows[0].update(mutation)
            self.write_rows(self.source, rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dataset_catalog(self.source)

    def test_modified_image_and_wrong_dimensions_are_rejected_before_install(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["width"] = "99"
        self.write_rows(self.source, rows)
        with self.assertRaisesRegex(ValueError, "wrong dimensions"):
            install_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.write_rows(self.source, self.rows)
        (self.source / self.rows[0]["image_path"]).write_bytes(b"corrupt image")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            install_dataset(self.source, self.output)

    def test_conflicting_existing_assets_are_preserved(self):
        install_dataset(self.source, self.output)
        before = self.output.read_bytes()
        image = self.output.parent / "face_presets/images/g0_a0_gl0_e0_id0.png"
        image.write_bytes(b"local change")
        with self.assertRaisesRegex(ValueError, "existing preset assets differ"):
            install_dataset(self.source, self.output)
        self.assertEqual(image.read_bytes(), b"local change")
        self.assertEqual(self.output.read_bytes(), before)

    def test_catalog_rejects_invalid_optional_hash(self):
        self.output.parent.mkdir()
        self.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "presets": [
                        {
                            "gender": "female",
                            "age": "10s",
                            "identity": 1,
                            "glasses": "off",
                            "exp": "none",
                            "image": "pending.png",
                            "sha256": "bad hash",
                        }
                    ],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "invalid preset SHA-256"):
            PresetCatalog(self.output)

    def test_source_latent_rejects_modified_asset_before_model_inference(self):
        renderer = InSwapperRenderer.__new__(InSwapperRenderer)
        renderer.sources = {}
        renderer.detect = Mock()
        path = self.source / self.rows[0]["image_path"]
        preset = FacePreset(FaceAttributes("female", "10s", "off", "none"), 1, path, "0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
            renderer.source_latent(preset)
        renderer.detect.assert_not_called()

    def test_yunet_downscales_then_restores_bbox_landmarks_and_score(self):
        renderer = InSwapperRenderer.__new__(InSwapperRenderer)
        detected = np.array(
            [[20, 40, 80, 100, 30, 60, 70, 60, 50, 80, 40, 100, 60, 100, 0.9]], np.float32
        )
        renderer.detector = Mock(detect=Mock(return_value=(None, detected)))
        (face,) = renderer.detect(np.zeros((1280, 640, 3), np.uint8))
        renderer.detector.setInputSize.assert_called_once_with((320, 640))
        np.testing.assert_allclose(face[:14], detected[0, :14] * 2)
        self.assertAlmostEqual(face[14], 0.9)
        self.assertEqual(renderer.detector.detect.call_args.args[0].shape, (640, 320, 3))


if __name__ == "__main__":
    unittest.main()
