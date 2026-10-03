#!/usr/bin/env python3
"""Validate and install the synthetic_faces dataset without changing its originals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.face_metadata import CLASSES, FaceAttributes
from service.face_presets import DEFAULT_PRESET_MANIFEST, IDENTITY_COUNT, PresetCatalog


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def dataset_catalog(source: Path, *, progress: Callable[[int, int], None] | None = None) -> dict:
    source = source.expanduser().resolve()
    manifest = source / "dataset_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(itertools.product(*CLASSES.values(), range(IDENTITY_COUNT)))
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} images, found {len(rows)} manifest rows")
    entries = []
    seen = set()
    hashes = set()
    names = set()
    for index, row in enumerate(rows, 1):
        attrs = FaceAttributes(row["gender"], row["age_group"], row["glasses"], row["expression"])
        local_id = int(row["local_id"])
        key = (attrs.gender, attrs.age, attrs.glasses, attrs.exp, local_id)
        if key not in expected or key in seen:
            raise ValueError(f"invalid or duplicate dataset combination: {key}")
        seen.add(key)
        gender, age, glasses, expression = [
            CLASSES[name].index(getattr(attrs, name)) for name in CLASSES
        ]
        name = f"g{gender}_a{age}_gl{glasses}_e{expression}_id{local_id}.png"
        person_key = f"g{gender}_a{age}_id{local_id}"
        if (
            row["filename"] != name
            or row["person_key"] != person_key
            or row["image_path"] != f"images/{name}"
        ):
            raise ValueError(f"filename/person_key/attributes mismatch: {row['filename']}")
        if row["status"] != "generated":
            raise ValueError(f"dataset image is not generated: {name}")
        path = (source / row["image_path"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f"dataset image escapes source directory: {name}")
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or sha256 != row["sha256"]:
            raise ValueError(f"dataset SHA-256 mismatch: {name}")
        if sha256 in hashes:
            raise ValueError(f"dataset reuses the same image bytes: {name}")
        hashes.add(sha256)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (int(row["height"]), int(row["width"])):
            raise ValueError(f"dataset image cannot be decoded or has wrong dimensions: {name}")
        names.add(name)
        entries.append(
            {
                "gender": attrs.gender,
                "age": attrs.age,
                "identity": local_id + 1,
                "glasses": attrs.glasses,
                "exp": attrs.exp,
                "image": f"face_presets/images/{name}",
                "sha256": sha256,
                "source_person_key": person_key,
                "source_local_id": local_id,
            }
        )
        if progress and (index % 56 == 0 or index == len(rows)):
            progress(index, len(rows))
    actual_names = {path.name for path in (source / "images").glob("*.png")}
    if seen != expected or actual_names != names:
        raise ValueError("dataset manifest and PNG combinations do not match exactly")
    entries.sort(
        key=lambda item: (
            item["gender"],
            item["age"],
            item["identity"],
            item["glasses"],
            item["exp"],
        )
    )
    return {
        "schema_version": 1,
        "provenance": {
            "dataset": "synthetic_faces",
            "dataset_manifest_sha256": digest(manifest),
            "image_count": len(entries),
            "identity_count": len(entries) // 14,
            "source_local_ids": [0, 4],
            "serving_identity_slots": [1, 5],
            "identity_mapping": "serving identity = source local_id + 1",
            "original_images_preserved": True,
        },
        "presets": entries,
    }


def install_dataset(
    source: Path,
    output: Path = DEFAULT_PRESET_MANIFEST,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    payload = dataset_catalog(source, progress=progress)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / "face_presets"
    if assets.exists():
        # A repeated import is idempotent; conflicting local assets are preserved.
        for entry in payload["presets"]:
            destination = output.parent / entry["image"]
            if not destination.is_file() or digest(destination) != entry["sha256"]:
                raise ValueError(f"existing preset assets differ from this dataset: {destination}")
    with tempfile.TemporaryDirectory(
        prefix=".face-presets-import-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        if not assets.exists():
            staged_assets = staging / "face_presets"
            images = staged_assets / "images"
            images.mkdir(parents=True)
            for entry in payload["presets"]:
                name = Path(entry["image"]).name
                destination = images / name
                shutil.copyfile(source / "images" / name, destination)
                if digest(destination) != entry["sha256"]:
                    raise ValueError(f"copied preset SHA-256 mismatch: {name}")
            for name in (
                "dataset_manifest.csv",
                "identity_manifest.csv",
                "codebook.json",
                "README.md",
            ):
                original = source / name
                if original.is_file():
                    shutil.copyfile(original, staged_assets / name)
            staged_assets.rename(assets)
        staged_manifest = staging / "catalog.json"
        staged_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staged_manifest.replace(output)
    catalog = PresetCatalog(output)
    if len(catalog.presets) != len(payload["presets"]):
        raise RuntimeError("installed catalog does not match the validated dataset")
    return payload["provenance"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRESET_MANIFEST)
    args = parser.parse_args()
    try:
        provenance = install_dataset(
            args.source,
            args.output,
            progress=lambda current, total: print(
                f"validated {current}/{total} presets", file=sys.stderr
            ),
        )
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from error
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
