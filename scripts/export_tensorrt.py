#!/usr/bin/env python3
"""Export the standard static B1-640 FP16 TensorRT face model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import tensorrt
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = ROOT / "models" / "best_b1.engine"
IMAGE_SIZE = 640
MODEL_CLASS = {0: "face"}
STANDARD_PROFILE = "B1-640-Q90-W5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT builder workspace in GiB",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing engine and manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if output.suffix.lower() != ".engine":
        raise SystemExit("--output must end in .engine")
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output} (use --force to replace it)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; build on the deployment NVIDIA host")

    source = YOLO(str(checkpoint), task="segment")
    names = {int(key): str(value).strip().lower() for key, value in source.names.items()}
    if source.task != "segment" or names != MODEL_CLASS:
        raise SystemExit(
            f"expected a face segmentation checkpoint with names={MODEL_CLASS}, "
            f"got task={source.task!r}, names={names!r}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trt-export-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        staged_checkpoint = temporary_path / "best.pt"
        shutil.copy2(checkpoint, staged_checkpoint)
        model = YOLO(str(staged_checkpoint), task="segment")
        exported = Path(
            model.export(
                format="engine",
                imgsz=IMAGE_SIZE,
                batch=1,
                quantize=16,
                dynamic=False,
                simplify=True,
                nms=False,
                workspace=args.workspace,
                device=args.device,
            )
        )
        if not exported.is_file():
            raise RuntimeError(f"Ultralytics did not create an engine: {exported}")
        staged_engine = output.with_suffix(output.suffix + ".tmp")
        shutil.copy2(exported, staged_engine)
        staged_engine.replace(output)

    manifest = {
        "schema_version": 1,
        "standard_profile": STANDARD_PROFILE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": checkpoint.name,
        "source_sha256": sha256_file(checkpoint),
        "engine": output.name,
        "engine_sha256": sha256_file(output),
        "precision": "fp16",
        "dynamic": False,
        "batch": 1,
        "image_size": IMAGE_SIZE,
        "class_names": {"0": "face"},
        "workspace_gib": args.workspace,
        "gpu": torch.cuda.get_device_name(int(args.device)),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "tensorrt": tensorrt.__version__,
        "python": platform.python_version(),
        "note": "Rebuild and re-run acceptance tests on every deployment GPU/TensorRT stack.",
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    staged_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    staged_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staged_manifest.replace(manifest_path)
    print(f"engine:   {output}")
    print(f"manifest: {manifest_path}")
    print(f"sha256:   {manifest['engine_sha256']}")


if __name__ == "__main__":
    main()
