#!/usr/bin/env python3
"""Export the standard static B1-640 FP16 TensorRT face model."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_ENGINE,
    EXPECTED_CLASS_NAMES,
    IMAGE_SIZE,
    sha256_file,
)

STANDARD_PROFILE = "B1-640-Q90-W5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENGINE)
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
        help="replace an existing engine and manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, output = _validated_paths(args)
    tensorrt, torch, ultralytics, yolo, device_index = _load_dependencies(args.device)
    _validate_checkpoint(checkpoint, yolo)
    _export_engine(
        checkpoint,
        output,
        yolo,
        device_index=device_index,
        workspace=args.workspace,
    )
    manifest = _manifest(
        checkpoint,
        output,
        device_index=device_index,
        workspace=args.workspace,
        tensorrt=tensorrt,
        torch=torch,
        ultralytics=ultralytics,
    )
    manifest_path = output.with_suffix(output.suffix + ".json")
    _write_json_atomically(manifest_path, manifest)
    print(f"engine:   {output}")
    print(f"manifest: {manifest_path}")
    print(f"sha256:   {manifest['engine_sha256']}")


def _validated_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if output.suffix.lower() != ".engine":
        raise SystemExit("--output must end in .engine")
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output} (use --force to replace it)")
    if not math.isfinite(args.workspace) or args.workspace <= 0:
        raise SystemExit("--workspace must be a positive finite number")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit(
            "TensorRT export requires a Linux x86_64 host with an NVIDIA GPU; "
            "use --backend auto or --backend pytorch on this machine"
        )
    return checkpoint, output


def _load_dependencies(device: str) -> tuple[Any, Any, Any, Any, int]:
    try:
        import tensorrt
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "TensorRT export dependencies are missing; install requirements-export.txt"
        ) from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; build on the deployment NVIDIA host")
    try:
        device_index = int(str(device).removeprefix("cuda:"))
    except ValueError as error:
        raise SystemExit("--device must be a CUDA device index") from error
    if not 0 <= device_index < torch.cuda.device_count():
        raise SystemExit(f"CUDA device is unavailable: {device}")
    return tensorrt, torch, ultralytics, YOLO, device_index


def _validate_checkpoint(checkpoint: Path, yolo: Any) -> None:
    source = yolo(str(checkpoint), task="segment")
    names = {int(key): str(value).strip().lower() for key, value in source.names.items()}
    if source.task != "segment" or names != EXPECTED_CLASS_NAMES:
        raise SystemExit(
            "expected a face segmentation checkpoint with "
            f"names={EXPECTED_CLASS_NAMES}, "
            f"got task={source.task!r}, names={names!r}"
        )


def _export_engine(
    checkpoint: Path,
    output: Path,
    yolo: Any,
    *,
    device_index: int,
    workspace: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trt-export-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        staged_checkpoint = temporary_path / "best.pt"
        shutil.copy2(checkpoint, staged_checkpoint)
        model = yolo(str(staged_checkpoint), task="segment")
        exported = Path(
            model.export(
                format="engine",
                imgsz=IMAGE_SIZE,
                batch=1,
                quantize=16,
                dynamic=False,
                simplify=True,
                nms=False,
                workspace=workspace,
                device=device_index,
            )
        )
        if not exported.is_file():
            raise RuntimeError(f"Ultralytics did not create an engine: {exported}")
        staged_engine = output.with_suffix(output.suffix + ".tmp")
        shutil.copy2(exported, staged_engine)
        staged_engine.replace(output)


def _manifest(
    checkpoint: Path,
    output: Path,
    *,
    device_index: int,
    workspace: float,
    tensorrt: Any,
    torch: Any,
    ultralytics: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "standard_profile": STANDARD_PROFILE,
        "created_at": datetime.now(UTC).isoformat(),
        "source_checkpoint": checkpoint.name,
        "source_sha256": sha256_file(checkpoint),
        "engine": output.name,
        "engine_sha256": sha256_file(output),
        "precision": "fp16",
        "dynamic": False,
        "batch": 1,
        "image_size": IMAGE_SIZE,
        "class_names": {"0": "face"},
        "workspace_gib": workspace,
        "gpu": torch.cuda.get_device_name(device_index),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "tensorrt": tensorrt.__version__,
        "python": platform.python_version(),
        "note": "Rebuild and re-run acceptance tests on every deployment GPU/TensorRT stack.",
    }


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    staged_manifest = path.with_suffix(path.suffix + ".tmp")
    staged_manifest.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staged_manifest.replace(path)


if __name__ == "__main__":
    main()
