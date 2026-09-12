#!/usr/bin/env python3
"""Build the dynamic FP16 YOLO26n-seg TensorRT engine used only by this lab."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.detection import EXPECTED_CLASS_NAMES  # noqa: E402
from service.runtime import IMAGE_SIZE  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "models" / "best.pt"
DEFAULT_ENGINE = ROOT / "models" / "best_swap_b4.engine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--workspace", type=float, default=8.0, help="TensorRT workspace GiB")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, output, device = _validate_args(args)
    tensorrt, torch, ultralytics, yolo = _load_dependencies(device)
    _validate_checkpoint(checkpoint, yolo)
    _export(
        checkpoint, output, yolo, device=device, max_batch=args.max_batch, workspace=args.workspace
    )
    manifest = _manifest(
        checkpoint,
        output,
        max_batch=args.max_batch,
        workspace=args.workspace,
        device=device,
        tensorrt=tensorrt,
        torch=torch,
        ultralytics=ultralytics,
    )
    _write_manifest(output.with_suffix(output.suffix + ".json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, int]:
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("dynamic TensorRT export must run on the 3090 Linux host")
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if output.suffix != ".engine":
        raise SystemExit("--output must end in .engine")
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; use --force to replace it")
    if args.max_batch < 1:
        raise SystemExit("--max-batch must be at least 1")
    if not math.isfinite(args.workspace) or args.workspace <= 0:
        raise SystemExit("--workspace must be a positive finite number")
    try:
        return checkpoint, output, int(str(args.device).removeprefix("cuda:"))
    except ValueError as error:
        raise SystemExit("--device must be a CUDA index") from error


def _load_dependencies(device: int) -> tuple[Any, Any, Any, Any]:
    try:
        import tensorrt
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("install requirements-export.txt and requirements-tensorrt.txt") from error
    if not torch.cuda.is_available() or not 0 <= device < torch.cuda.device_count():
        raise SystemExit(f"CUDA device {device} is unavailable")
    return tensorrt, torch, ultralytics, YOLO


def _validate_checkpoint(checkpoint: Path, yolo: Any) -> None:
    model = yolo(str(checkpoint), task="segment")
    names = {int(key): str(value).strip().lower() for key, value in model.names.items()}
    if model.task != "segment" or names != EXPECTED_CLASS_NAMES:
        raise SystemExit(
            f"expected segmentation classes {EXPECTED_CLASS_NAMES}; got task={model.task!r}, names={names!r}"
        )


def _export(
    checkpoint: Path,
    output: Path,
    yolo: Any,
    *,
    device: int,
    max_batch: int,
    workspace: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trt-swap-export-", dir=output.parent) as directory:
        staged = Path(directory)
        staged_checkpoint = staged / checkpoint.name
        shutil.copy2(checkpoint, staged_checkpoint)
        engine = yolo(str(staged_checkpoint), task="segment").export(
            format="engine",
            imgsz=IMAGE_SIZE,
            batch=max_batch,
            dynamic=True,
            half=True,
            simplify=True,
            nms=False,
            workspace=workspace,
            device=device,
        )
        exported = Path(engine)
        if not exported.is_file():
            raise RuntimeError(f"Ultralytics did not create an engine: {exported}")
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        shutil.copy2(exported, temporary_output)
        temporary_output.replace(output)


def _manifest(
    checkpoint: Path,
    output: Path,
    *,
    max_batch: int,
    workspace: float,
    device: int,
    tensorrt: Any,
    torch: Any,
    ultralytics: Any,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "independent TensorRT face-swap test client",
        "created_at": datetime.now(UTC).isoformat(),
        "source_checkpoint": checkpoint.name,
        "source_sha256": _sha256(checkpoint),
        "engine": output.name,
        "engine_sha256": _sha256(output),
        "precision": "fp16",
        "dynamic": True,
        "batch_range": {"min": 1, "opt": max_batch, "max": max_batch},
        "image_size": IMAGE_SIZE,
        "class_names": {str(key): value for key, value in EXPECTED_CLASS_NAMES.items()},
        "workspace_gib": workspace,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "tensorrt": tensorrt.__version__,
        "python": platform.python_version(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
