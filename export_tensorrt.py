from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed-batch TensorRT engine")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.getenv("AI_SOURCE_MODEL", "models/yolo.pt")),
    )
    parser.add_argument("--batch", type=int, choices=(1, 4), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def export_engine(source: Path, output: Path, batch_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        staged_model = Path(directory) / output.with_suffix(".pt").name
        shutil.copy2(source, staged_model)
        exported = YOLO(str(staged_model), task="segment").export(
            format="engine",
            batch=batch_size,
            dynamic=False,
            quantize=16,
            imgsz=int(os.getenv("AI_IMAGE_SIZE", "640")),
            device=os.getenv("AI_DEVICES", "0").split(",")[0],
            workspace=float(os.getenv("AI_TENSORRT_WORKSPACE_GB", "4")),
        )
        shutil.move(exported, output)


if __name__ == "__main__":
    arguments = parse_args()
    destination = arguments.output or Path(f"models/yolo_b{arguments.batch}.engine")
    export_engine(arguments.source, destination, arguments.batch)
