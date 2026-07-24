#!/usr/bin/env python3
"""Build a GPU-specific TensorRT engine for the head segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Maximum TensorRT builder workspace in GiB",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Support variable shapes/batches at a small performance cost",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Build INT8 instead of FP16; requires representative calibration data",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Ultralytics dataset YAML used for INT8 calibration",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the calibration dataset to use for INT8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Model does not exist: {model_path}")
    if args.int8 and args.data is None:
        raise SystemExit("--int8 requires --data with representative head images")
    if args.data is not None and not args.data.expanduser().is_file():
        raise SystemExit(f"Dataset YAML does not exist: {args.data.expanduser()}")

    model = YOLO(str(model_path), task="segment")
    names = {int(key): str(value) for key, value in model.names.items()}
    if model.task != "segment" or names.get(0, "").lower() != "head":
        raise SystemExit(
            f"Expected a segment model with class mapping 0: head, got {model.task=} {names=}"
        )

    export_args = {
        "format": "engine",
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "workspace": args.workspace,
        "quantize": 8 if args.int8 else 16,
        "simplify": True,
        "nms": True,
    }
    if args.int8:
        export_args["data"] = str(args.data.expanduser().resolve())
        export_args["fraction"] = args.fraction

    engine_path = Path(model.export(**export_args)).resolve()
    print(f"TensorRT engine written to: {engine_path}")
    print("The server will automatically prefer it over best.pt on this GPU.")


if __name__ == "__main__":
    main()
