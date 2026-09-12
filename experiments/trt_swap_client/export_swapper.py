#!/usr/bin/env python3
"""Build a static TensorRT 11 engine for inswapper_128 on the current host."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
DEFAULT_ENGINE = ROOT / "models" / "face_swap" / "inswapper_128_trt11.engine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--workspace", type=float, default=2.0, help="workspace GiB")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path, output = args.onnx.expanduser().resolve(), args.output.expanduser().resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"InSwapper ONNX not found: {onnx_path}")
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; use --force to replace it")
    if args.workspace <= 0:
        raise SystemExit("--workspace must be positive")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise SystemExit("install requirements-tensorrt.txt in the TensorRT 11 environment") from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TensorRT 10+ is always explicit-batch; the EXPLICIT_BATCH flag was removed.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise SystemExit(f"could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace * 1024**3))
    # TensorRT 11 uses strongly typed networks and removed BuilderFlag.FP16.
    # Keep the ONNX model's FP32 type rather than introducing a quality-changing
    # conversion at export time.
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT InSwapper engine build failed; inspect TensorRT logs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    print(f"created {output} with TensorRT {trt.__version__}")


if __name__ == "__main__":
    main()
