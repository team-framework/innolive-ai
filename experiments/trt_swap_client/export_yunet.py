#!/usr/bin/env python3
"""Build a dynamic-shape TensorRT engine for YuNet on the current host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_ENGINE = ROOT / "models" / "face_detection_yunet_trt.engine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--workspace", type=float, default=1.0, help="workspace GiB")
    parser.add_argument("--opt-size", type=int, default=320)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path, output = args.onnx.expanduser().resolve(), args.output.expanduser().resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"YuNet ONNX not found: {onnx_path}")
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; use --force to replace it")
    if not 32 <= args.opt_size <= args.max_size <= 2048:
        raise SystemExit("require 32 <= opt-size <= max-size <= 2048")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise SystemExit("install requirements-tensorrt.txt in the TensorRT 11 environment") from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise SystemExit(f"could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace * 1024**3))
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(tensor.shape)
        if len(shape) != 4:
            raise SystemExit(f"unexpected YuNet input rank: {tensor.name} {shape}")
        tensor.shape = (-1, shape[1], -1, -1)
        profile.set_shape(
            tensor.name,
            (1, shape[1], 32, 32),
            (1, shape[1], args.opt_size, args.opt_size),
            (1, shape[1], args.max_size, args.max_size),
        )
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT YuNet engine build failed; inspect TensorRT logs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    manifest = {
        "format": 1,
        "model_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "engine_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tensorrt_version": trt.__version__,
        "opt_size": args.opt_size,
        "max_size": args.max_size,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"created {output} with TensorRT {trt.__version__}")


if __name__ == "__main__":
    main()
