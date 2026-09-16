#!/usr/bin/env python3
"""Build a TensorRT 11 engine for inswapper_128 on the current host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
DEFAULT_ENGINE = ROOT / "models" / "face_swap" / "inswapper_128_trt11.engine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--workspace", type=float, default=2.0, help="workspace GiB")
    parser.add_argument(
        "--builder-opt-level",
        type=int,
        default=3,
        help="TensorRT builder optimization level 0..5 (default 3 keeps stock tactics)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1,
        help="multi-face batch rows (default 1 keeps the static engine; >1 builds "
        "a dynamic-batch engine with min=1/opt=max/max=max profile)",
    )
    parser.add_argument(
        "--static-batch",
        type=int,
        default=0,
        help="fixed batch rows without a profile (0 disables; best tactics but "
        "only runs exactly this batch size)",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32",),
        default="fp32",
        help="FP32 IO is the supported InSwapper quality path; mixed precision "
        "comes from build_mixed_onnx.py graph surgery, not a builder flag "
        "(TRT 11 removed the FP16 flag and per-layer precision API)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mixed-base",
        type=Path,
        default=None,
        help="official FP32 ONNX when --onnx is a mixed-precision build from "
        "build_mixed_onnx.py; records provenance so the runtime gate can accept it",
    )
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
    if args.max_batch < 1:
        raise SystemExit("--max-batch must be >= 1")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise SystemExit("install requirements-tensorrt.txt in the TensorRT 11 environment") from error

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # Keep the vendor ONNX graph intact.  Converting all graph constants and
    # Resize inputs to FP16 changes Resize semantics on some TRT 11 parsers.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise SystemExit(f"could not parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace * 1024**3))
    if not 0 <= args.builder_opt_level <= 5:
        raise SystemExit("--builder-opt-level must be in 0..5")
    if args.builder_opt_level != 3:
        config.builder_optimization_level = args.builder_opt_level
    batch_range = {"min": 1, "opt": 1, "max": 1}
    if args.static_batch > 1:
        if args.max_batch > 1:
            raise SystemExit("use only one of --max-batch and --static-batch")
        batch_range = {
            "min": args.static_batch,
            "opt": args.static_batch,
            "max": args.static_batch,
        }
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            shape = tuple(tensor.shape)
            if len(shape) == 4:
                tensor.shape = (args.static_batch, *shape[1:])
            elif len(shape) == 2:
                tensor.shape = (args.static_batch, shape[1])
            else:
                raise SystemExit(f"unexpected InSwapper input rank: {tensor.name} {shape}")
    elif args.max_batch > 1:
        batch_range = {"min": 1, "opt": args.max_batch, "max": args.max_batch}
        _mark_dynamic_batch(network, trt)
        profile = builder.create_optimization_profile()
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            shape = tuple(tensor.shape)
            if len(shape) == 4:
                profile.set_shape(
                    tensor.name, (1, *shape[1:]), (args.max_batch, *shape[1:]), (args.max_batch, *shape[1:])
                )
            elif len(shape) == 2:
                profile.set_shape(
                    tensor.name, (1, shape[1]), (args.max_batch, shape[1]), (args.max_batch, shape[1])
                )
            else:
                raise SystemExit(f"unexpected InSwapper input rank: {tensor.name} {shape}")
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT InSwapper engine build failed; inspect TensorRT logs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    manifest: dict[str, object] = {
        "format": 1,
        "model_sha256": _sha256(onnx_path),
        "engine_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tensorrt_version": trt.__version__,
        "precision": args.precision,
        "preserve_onnx_fp32_io": True,
        "batch_range": batch_range,
    }
    if args.mixed_base is not None:
        base = args.mixed_base.expanduser().resolve()
        if not base.is_file():
            raise SystemExit(f"--mixed-base ONNX not found: {base}")
        manifest["base_model_sha256"] = _sha256(base)
        recipe_path = onnx_path.with_suffix(onnx_path.suffix + ".recipe.json")
        if recipe_path.is_file():
            manifest["mixed_recipe"] = json.loads(recipe_path.read_text())
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"created {output} with TensorRT {trt.__version__} "
        f"({args.precision}, batch {batch_range['min']}..{batch_range['max']})"
    )


def _mark_dynamic_batch(network: object, trt: object) -> None:
    """Mark dim 0 of the image/latent inputs dynamic; other dims stay fixed."""

    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(tensor.shape)
        if len(shape) == 4:
            tensor.shape = (-1, *shape[1:])
        elif len(shape) == 2:
            tensor.shape = (-1, shape[1])
        else:
            raise SystemExit(f"unexpected InSwapper input rank: {tensor.name} {shape}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
