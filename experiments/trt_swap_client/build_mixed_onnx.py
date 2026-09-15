#!/usr/bin/env python3
"""Build a surgical mixed-precision InSwapper ONNX (experiment, issue #21).

Inserts FP32->FP16 Casts around heavy Conv data inputs (and FP16 weights),
with a Cast back to FP32 on outputs.  Unlike whole-graph FP16 conversion this
leaves Resize/shape/AdaIN statistics paths untouched.  Layer profile
(3090, TRT 11.1) showed 16 Convs taking ~83% of forward time; converting 19
Convs cut forward 14.25ms -> 5.77ms with real-face MAE 1.4e-3 vs the FP32
engine (random-noise MAE looks worse; validate on real blobs).

Usage:
    python -m experiments.trt_swap_client.build_mixed_onnx \
        --onnx models/face_swap/inswapper_128.onnx \
        --output models/face_swap/inswapper_128_mixed19.onnx \
        [--keep-fp32 Conv_590,Conv_594]
    python -m experiments.trt_swap_client.export_swapper \
        --onnx models/face_swap/inswapper_128_mixed19.onnx \
        --output models/face_swap/inswapper_128_trt11_mixed.engine \
        --mixed-base models/face_swap/inswapper_128.onnx --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Conv layers taking ~83% of forward time on 3090 (TRT layer profile).
HEAVY_CONVS = [
    "Conv_590", "Conv_594", "Conv_287", "Conv_557", "Conv_377", "Conv_107",
    "Conv_467", "Conv_197", "Conv_332", "Conv_512", "Conv_62", "Conv_152",
    "Conv_422", "Conv_242", "Conv_612", "Conv_44", "Conv_46", "Conv_596",
    "Conv_42",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep-fp32",
        default="",
        help="comma-separated Conv names left in FP32 (default: convert all heavy)",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; use --force to replace it")
    keep = {part.strip() for part in args.keep_fp32.split(",") if part.strip()}
    targets = [name for name in HEAVY_CONVS if name not in keep]
    unknown = keep - set(HEAVY_CONVS)
    if unknown:
        raise SystemExit(f"--keep-fp32 names not in heavy list: {sorted(unknown)}")

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {item.name: item for item in graph.initializer}
    converted = 0
    for node in list(graph.node):
        if node.op_type != "Conv" or node.name not in targets:
            continue
        for const_in in list(node.input[1:]):
            if const_in not in inits:
                continue
            weights = numpy_helper.to_array(inits[const_in]).astype(np.float16)
            graph.initializer.remove(inits[const_in])
            fresh = numpy_helper.from_array(weights, name=const_in)
            graph.initializer.append(fresh)
            inits[const_in] = fresh
        data_cast = f"{node.name}/fp16_in"
        out_cast = f"{node.name}/fp32_out"
        orig_out = node.output[0]
        cast_in = helper.make_node("Cast", [node.input[0]], [data_cast], to=TensorProto.FLOAT16)
        cast_out = helper.make_node("Cast", [out_cast], [orig_out], to=TensorProto.FLOAT)
        node.input[0] = data_cast
        node.output[0] = out_cast
        index = list(graph.node).index(node)
        graph.node.insert(index, cast_in)
        graph.node.insert(index + 2, cast_out)
        converted += 1
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    recipe = {
        "base_model_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "converted_convs": sorted(targets),
        "kept_fp32_convs": sorted(keep),
    }
    output.with_suffix(output.suffix + ".recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"converted {converted} convs to FP16 datatypes (kept FP32: {sorted(keep)})")


if __name__ == "__main__":
    main()
