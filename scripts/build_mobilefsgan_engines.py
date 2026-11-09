#!/usr/bin/env python3
"""Build device-specific K22741 engines without overwriting the supplied bundle.

Mixed FP16 is opt-in and must pass benchmark_mobilefsgan parity checks before use.
Identity encoder C stays FP32; G normalization/reductions remain FP32.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.mobilefsgan import DEFAULT_BUNDLE, sha256


def main(args):
    import onnx
    import tensorrt as trt
    import torch

    if args.output.exists():
        raise ValueError("choose a new output directory; existing engines are never overwritten")
    model_manifest = json.loads((args.bundle / "model_manifest.json").read_text())
    for name in ("C", "G"):
        if (
            sha256(args.bundle / f"onnx/{name}_fp32.onnx")
            != model_manifest["asset_sha256"][f"onnx/{name}_fp32.onnx"]
        ):
            raise ValueError("MobileFSGAN source ONNX hash mismatch")
    args.output.mkdir(parents=True)
    manifest = {
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "precision": args.precision,
        "model_onnx_sha256": {},
        "engines": {},
        "engine_files": {},
        "compiled_onnx_sha256": {},
        "receipts": [],
        "max_batch": args.max_batch,
    }
    logger = trt.Logger(trt.Logger.WARNING)
    for name in ("C", "G"):
        source = args.bundle / f"onnx/{name}_fp32.onnx"
        graph = onnx.load(str(source))
        precision = "fp32"
        if name == "G" and args.precision == "mixed_fp16":
            from onnxconverter_common import float16

            precision = "mixed_fp16"
            graph = float16.convert_float_to_float16(
                graph,
                keep_io_types=True,
                op_block_list=[
                    *float16.DEFAULT_OP_BLOCK_LIST,
                    "ReduceMean",
                    "ReduceSum",
                    "Sqrt",
                    "Pow",
                    "Div",
                ],
            )
        graph_path = args.output / f"{name}_{precision}.onnx"
        onnx.save(graph, str(graph_path))
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(graph_path.read_bytes()):
            raise RuntimeError(
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 3
        config.max_aux_streams = 0
        profile = builder.create_optimization_profile()
        maximum = args.max_batch if name == "G" else 1
        shapes = {}
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            tail = tuple(tensor.shape)[1:]
            shapes[tensor.name] = [(1, *tail), (min(4, maximum), *tail), (maximum, *tail)]
            profile.set_shape(tensor.name, *shapes[tensor.name])
        config.add_optimization_profile(profile)
        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"MobileFSGAN {name} build failed")
        plan = args.output / f"{name}_{precision}.plan"
        plan.write_bytes(bytes(serialized))
        manifest["engine_files"][name] = plan.name
        manifest["engines"][plan.name] = sha256(plan)
        manifest["model_onnx_sha256"][name] = sha256(source)
        manifest["compiled_onnx_sha256"][graph_path.name] = sha256(graph_path)
        manifest["receipts"].append(
            {"name": name, "seconds": time.perf_counter() - start, "shapes": shapes}
        )
        print(json.dumps(manifest["receipts"][-1]), flush=True)
    if args.with_yunet:
        source = args.bundle / "detector/yunet.onnx"
        if sha256(source) != model_manifest["asset_sha256"]["detector/yunet.onnx"]:
            raise ValueError("YuNet source hash mismatch")
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(source.read_bytes()):
            raise RuntimeError("YuNet ONNX parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.max_aux_streams = 0
        profile = builder.create_optimization_profile()
        tensor = network.get_input(0)
        tensor.shape = (1, 3, -1, -1)
        profile.set_shape(tensor.name, (1, 3, 32, 32), (1, 3, 480, 640), (1, 3, 640, 640))
        config.add_optimization_profile(profile)
        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("YuNet TensorRT build failed")
        path = args.output / "yunet_fp32.plan"
        path.write_bytes(bytes(serialized))
        manifest["landmarks_engine"] = path.name
        manifest["landmarks_onnx_sha256"] = sha256(source)
        manifest["engines"][path.name] = sha256(path)
        manifest["receipts"].append({"name": "YuNet", "seconds": time.perf_counter() - start})
        print(json.dumps(manifest["receipts"][-1]), flush=True)
    (args.output / "engine_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "mixed_fp16"), default="fp32")
    parser.add_argument("--max-batch", type=int, choices=range(1, 9), default=8)
    parser.add_argument(
        "--with-yunet",
        action="store_true",
        help="also accelerate target landmarks; source alignment stays CPU",
    )
    main(parser.parse_args())
