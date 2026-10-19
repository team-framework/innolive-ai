#!/usr/bin/env python3
"""Build an opt-in SCRFD engine on the target NVIDIA GPU (TensorRT 10/11)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.scrfd_tensorrt import sha256


def build(args):
    import onnx
    import tensorrt as trt
    import torch

    model = onnx.load(str(args.onnx))
    outputs = [out.name for out in model.graph.output]
    if len(model.graph.input) != 1 or len(outputs) != 9:
        raise ValueError("expected SCRFD BNKPS with nine outputs")
    dims = model.graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_param = "batch"
    dims[2].dim_value = dims[3].dim_value = 640
    for out in model.graph.output:
        out.type.tensor_type.shape.dim[0].dim_param = "anchors"
    if args.precision == "fp16":
        from onnxconverter_common import float16

        model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(model)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model.SerializeToString()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    # Disable implicit TF32; FP16 requires explicit ONNX conversion above.
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        network.get_input(0).name,
        (1, 3, 640, 640),
        (min(4, args.max_batch), 3, 640, 640),
        (args.max_batch, 3, 640, 640),
    )
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("SCRFD engine build failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(plan))
    metadata = {
        "format": 1,
        "onnx_sha256": sha256(args.onnx),
        "engine_sha256": sha256(args.output),
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(),
        "precision": args.precision,
        "outputs": outputs,
        "max_batch": args.max_batch,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    build(parser.parse_args())
