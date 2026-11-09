#!/usr/bin/env python3
"""Build the static RF-DETR Seg Small B1-624 engine on the serving NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.detection import EXPECTED_CLASS_NAMES
from service.rfdetr import MODEL_NAME, checkpoint_metadata
from service.runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_ENGINE,
    IMAGE_SIZE,
    STANDARD_PROFILE,
    sha256_file,
)


def build(checkpoint, output, device="0", workspace=4.0, force=False):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if output.suffix.lower() != ".engine":
        raise ValueError("output must end in .engine")
    if output.exists() and not force:
        raise FileExistsError(f"engine already exists: {output}; use --force")
    if workspace <= 0 or not math.isfinite(workspace):
        raise ValueError("workspace must be positive and finite")

    import tensorrt as trt
    import torch
    from rfdetr import RFDETR
    from rfdetr.export._tensorrt.exporter import fp16_source_graph

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("export requires Linux and a CUDA GPU")
    metadata = checkpoint_metadata(checkpoint)
    index = int(device.removeprefix("cuda:"))
    torch.cuda.set_device(index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rfdetr-export-", dir=output.parent) as temporary:
        model = RFDETR.from_checkpoint(str(checkpoint), device=f"cuda:{index}")
        onnx = model.export(output_dir=temporary, format="onnx", verbose=False)
        del model
        torch.cuda.empty_cache()
        major = int(trt.__version__.split(".")[0])
        # TensorRT 11 removed precision builder flags; use upstream's typed graph converter.
        from contextlib import nullcontext

        graph_context = fp16_source_graph(str(onnx)) if major >= 11 else nullcontext(str(onnx))
        with graph_context as graph:
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            flags = (
                (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if major >= 11 else 0
            )
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(graph)):
                raise RuntimeError(
                    "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                )
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace * 1024**3))
            if major < 11:
                config.set_flag(trt.BuilderFlag.FP16)
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError("RF-DETR TensorRT build failed")
            staged = Path(temporary) / output.name
            staged.write_bytes(bytes(engine))
            engine_hash = sha256_file(staged)
            staged.replace(output)
    manifest = {
        "schema_version": 2,
        "model_name": MODEL_NAME,
        "standard_profile": STANDARD_PROFILE,
        "preprocessing": "rgb-square-imagenet",
        "background_class_id": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "source_checkpoint": os.path.relpath(checkpoint, output.parent),
        "source_sha256": sha256_file(checkpoint),
        "engine_sha256": engine_hash,
        "precision": "fp16",
        "dynamic": False,
        "batch": 1,
        "image_size": IMAGE_SIZE,
        "class_names": {str(k): v for k, v in EXPECTED_CLASS_NAMES.items()},
        "workspace_gib": workspace,
        "gpu": torch.cuda.get_device_name(index),
        "torch": torch.__version__,
        "rfdetr": version("rfdetr"),
        "tensorrt": trt.__version__,
        "checkpoint_metadata": metadata,
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
