#!/usr/bin/env python3
"""Compare local MobileFaceSwap preprocessing/output with a supplied upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.face_metadata import FaceAttributes
from service.face_presets import FacePreset
from service.mobilefaceswap import (
    DEFAULT_MODEL_DIR,
    MobileFaceSwapRenderer,
    alignment,
    image_tensor,
    paste_back,
    source_tensor,
)


def module(path):
    spec = importlib.util.spec_from_file_location("reference_" + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify(args):
    import paddle

    upstream_model = module(args.upstream / "models/model.py")
    upstream_arcface = module(args.upstream / "models/arcface.py")
    upstream_align = module(args.upstream / "utils/align_face.py")
    upstream_util = module(args.upstream / "utils/util.py")
    renderer = MobileFaceSwapRenderer(args.model_dir)
    crops, matrices = [], []
    errors = {}
    for name, path in (("source", args.source), ("target", args.target)):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot decode {path}")
        boxes, points = renderer.detector.detect(image)
        if len(boxes) != 1:
            raise ValueError(f"{name} must contain exactly one detectable face")
        matrix = alignment(points[0])
        reference = upstream_align.align_with_five_points(points[0])
        errors[name + "_alignment"] = float(np.max(np.abs(matrix - reference)))
        crop = cv2.warpAffine(image, matrix, (224, 224))
        crops.append(crop)
        matrices.append(matrix)
    reference_source = upstream_util.cv2paddle(cv2.resize(crops[0], (112, 112)))
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    reference_source = (reference_source - mean) / std
    errors["source_tensor"] = float(
        np.max(np.abs(source_tensor(crops[0]) - reference_source.numpy()))
    )
    reference_target = upstream_util.cv2paddle(crops[1])
    errors["target_tensor"] = float(
        np.max(np.abs(image_tensor(crops[1]) - reference_target.numpy()))
    )
    identity_net = upstream_arcface.ResNet(upstream_arcface.IRBlock, [3, 4, 23, 3])
    identity_net.set_state_dict(paddle.load(str(args.model_dir / "arcface.pdparams")))
    identity_net.eval()
    with paddle.no_grad():
        embedding, features = identity_net(reference_source)
        reference_model = upstream_model.FaceSwap(use_gpu=False)
        reference_model.set_model_param(
            upstream_model.l2_norm(embedding),
            features,
            paddle.load(str(args.model_dir / "MobileFaceSwap_224.pdparams")),
        )
        reference_model.eval()
        expected, expected_mask = reference_model(reference_target)
        preset = FacePreset(FaceAttributes("female", "20s", "off", "none"), 1, args.source)
        local_model = renderer.source_model(preset)
        actual, mask = local_model(paddle.to_tensor(image_tensor(crops[1])))
        for name, value, reference in (
            ("generated", actual.numpy(), expected.numpy()),
            ("mask", mask.numpy(), expected_mask.numpy()),
        ):
            errors[name] = float(np.max(np.abs(value - reference)))
        times = []
        for _ in range(5):
            started = time.perf_counter()
            local_model(paddle.to_tensor(image_tensor(crops[1])))
            times.append((time.perf_counter() - started) * 1000)
    image = cv2.imread(str(args.target))
    height, width = image.shape[:2]
    polygon = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    output = paste_back(image, actual.numpy(), mask.numpy(), matrices[1], polygon)
    reference_output = upstream_align.dealign(
        upstream_util.paddle2cv(expected),
        image,
        upstream_align.back_matrix(matrices[1]),
        expected_mask.numpy()[0].transpose(1, 2, 0),
    )
    errors["paste_back_uint8"] = int(
        np.max(np.abs(output.astype(int) - reference_output.astype(int)))
    )
    torch_times = []
    if args.torch_provider:
        import torch

        from service.mobilefaceswap_torch import from_paddle

        model = from_paddle(local_model).to(args.torch_provider, memory_format=torch.channels_last)
        values = torch.from_numpy(image_tensor(crops[1])).to(
            args.torch_provider, memory_format=torch.channels_last
        )
        with torch.inference_mode():
            for index in range(18):
                started = time.perf_counter()
                generated, mask = model(values)
                generated, mask = generated.cpu().numpy(), mask.cpu().numpy()
                if index >= 3:
                    torch_times.append((time.perf_counter() - started) * 1000)
        errors["torch_generated"] = float(np.max(np.abs(generated - actual.numpy())))
        errors["torch_mask"] = float(np.max(np.abs(mask - expected_mask.numpy())))
    report = {
        "paddle_version": paddle.__version__,
        "provider": "cpu",
        "source": str(args.source),
        "target": str(args.target),
        "max_abs_errors": errors,
        "generator_only_ms": times,
        "torch_provider": args.torch_provider,
        "torch_generator_only_ms": torch_times,
        "best_pt_sha256": hashlib.sha256((ROOT / "models/best.pt").read_bytes()).hexdigest(),
        "upstream": str(args.upstream),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "generated.jpg"), output)
    (args.output_dir / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(value > (1 if name == "paste_back_uint8" else 1e-4) for name, value in errors.items()):
        raise SystemExit("MobileFaceSwap upstream parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--torch-provider", choices=("mps", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/mobilefaceswap/parity")
    verify(parser.parse_args())
