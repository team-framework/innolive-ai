#!/usr/bin/env python3
"""Paired serial/graph+batch K22741 measurements on identical real face crops.

These are renderer measurements, not streaming FPS. Use benchmark_face_capacity
for the actual gRPC detection/tracking/metadata/swap/composition path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.face_presets import DEFAULT_PRESET_MANIFEST, PresetCatalog
from service.mobilefsgan import (
    DEFAULT_BUNDLE,
    MobileFSGANRenderer,
    alignment,
    crop_face,
    image_tensor,
)


def quantiles(values):
    return {
        name: round(float(np.percentile(values, p)), 3) for name, p in (("p50", 50), ("p95", 95))
    }


def pixels(value):
    return np.rint((value.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]


def main(args):
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = PresetCatalog(args.manifest)
    sources = [
        p
        for p in catalog.presets.values()
        if p.attributes.age == "20s"
        and p.attributes.exp == "none"
        and p.attributes.glasses == "off"
    ][:8]
    if len(sources) < 2:
        raise ValueError("at least two distinct source portraits required")
    baseline = MobileFSGANRenderer(args.bundle, max_batch=1, cuda_graphs=False)
    optimized = MobileFSGANRenderer(
        args.bundle, engine_dir=args.engine_dir, max_batch=8, cuda_graphs=True
    )
    crops = []
    for preset in sources:
        image = cv2.imread(str(preset.image))
        boxes, points = baseline.detector.detect(image)
        if len(boxes) != 1:
            raise ValueError("benchmark portrait must contain exactly one face")
        crops.append(crop_face(image, alignment(points[0])))
    sources = sources[1:] + sources[:1]
    baseline.warmup(sources)
    optimized.warmup(sources)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "engine_manifest": optimized.manifest,
        "scope": "aligned crops, source embeddings cached; H2D/G/D2H included; no detection/network/paste",
        "cases": [],
    }
    for count in (1, 2, 4, 8):
        batch_crops = [crops[i % len(crops)] for i in range(count)]
        batch_sources = [sources[i % len(sources)] for i in range(count)]
        variants = {"serial": baseline, "batch_graph": optimized}
        outputs = {
            name: renderer.render_crops(batch_crops, batch_sources)
            for name, renderer in variants.items()
        }
        for result in outputs.values():
            for value in result:
                if isinstance(value, Exception):
                    raise value
        error = np.abs(np.stack(outputs["serial"]) - np.stack(outputs["batch_graph"])) / 2
        numeric = {
            "mae": float(error.mean()),
            "p99": float(np.quantile(error, 0.99)),
            "max": float(error.max()),
        }
        if numeric["mae"] > 5e-4 or numeric["p99"] > 5e-3:
            raise ValueError(f"batched output parity failed: {numeric}")
        times = {name: [] for name in variants}
        for repeat in range(args.repeats + 10):
            for name in list(variants)[:: 1 if repeat % 2 else -1]:
                started = time.perf_counter()
                result = variants[name].render_crops(batch_crops, batch_sources)
                elapsed = (time.perf_counter() - started) * 1000
                if any(isinstance(v, Exception) for v in result):
                    raise RuntimeError("renderer failed during benchmark")
                if repeat >= 10:
                    times[name].append(elapsed)
        case = {
            "faces": count,
            "parity": numeric,
            "ms": {name: quantiles(values) for name, values in times.items()},
        }
        report["cases"].append(case)
        cv2.imwrite(
            str(args.output / f"crops-{count}.jpg"),
            np.concatenate(
                [
                    np.concatenate(
                        [
                            batch_crops[i],
                            pixels(outputs["serial"][i]),
                            pixels(outputs["batch_graph"][i]),
                        ],
                        axis=1,
                    )
                    for i in range(count)
                ],
                axis=0,
            ),
        )
        print(json.dumps(case), flush=True)
    if args.onnx_reference:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        sessions = {
            n: ort.InferenceSession(
                str(args.bundle / f"onnx/{n}_fp32.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for n in ("C", "G")
        }
        source = image_tensor(crops[1])[None]
        target = image_tensor(crops[0])[None]
        z = sessions["C"].run(None, {"source_img": source})[0]
        expected = sessions["G"].run(None, {"target_img": target, "source_z": z})[0]
        actual_z = optimized.c.numpy({"source_img": source})
        actual = optimized.g.numpy({"target_img": target, "source_z": actual_z})
        error = np.abs(actual - expected) / 2
        cosine = float(
            np.dot(z.ravel(), actual_z.ravel()) / (np.linalg.norm(z) * np.linalg.norm(actual_z))
        )
        report["onnx_reference"] = {
            "mae": float(error.mean()),
            "p99": float(np.quantile(error, 0.99)),
            "max": float(error.max()),
            "C_cosine": cosine,
        }
        if error.mean() > 5e-4 or np.quantile(error, 0.99) > 5e-3 or cosine < 0.9999:
            raise ValueError("ONNX reference parity failed")
    if args.quality_samples:
        candidates = list(catalog.presets.values())
        np.random.default_rng(100).shuffle(candidates)
        quality_crops, quality_sources, skipped = [], [], []
        for i, preset in enumerate(candidates):
            image = cv2.imread(str(preset.image))
            boxes, points = baseline.detector.detect(image)
            if len(boxes) != 1:
                skipped.append(preset.key)
                continue
            try:
                crop = crop_face(image, alignment(points[0]))
            except ValueError:
                skipped.append(preset.key)
                continue
            rotation = cv2.getRotationMatrix2D((128, 128), (i % 3 - 1) * 8, 1)
            quality_crops.append(
                cv2.warpAffine(crop, rotation, (256, 256), borderMode=cv2.BORDER_REFLECT_101)
            )
            quality_sources.append(sources[i % len(sources)])
            if len(quality_crops) == args.quality_samples:
                break
        if len(quality_crops) != args.quality_samples:
            raise ValueError("insufficient valid quality samples")
        reference = baseline.render_crops(quality_crops, quality_sources)
        candidate = optimized.render_crops(quality_crops, quality_sources)
        rows, errors = [], []
        for i, (ref, value) in enumerate(zip(reference, candidate, strict=True)):
            if isinstance(ref, Exception):
                raise ref
            if isinstance(value, Exception):
                raise value
            error = np.abs(ref - value) / 2
            errors.append(
                {
                    "mae": float(error.mean()),
                    "p99": float(np.quantile(error, 0.99)),
                    "max": float(error.max()),
                }
            )
            if i < 12:
                rows.append(np.concatenate([quality_crops[i], pixels(ref), pixels(value)], axis=1))
        report["quality"] = {
            "samples": len(errors),
            "skipped": skipped,
            "max_sample_mae": max(e["mae"] for e in errors),
            "max_sample_p99": max(e["p99"] for e in errors),
            "max_pixel_error": max(e["max"] for e in errors),
            "errors": errors,
        }
        if report["quality"]["max_sample_mae"] > 5e-4 or report["quality"]["max_sample_p99"] > 5e-3:
            (args.output / "failed-quality.json").write_text(json.dumps(report, indent=2))
            raise ValueError("real-face mixed precision quality gate failed")
        cv2.imwrite(str(args.output / "quality.jpg"), np.concatenate(rows, axis=0))
    report["workspace_bytes"] = {
        "C": optimized.c.workspace.numel(),
        "G": optimized.g.workspace.numel(),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PRESET_MANIFEST)
    parser.add_argument(
        "--engine-dir",
        type=Path,
        help="optional rebuilt candidate; baseline always uses original FP32",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--onnx-reference", action="store_true")
    parser.add_argument("--quality-samples", type=int, default=0)
    args = parser.parse_args()
    if not 10 <= args.repeats <= 1000:
        parser.error("repeats must be 10..1000")
    if not 0 <= args.quality_samples <= 560:
        parser.error("quality-samples must be 0..560")
    main(args)
