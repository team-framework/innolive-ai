#!/usr/bin/env python3
"""Compare existing InSwapper CUDA and MobileFSGAN TensorRT on identical portraits.

Each model keeps its native alignment, resolution, normalization and identity
encoder. Timings cover cached-source, aligned BGR crops to CPU model outputs,
including preprocessing and H2D/D2H. Detection/alignment/paste/network are excluded.
This measures the deployed implementations, not an architecture-only speedup.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.face_presets import DEFAULT_PRESET_MANIFEST, PresetCatalog
from service.inswapper import DEFAULT_ARCFACE, DEFAULT_SWAPPER, DEFAULT_YUNET, InSwapperRenderer
from service.inswapper import alignment as inswapper_alignment
from service.mobilefsgan import DEFAULT_BUNDLE, MobileFSGANRenderer, alignment, crop_face, sha256


def stats(values):
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def main(args):
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    ort.preload_dlls()
    args.output.mkdir(parents=True, exist_ok=True)
    old = InSwapperRenderer(args.inswapper, args.arcface, args.yunet, "cuda")
    for session in (old.swapper, old.arcface):
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("InSwapper CUDA did not initialize; refuse a CPU comparison")
    new = MobileFSGANRenderer(args.bundle, engine_dir=args.engine_dir)
    portraits = [
        p
        for p in PresetCatalog(args.manifest).presets.values()
        if p.attributes.age == "20s"
        and p.attributes.exp == "none"
        and p.attributes.glasses == "off"
    ][:8]
    if len(portraits) != 8:
        raise ValueError("eight source/target portraits required")
    sources = portraits[1:] + portraits[:1]
    native_old, native_new = [], []
    for preset in portraits:
        image = cv2.imread(str(preset.image))
        faces = old.detect(image)
        if len(faces) != 1:
            raise ValueError("comparison target must contain exactly one face")
        points = faces[0][4:14].reshape(5, 2)
        native_old.append(cv2.warpAffine(image, inswapper_alignment(points, 128), (128, 128)))
        native_new.append(crop_face(image, alignment(points)))
    for preset in sources:
        old.source_latent(preset)
    new.warmup(sources)
    report = {
        "measured_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "providers": old.swapper.get_providers(),
        "scope": __doc__,
        "source_keys": [p.key for p in sources],
        "target_keys": [p.key for p in portraits],
        "source_sha256": [p.sha256 for p in sources],
        "inswapper_sha256": sha256(args.inswapper),
        "arcface_sha256": sha256(args.arcface),
        "mobilefsgan_engine_manifest": new.manifest,
        "cases": [],
    }
    for count in (1, 4, 8):

        def inswapper(count=count):
            return [
                old.swapper.run(
                    None,
                    {
                        old.inputs[0].name: cv2.dnn.blobFromImage(
                            native_old[i], 1 / 255, (128, 128), swapRB=True
                        ),
                        old.inputs[1].name: old.source_latent(sources[i]),
                    },
                )[0][0]
                for i in range(count)
            ]

        def mobilefsgan(count=count):
            return new.render_crops(native_new[:count], sources[:count])

        functions = {"inswapper_onnx_cuda": inswapper, "mobilefsgan_tensorrt": mobilefsgan}
        times = {key: [] for key in functions}
        for repeat in range(args.warmup + args.repeats):
            for name in list(functions)[:: 1 if repeat % 2 else -1]:
                started = time.perf_counter()
                values = functions[name]()
                elapsed = (time.perf_counter() - started) * 1000
                if len(values) != count or any(
                    isinstance(v, Exception) or not np.isfinite(v).all() for v in values
                ):
                    raise ValueError(f"invalid {name} output")
                if repeat >= args.warmup:
                    times[name].append(elapsed)
        case = {"faces": count, "ms": {k: stats(v) for k, v in times.items()}}
        report["cases"].append(case)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inswapper", type=Path, default=DEFAULT_SWAPPER)
    parser.add_argument("--arcface", type=Path, default=DEFAULT_ARCFACE)
    parser.add_argument("--yunet", type=Path, default=DEFAULT_YUNET)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PRESET_MANIFEST)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 10 <= args.repeats <= 500 or not 10 <= args.warmup <= 100:
        parser.error("repeats must be 10..500 and warmup 10..100")
    main(args)
