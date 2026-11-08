#!/usr/bin/env python3
"""Compare CPU/TensorRT target landmarks on moving 1/2/4/8-face grids."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_face_capacity import grid_frames, portraits
from service.face_presets import DEFAULT_PRESET_MANIFEST
from service.mobilefsgan import DEFAULT_BUNDLE, MobileFSGANRenderer, alignment, crop_face


def main(args):
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    renderer = MobileFSGANRenderer(args.bundle, engine_dir=args.engine_dir)
    images = portraits(args.manifest)
    rows = []
    for width, height in ((640, 480), (1280, 720), (1920, 1080)):
        for count in (1, 2, 4, 8):
            for phase, image in enumerate(grid_frames(images, count, width, height)[::10]):
                before = time.perf_counter()
                boxes, points = renderer.source_detector.detect(image)
                cpu_ms = (time.perf_counter() - before) * 1000
                # Exclude the first shape allocation from timings.
                renderer.detector.detect(image)
                before = time.perf_counter()
                fast_boxes, fast_points = renderer.detector.detect(image)
                gpu_ms = (time.perf_counter() - before) * 1000
                if len(boxes) != count or len(fast_boxes) != count:
                    raise ValueError(
                        f"landmark face count mismatch {width}x{height}: {count}, {len(boxes)}, {len(fast_boxes)}"
                    )
                errors, crop_mae, selected = [], [], set()
                for box, source_points in zip(boxes, points, strict=True):
                    distances = np.abs(fast_boxes[:, :4] - box[:4]).max(axis=1)
                    index = int(np.argmin(distances))
                    if index in selected or distances[index] > 1:
                        raise ValueError("landmark box association/coordinate gate failed")
                    selected.add(index)
                    error = float(np.abs(source_points - fast_points[index]).max())
                    errors.append(error)
                    a = crop_face(image, alignment(source_points))
                    b = crop_face(image, alignment(fast_points[index]))
                    crop_mae.append(float(np.abs(a.astype(float) - b).mean() / 255))
                if max(errors) > 1 or max(crop_mae) > 0.005:
                    raise ValueError("landmark/crop parity gate failed")
                rows.append(
                    {
                        "resolution": [width, height],
                        "faces": count,
                        "phase": phase,
                        "max_landmark_px": max(errors),
                        "max_crop_mae": max(crop_mae),
                        "cpu_ms": cpu_ms,
                        "trt_ms": gpu_ms,
                    }
                )
    report = {
        "samples": len(rows),
        "all_face_counts_equal": True,
        "max_landmark_px": max(r["max_landmark_px"] for r in rows),
        "max_crop_mae": max(r["max_crop_mae"] for r in rows),
        "cpu_p50_ms": float(np.median([r["cpu_ms"] for r in rows])),
        "trt_p50_ms": float(np.median([r["trt_ms"] for r in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PRESET_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
