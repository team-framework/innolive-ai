#!/usr/bin/env python3
"""Compare matched RF-DETR FP32/TRT boxes and masks on supplied images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.rfdetr import RFDETRDetector, decode
from service.runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_ENGINE,
    RuntimeConfig,
    sha256_file,
    validate_engine,
)


def polygon_mask(polygon, shape):
    result = np.zeros((624, 624), np.uint8)
    points = np.asarray(polygon) * (624 / shape[1], 624 / shape[0])
    if len(points) >= 3:
        cv2.fillPoly(result, [points.round().astype(np.int32)], 1)
    return result


def compare(reference, candidate, shape):
    a, b = reference.boxes, candidate.boxes
    refs, candidates = np.flatnonzero(a.conf >= 0.25), np.flatnonzero(b.conf >= 0.25)
    available = set(candidates.tolist())
    results = []
    for i in refs:
        choices = [j for j in available if a.cls[i] == b.cls[j]]
        if not choices:
            results.append({"reference": int(i), "matched": False})
            continue
        left, right = a.xyxy[i], b.xyxy[choices]
        intersection = np.maximum(
            0, np.minimum(left[2:], right[:, 2:]) - np.maximum(left[:2], right[:, :2])
        ).prod(axis=1)
        union = (
            (left[2:] - left[:2]).prod() + (right[:, 2:] - right[:, :2]).prod(axis=1) - intersection
        )
        ious = intersection / np.maximum(union, 1e-9)
        best = int(np.argmax(ious))
        j = choices[best]
        available.remove(j)
        first = polygon_mask(reference.masks.xy[i], shape).astype(bool)
        second = polygon_mask(candidate.masks.xy[j], shape).astype(bool)
        mask_iou = float((first & second).sum() / max(1, (first | second).sum()))
        results.append(
            {
                "class_id": int(a.cls[i]),
                "box_iou": float(ious[best]),
                "mask_iou": mask_iou,
                "score_delta": float(abs(a.conf[i] - b.conf[j])),
                "matched": bool(ious[best] >= 0.9 and mask_iou >= 0.9),
            }
        )
    return {
        "reference_count": len(refs),
        "candidate_count": len(candidates),
        "unmatched_candidates": len(available),
        "matches": results,
        "passed": len(refs) == len(candidates)
        and not available
        and all(r["matched"] for r in results),
    }


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    manifest = validate_engine(RuntimeConfig(checkpoint=args.checkpoint, engine=args.engine))
    models = {
        b: RFDETRDetector(p, b, "0")
        for b, p in [("pytorch", args.checkpoint), ("tensorrt", args.engine)]
    }
    report = {"manifest": manifest, "inputs": []}
    for path in args.images:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"unreadable image: {path}")
        predictions, timing = {}, {}
        for name, model in models.items():
            for _ in range(5):
                model.predict(frame)
            times = []
            for _ in range(30):
                started = time.perf_counter()
                predictions[name] = model.predict(frame)
                list(
                    predictions[name].masks.xy
                )  # Include all candidate contours in this microbenchmark.
                times.append((time.perf_counter() - started) * 1000)
            timing[name] = {
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
            }
        result = compare(predictions["pytorch"], predictions["tensorrt"], frame.shape)
        raw = models["tensorrt"].raw(frame)
        full = decode(raw["dets"], raw["labels"], raw["masks"], frame.shape)
        optimized = predictions["tensorrt"]
        result["selective_transfer_exact"] = bool(
            np.array_equal(full.boxes.data, optimized.boxes.data)
            and len(full.masks.xy) == len(optimized.masks.xy)
            and all(
                np.array_equal(a, b) for a, b in zip(full.masks.xy, optimized.masks.xy, strict=True)
            )
        )
        result["passed"] = result["passed"] and result["selective_transfer_exact"]
        report["inputs"].append(
            {"file": str(path), "sha256": sha256_file(path), "timing": timing, **result}
        )
    report["passed"] = all(item["passed"] for item in report["inputs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
