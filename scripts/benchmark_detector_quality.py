#!/usr/bin/env python3
"""Evaluate RF-DETR on a local YOLO polygon dataset (requires pycocotools).

Boxes and polygons are rasterized in a common 624-square coordinate system.
Only explicitly selected classes are scored; this is important for datasets
whose labels cover faces or plates exclusively. No dataset split is inferred.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.rfdetr import RESOLUTION, RFDETRDetector
from service.runtime import DEFAULT_CHECKPOINT, DEFAULT_ENGINE, sha256_file


def dataset_items(root):
    items = []
    for image in sorted((root / "images").rglob("*")):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        label = (root / "labels" / image.relative_to(root / "images")).with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"missing label: {label}")
        items.append((image, label))
    if not items:
        raise ValueError("no labeled images")
    return items


def mask_annotation(polygon):
    from pycocotools import mask as coco_mask

    pixels = np.zeros((RESOLUTION, RESOLUTION), np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(pixels, [np.rint(polygon).astype(np.int32)], 1)
    encoded = coco_mask.encode(np.asfortranarray(pixels))
    area = float(coco_mask.area(encoded))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded, area


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="+", choices=(0, 1), required=True)
    parser.add_argument("--backend", choices=("pytorch", "tensorrt"), default="tensorrt")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument(
        "--predictions", type=Path, help="Previously exported detector-neutral rows"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    items = dataset_items(args.dataset)
    sources, images, annotations, predictions = [], [], [], []
    rows = json.loads(args.predictions.read_text()) if args.predictions else None
    if rows is not None and len(rows) != len(items):
        raise ValueError("prediction image count mismatch")
    artifact = args.checkpoint if args.backend == "pytorch" else args.engine
    model = None
    if rows is None:
        import torch

        torch.set_num_threads(1)
        model = RFDETRDetector(artifact, args.backend, "0")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    for index, (path, label) in enumerate(items, 1):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unreadable image: {path}")
        scale = np.asarray((RESOLUTION / image.shape[1], RESOLUTION / image.shape[0]))
        images.append({"id": index, "width": RESOLUTION, "height": RESOLUTION})
        sources.append(
            {"image": str(path), "sha256": sha256_file(path), "label_sha256": sha256_file(label)}
        )
        for line in label.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            cls = int(fields[0])
            if cls not in args.classes:
                continue
            polygon = np.asarray(fields[1:], np.float32).reshape(-1, 2) * RESOLUTION
            if len(polygon) < 3:
                raise ValueError(f"expected polygon labels: {label}")
            mask, area = mask_annotation(polygon)
            left, right = polygon.min(axis=0), polygon.max(axis=0)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": cls,
                    "bbox": [*left.tolist(), *(right - left).tolist()],
                    "segmentation": mask,
                    "area": area,
                    "iscrowd": 0,
                }
            )
        if rows is None:
            result = model.predict(image)
            boxes, polygons = result.boxes.data, result.masks.xy
        else:
            row = rows[index - 1]
            if row["file"] != str(path) or row["sha256"] != sources[-1]["sha256"]:
                raise ValueError("prediction input provenance mismatch")
            boxes, polygons = np.asarray(row["boxes"], np.float32).reshape(-1, 6), row["polygons"]
        for i, box in enumerate(boxes):
            cls = int(box[5])
            if cls not in args.classes:
                continue
            mask, _ = mask_annotation(np.asarray(polygons[i]).reshape(-1, 2) * scale)
            left, right = box[:2] * scale, box[2:4] * scale
            predictions.append(
                {
                    "image_id": index,
                    "category_id": cls,
                    "score": float(box[4]),
                    "bbox": [*left.tolist(), *(right - left).tolist()],
                    "segmentation": mask,
                }
            )
        if index % 100 == 0:
            print(f"evaluated {index}/{len(items)}", flush=True)
    truth = COCO()
    truth.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cls, "name": ("face", "number_plate")[cls]} for cls in args.classes],
    }
    metrics = {}
    with contextlib.redirect_stdout(io.StringIO()):
        truth.createIndex()
        if predictions:
            detected = truth.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {**truth.dataset, "annotations": []}
            detected.createIndex()
        for kind in ("bbox", "segm"):
            evaluator = COCOeval(truth, detected, kind)
            evaluator.params.catIds = args.classes
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            tp = fp = 0
            for row in evaluator.evalImgs:
                if row is None or row["aRng"] != evaluator.params.areaRng[0]:
                    continue
                keep = np.asarray(row["dtScores"]) >= 0.25
                valid = keep & ~row["dtIgnore"][0]
                tp += int(((row["dtMatches"][0] > 0) & valid).sum())
                fp += int(((row["dtMatches"][0] == 0) & valid).sum())
            metrics[kind] = {
                "AP50_95": float(evaluator.stats[0]),
                "AP50": float(evaluator.stats[1]),
                "AP75": float(evaluator.stats[2]),
                "confidence": 0.25,
                "IoU": 0.5,
                "TP": tp,
                "FP": fp,
                "FN": len(annotations) - tp,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(len(annotations), 1),
            }
    report = {
        "dataset": str(args.dataset),
        "classes": args.classes,
        "image_count": len(items),
        "instances": len(annotations),
        "raster_size": RESOLUTION,
        "metrics": metrics,
        "source_digest": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "sources": sources,
        "prediction_source": str(args.predictions) if rows is not None else args.backend,
        "artifact_sha256": sha256_file(args.predictions or artifact),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
