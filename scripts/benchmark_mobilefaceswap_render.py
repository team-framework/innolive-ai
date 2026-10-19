#!/usr/bin/env python3
"""Compare serial and grouped CUDA transfers on identical detected faces/presets.

Includes source loading separately, asserts exact RGB parity and unchanged pixels
outside the face polygons. These timings exclude detector, transport and encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_face_capacity import grid_frames, portraits, quantiles
from service.face_presets import PresetCatalog
from service.mobilefaceswap_torch import TorchMobileFaceSwapRenderer
from service.runtime import RuntimeConfig, RuntimeManager
from service.tracking import StreamTracker


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeManager(RuntimeConfig(backend="tensorrt", engine=args.engine, device="0"))
    renderer = TorchMobileFaceSwapRenderer(args.models, "cuda", precision="fp16")
    catalog = PresetCatalog(args.manifest)
    sources = [
        p
        for p in catalog.presets.values()
        if p.attributes.age == "20s"
        and p.attributes.exp == "none"
        and p.attributes.glasses == "off"
    ][:8]
    started = time.perf_counter()
    for preset in sources:
        renderer.torch_source_model(preset)
    preparation = (time.perf_counter() - started) * 1000
    results = []
    try:
        for count in (1, 2, 4, 8):
            tracker = StreamTracker(device="0")
            frames = grid_frames(portraits(args.input_manifest or args.manifest), count)
            samples = []
            for phase in (0, 5, 10, 15, 20, 25):
                image = cv2.resize(frames[phase], (args.width, args.height))
                objects = runtime._infer_sync(image, tracker)["objects"]
                faces = [
                    item
                    for item in objects
                    if item["class_name"] == "face" and not item.get("held")
                ]
                if len(faces) != count:
                    raise RuntimeError(
                        f"expected {count} faces, detected {len(faces)} at phase {phase}"
                    )
                target = renderer.prepare_frame(image)
                target.prime([item["bbox"] for item in faces])
                # Resolve fallback detections before timing either composition path.
                for item in faces:
                    target.roi_points[tuple(item["bbox"])] = target.landmarks(item["bbox"])
                jobs = [(item, sources[index % len(sources)]) for index, item in enumerate(faces)]

                def serial(image=image, jobs=jobs, target=target):
                    output = image.copy()
                    for item, preset in jobs:
                        output = renderer.swap(output.copy(), item, preset, target=target)
                    return output

                def grouped(image=image, jobs=jobs, target=target):
                    output = image.copy()
                    errors = renderer.swap_many(output, jobs, target=target)
                    if any(error is not None for error in errors):
                        raise RuntimeError(str(errors))
                    return output

                # priming is shared and precomputed; do not include detection twice.
                target.prime = lambda boxes: None
                a, b = serial(), grouped()
                np.testing.assert_array_equal(a, b)
                mask = np.zeros(image.shape[:2], np.uint8)
                for item in faces:
                    cv2.fillPoly(mask, [np.rint(item["mask_polygon"]).astype(np.int32)], 1)
                np.testing.assert_array_equal(image[mask == 0], b[mask == 0])
                times = {"serial": [], "grouped": []}
                for repeat in range(args.repeats):
                    order = [("serial", serial), ("grouped", grouped)]
                    for name, function in order[:: 1 if repeat % 2 else -1]:
                        started = time.perf_counter()
                        function()
                        times[name].append((time.perf_counter() - started) * 1000)
                samples.append({"phase": phase, "ms": {k: quantiles(v) for k, v in times.items()}})
                if phase == 0:
                    cv2.imwrite(str(args.output / f"input-{count}.png"), image)
                    cv2.imwrite(str(args.output / f"serial-{count}.png"), a)
                    cv2.imwrite(str(args.output / f"grouped-{count}.png"), b)
                    cv2.imwrite(
                        str(args.output / f"compare-{count}.jpg"),
                        np.concatenate([image, a, b], axis=1),
                    )
            result = {
                "faces": count,
                "samples": samples,
                "rgb_bitwise_equal": True,
                "outside_face_polygons_unchanged": True,
                "mean_sample_p50_ms": {
                    key: float(np.mean([s["ms"][key]["p50"] for s in samples])) for key in times
                },
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        runtime.close()
    (args.output / "result.json").write_text(
        json.dumps(
            {
                "resolution": [args.width, args.height],
                "source_prepare_ms": preparation,
                "conditions": "same detected polygons, crops and preset weights; alternating timings",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/face_presets.json"))
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models/mobilefaceswap"))
    parser.add_argument("--engine", type=Path, default=Path("models/best_b1.engine"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100 or (args.width, args.height) not in (
        (640, 480),
        (1280, 720),
        (1920, 1080),
    ):
        parser.error("unsupported measurement bounds")
    main(args)
