#!/usr/bin/env python3
"""Measure gRPC face-swap capacity and RTT with a reproducible moving face grid.

Run on the serving host to isolate pipeline capacity from browser/network limits.
Each face keeps the same tile size for 1..8 faces. Reports include actual swap
counts, so failed detections or blur cannot silently count as successful capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpc_client import VideoFrame, VideoProcessorClient
from service.face_presets import PresetCatalog


def portraits(manifest):
    catalog = PresetCatalog(manifest)
    selected = []
    for gender, identity in (("female", 1), ("male", 2), ("female", 3), ("male", 4)):
        preset = next(
            p
            for p in catalog.presets.values()
            if p.attributes.gender == gender
            and p.identity == identity
            and p.attributes.age == "20s"
            and p.attributes.glasses == "off"
            and p.attributes.exp == "none"
        )
        pixels = cv2.imread(str(preset.image))
        if pixels is None:
            raise ValueError(f"missing input portrait: {preset.image}")
        selected.append(pixels)
    return selected


def grid_frames(images, faces, width=640, height=480):
    frames = []
    rows = 2 if faces <= 8 else 4
    tile_height = height // rows
    tile_width = width // 4
    size = min(tile_width, tile_height)
    for phase in range(30):
        frame = np.full((height, width, 3), 100, np.uint8)
        for index in range(faces):
            tile = cv2.resize(images[index % len(images)], (size, size))
            matrix = cv2.getRotationMatrix2D((size / 2, size / 2), 3 * math.sin(phase / 5), 0.95)
            tile = cv2.warpAffine(tile, matrix, (size, size), borderValue=(100,) * 3)
            left = index % 4 * tile_width + (tile_width - size) // 2
            top = index // 4 * tile_height + (tile_height - size) // 2
            frame[top : top + size, left : left + size] = tile
        frames.append(frame)
    return frames


def quantiles(values):
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "max": round(float(max(values)), 3),
    }


async def run_case(args, images, faces, sessions):
    pixels = grid_frames(images, faces, args.width, args.height)
    if args.transport == "raw":
        payloads = [cv2.cvtColor(p, cv2.COLOR_BGR2YUV_I420).tobytes() for p in pixels]
    else:
        payloads = []
        for frame in pixels:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("benchmark JPEG encoding failed")
            payloads.append(encoded.tobytes())
    cv2.imwrite(str(args.output / f"input-{faces}.jpg"), pixels[0])
    started = asyncio.Event()
    ready = [asyncio.Event() for _ in range(sessions)]
    clock = {}
    run_id = uuid.uuid4().hex

    async def stream(index):
        records = []
        sent = {}
        last_sequence = 0

        async def frames():
            sequence = 0
            while "end" not in clock or time.perf_counter() < clock["end"]:
                sequence += 1
                sent[sequence] = time.perf_counter()
                yield VideoFrame(
                    payloads[(sequence // 3) % len(payloads)],
                    time.perf_counter_ns(),
                    sequence,
                    anonymization_mode=args.mode,
                    width=args.width if args.transport == "raw" else 0,
                    height=args.height if args.transport == "raw" else 0,
                    pix_fmt="yuv420p" if args.transport == "raw" else "",
                )

        async with VideoProcessorClient(args.target) as client:
            async for result in client.process_video(
                frames(), session_id=f"capacity-{run_id}-{index}", window=args.window, timeout=180
            ):
                now = time.perf_counter()
                response = result.response
                sequence = int(response.frame_id)
                if sequence != last_sequence + 1 or sequence not in sent:
                    raise RuntimeError("benchmark response sequence mismatch")
                last_sequence = sequence
                rtt = (now - sent.pop(sequence)) * 1000
                if not result.processed_data:
                    raise RuntimeError("benchmark received no rendered pixels")
                if not ready[index].is_set():
                    ready[index].set()
                    await started.wait()
                now = time.perf_counter()
                if now < clock["measure"] or now >= clock["end"]:
                    continue
                counts = Counter(
                    f.anonymization.status for f in response.faces if f.anonymization.status
                )
                records.append(
                    {
                        "at": now - clock["measure"],
                        "faces": dict(counts),
                        "detected_faces": len(response.faces),
                        "server": response.timing.server_total_ms,
                        "queue": response.timing.queue_ms,
                        "detect": response.timing.inference_ms,
                        "render": response.timing.blur_encode_ms,
                        "rtt": rtt,
                        "fallbacks": dict(
                            Counter(
                                f.anonymization.fallback_reason
                                for f in response.faces
                                if f.anonymization.fallback_reason
                            )
                        ),
                    }
                )
                if len(records) == 20:
                    if args.transport == "raw":
                        output = np.frombuffer(result.processed_data, np.uint8).reshape(
                            args.height * 3 // 2, args.width
                        )
                        output = cv2.cvtColor(output, cv2.COLOR_YUV2BGR_I420)
                    else:
                        output = cv2.imdecode(
                            np.frombuffer(result.processed_data, np.uint8), cv2.IMREAD_COLOR
                        )
                    if output is None or output.shape != (args.height, args.width, 3):
                        raise RuntimeError("benchmark output shape mismatch")
                    cv2.imwrite(
                        str(args.output / f"output-{faces}x{sessions}-{index}.jpg"),
                        output,
                    )
        if not records:
            raise RuntimeError("no measured results")
        return {
            "fps": round(len(records) / args.seconds, 3),
            "frames": len(records),
            "all_faces_swapped_fraction": round(
                sum(r["faces"].get("swapped", 0) == faces for r in records) / len(records), 4
            ),
            "all_faces_detected_fraction": round(
                sum(r["detected_faces"] == faces for r in records) / len(records), 4
            ),
            "face_counts": dict(Counter(str(r["faces"]) for r in records)),
            "fallback_reasons": dict(sum((Counter(r["fallbacks"]) for r in records), Counter())),
            "max_inflight": client.max_inflight_observed,
            "timings_ms": {
                key: quantiles([r[key] for r in records])
                for key in ("server", "queue", "detect", "render", "rtt")
            },
        }

    jobs = [asyncio.create_task(stream(index)) for index in range(sessions)]
    readiness = asyncio.ensure_future(asyncio.gather(*(r.wait() for r in ready)))
    try:
        done, _ = await asyncio.wait(
            [readiness, *jobs], timeout=90, return_when=asyncio.FIRST_COMPLETED
        )
        if readiness not in done:
            for job in done:
                job.result()
            raise RuntimeError("streams did not become ready within 90 seconds")
        clock.update(
            measure=time.perf_counter() + args.warmup,
            end=time.perf_counter() + args.warmup + args.seconds,
        )
        started.set()
        results = await asyncio.gather(*jobs)
        return {
            "input_faces": faces,
            "sessions": sessions,
            "seconds": args.seconds,
            "measurement_monotonic": [clock["measure"], clock["end"]],
            "resolution": [args.width, args.height],
            "transport": args.transport,
            "window": args.window,
            "scope": "pre-encoded input to gRPC response; client display/encoding excluded",
            "aggregate_fps": round(sum(r["fps"] for r in results), 3),
            "streams": results,
        }
    finally:
        readiness.cancel()
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        await asyncio.gather(readiness, return_exceptions=True)


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    images = portraits(args.manifest)
    results = []
    for case in args.cases.split(","):
        faces, sessions = map(int, case.split("x"))
        if faces not in (1, 2, 4, 8, 16) or not 1 <= sessions <= 16:
            raise ValueError("faces must be 1/2/4/8/16 and sessions 1..16")
        result = await run_case(args, images, faces, sessions)
        results.append(result)
        (args.output / "result.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:50062")
    parser.add_argument("--manifest", type=Path, default=Path("config/face_presets.json"))
    parser.add_argument("--cases", default="1x1,2x1,4x1,8x1,1x2,2x2,1x4,2x4,1x8")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--transport", choices=("raw", "jpeg"), default="raw")
    parser.add_argument("--window", type=int, choices=(1, 2, 5), default=1)
    parser.add_argument("--mode", choices=("face_swap", "blur"), default="face_swap")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120 or not 0 <= args.warmup <= 30:
        parser.error("invalid measurement duration")
    if (args.width, args.height) not in ((640, 480), (1280, 720), (1920, 1080)):
        parser.error("resolution must be 640x480, 1280x720 or 1920x1080")
    asyncio.run(main(args))
