#!/usr/bin/env python3
"""Throughput benchmark for the batched YOLO face-swap lab (issue #21).

Drives SwapLab in-process with synthetic multi-face frames and concurrent
sessions, then reports FPS, swaps/sec, p50/p95 latency, stage breakdown and
GPU utilization.  Runs on the 3090 host; every metric is also computable from
the per-frame metadata so base/optimized builds can share this harness.

Example:
    python -m experiments.trt_swap_client.bench_lab \
        --detector models/best_swap_b4.engine \
        --swapper models/face_swap/inswapper_128.onnx \
        --swapper-engine models/face_swap/inswapper_128_trt11_fp32.engine \
        --source ~/Documents/input.png \
        --faces 1 2 4 --sessions 1 2 4 --frames-per-session 60 \
        --output reports/issue-21-bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.trt_swap_client.app import Settings, SwapLab

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720


def build_face_frame(source: np.ndarray, faces: int) -> np.ndarray:
    """Tile the source portrait into a canvas with ``faces`` detectable faces."""

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (32, 32, 32)
    slots = {1: (1, 1), 2: (2, 1), 4: (2, 2)}[faces]
    cols, rows = slots
    cell_w, cell_h = CANVAS_WIDTH // cols, CANVAS_HEIGHT // rows
    scale = min(cell_w / source.shape[1], cell_h / source.shape[0], 1.0)
    resized = cv2.resize(
        source,
        (max(32, int(source.shape[1] * scale)), max(32, int(source.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    for index in range(faces):
        col, row = index % cols, index // cols
        x = col * cell_w + (cell_w - resized.shape[1]) // 2
        y = row * cell_h + (cell_h - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


@dataclass
class GpuSampler:
    stop: threading.Event = field(default_factory=threading.Event)
    samples: list[float] = field(default_factory=list)

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.25)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


async def drive_session(
    lab: SwapLab, stream: Any, frame: np.ndarray, count: int
) -> tuple[list[float], list[dict[str, Any]], int]:
    latencies: list[float] = []
    metas: list[dict[str, Any]] = []
    drops = 0
    for _ in range(count):
        started = time.perf_counter()
        try:
            _, meta = await lab.submit(frame.copy(), stream)
        except RuntimeError:
            drops += 1
            continue
        latencies.append((time.perf_counter() - started) * 1_000)
        metas.append(meta)
    return latencies, metas, drops


async def run_config(
    lab: SwapLab, frame: np.ndarray, sessions: int, frames_per_session: int
) -> dict[str, Any]:
    streams = [lab.create_stream(f"bench-{index}") for index in range(sessions)]
    # Closed-loop per session: next frame goes out as soon as the previous
    # result returns, so the measured rate is the sustained service rate.
    sampler = GpuSampler()
    sampler.stop.clear()
    thread = threading.Thread(target=sampler.run, daemon=True)
    wall_started = time.perf_counter()
    thread.start()
    try:
        results = await asyncio.gather(
            *(drive_session(lab, stream, frame, frames_per_session) for stream in streams)
        )
    finally:
        wall = time.perf_counter() - wall_started
        sampler.stop.set()
        thread.join(timeout=5)
        for stream in streams:
            stream.recognition.close()
            stream.tracker.reset()
    latencies = [value for result in results for value in result[0]]
    metas = [meta for result in results for meta in result[1]]
    drops = sum(result[2] for result in results)
    swaps = sum(int(meta.get("swap_faces", 0)) for meta in metas)

    def stage(name: str) -> list[float]:
        return [float(m.get(name, 0.0)) for m in metas if name in m]

    detector = [float(m.get("detector_batch_ms", 0.0)) for m in metas]
    return {
        "frames_completed": len(latencies),
        "frames_dropped": drops,
        "wall_s": round(wall, 3),
        "fps": round(len(latencies) / wall, 2) if wall > 0 else 0.0,
        "face_swaps": swaps,
        "swaps_per_sec": round(swaps / wall, 2) if wall > 0 else 0.0,
        "faces_per_frame_mean": round(swaps / len(metas), 2) if metas else 0.0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "detector_batch_ms_mean": round(statistics.fmean(detector), 2) if detector else None,
        "swap_ms_mean": _mean(stage("swap_ms")),
        "swap_prepare_ms_mean": _mean(stage("swap_prepare_ms")),
        "swap_forward_ms_mean": _mean(stage("swap_forward_ms")),
        "swap_paste_ms_mean": _mean(stage("swap_paste_ms")),
        "swap_alignment_ms_mean": _mean(stage("swap_alignment_ms")),
        "swap_generator_ms_mean": _mean(stage("swap_generator_ms")),
        "yolo_batch_mean": _mean(stage("yolo_batch")),
        "swap_batch_size_mean": _mean(stage("swap_batch_size")),
        "gpu_util_mean": round(statistics.fmean(sampler.samples), 1) if sampler.samples else None,
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    source = cv2.imread(str(args.source.expanduser()), cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit(f"could not read source: {args.source}")
    settings = Settings(
        detector=args.detector.expanduser().resolve(),
        swapper=args.swapper.expanduser().resolve(),
        swapper_engine=args.swapper_engine.expanduser().resolve(),
        source=args.source.expanduser().resolve(),
        device=args.device,
        max_batch=args.max_batch,
        batch_wait_ms=args.batch_wait_ms,
        max_queue=args.max_queue,
        swap_ort_mem_gib=args.swap_ort_mem_gib,
        swap_min_mask_area_px=args.swap_min_mask_area_px,
        swapper_backend=args.swapper_backend,
        swapper_trt_cache=args.swapper_trt_cache.expanduser().resolve(),
        swapper_trt_workspace_gib=args.swapper_trt_workspace_gib,
        target_aligner=args.target_aligner,
        target_yunet=args.target_yunet.expanduser().resolve(),
        input_video=None,
        hls_dir=args.hls_dir.expanduser().resolve(),
        swap_debug_dir=None,
        swap_debug_frames=0,
    )
    lab = SwapLab(settings)
    await lab.start()
    try:
        report: dict[str, Any] = {"configs": {}}
        for faces in args.faces:
            frame = build_face_frame(source, faces)
            # One warm-up config per face count so engine/tactic caches settle.
            warmup_stream = lab.create_stream("bench-warmup")
            try:
                await drive_session(lab, warmup_stream, frame, args.warmup)
            finally:
                warmup_stream.recognition.close()
                warmup_stream.tracker.reset()
            for sessions in args.sessions:
                key = f"faces{faces}xSessions{sessions}"
                print(f"[{key}] running...", flush=True)
                report["configs"][key] = await run_config(
                    lab, frame, sessions, args.frames_per_session
                )
                print(f"[{key}] {json.dumps(report['configs'][key])}", flush=True)
        return report
    finally:
        await lab.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--swapper", type=Path, required=True)
    parser.add_argument("--swapper-engine", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=3.0)
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument("--swap-ort-mem-gib", type=float, default=2.0)
    parser.add_argument("--swap-min-mask-area-px", type=float, default=5_625)
    parser.add_argument("--swapper-backend", choices=("tensorrt", "cuda"), default="tensorrt")
    parser.add_argument(
        "--swapper-trt-cache", type=Path, default=Path("face_swap_lab_output/trt_swapper_cache")
    )
    parser.add_argument("--swapper-trt-workspace-gib", type=float, default=1.0)
    parser.add_argument(
        "--target-aligner", choices=("yunet_roi", "insightface"), default="yunet_roi"
    )
    parser.add_argument(
        "--target-yunet", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx")
    )
    parser.add_argument("--hls-dir", type=Path, default=Path("face_swap_lab_output/trt_hls"))
    parser.add_argument("--faces", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--frames-per-session", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    report = asyncio.run(amain(args))
    report["elapsed_s"] = round(time.time() - started, 1)
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
