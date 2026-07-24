#!/usr/bin/env python3
"""Measure per-frame gRPC bidirectional-stream communication latency."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import grpc


ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "__generated__"
sys.path.insert(0, str(GENERATED_DIR))

import ai_processor_pb2_grpc  # noqa: E402
from ai_processor_pb2 import VideoChunk  # noqa: E402
from run_benchmarks import (  # noqa: E402
    file_sha256,
    load_font,
    package_version,
    split_target,
    terminate_process,
    total_memory_bytes,
    wait_for_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results")
    return parser.parse_args()


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentage / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_frames(path: Path, fps: float, max_frames: int, quality: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_fps = min(fps, source_fps) if source_fps > 0 else fps
    if sample_fps <= 0 or max_frames <= 0 or not 1 <= quality <= 100:
        capture.release()
        raise ValueError("fps/max-frames must be positive and jpeg-quality must be 1..100")

    frames: list[bytes] = []
    frame_index = 0
    next_sample_time = 0.0
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_time = frame_index / source_fps if source_fps > 0 else len(frames) / sample_fps
        frame_index += 1
        if frame_time + 1e-9 < next_sample_time:
            continue
        encoded_ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not encoded_ok:
            capture.release()
            raise RuntimeError(f"Failed to encode frame {frame_index - 1}")
        frames.append(encoded.tobytes())
        next_sample_time += 1 / sample_fps
    capture.release()
    if not frames:
        raise RuntimeError("No video frames were extracted")
    return frames, {
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path.resolve()),
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "source_frame_count": source_count,
        "sample_fps": sample_fps,
        "sampled_frame_count": len(frames),
        "jpeg_quality": quality,
        "frame_bytes_average": sum(map(len, frames)) / len(frames),
        "frame_bytes_min": min(map(len, frames)),
        "frame_bytes_max": max(map(len, frames)),
    }


async def run_stream(stub, stream_id: int, frames: list[bytes], fps: float, timeout: float):
    interval_ns = int(1_000_000_000 / fps)
    scheduled_start = time.perf_counter_ns() + 300_000_000
    sent: dict[int, tuple[int, int, int, int]] = {}
    schedule_lags_ms: list[float] = []

    async def requests():
        for frame_index, frame in enumerate(frames):
            scheduled_ns = scheduled_start + frame_index * interval_ns
            delay = (scheduled_ns - time.perf_counter_ns()) / 1_000_000_000
            if delay > 0:
                await asyncio.sleep(delay)
            sent_ns = time.perf_counter_ns()
            timestamp = sent_ns
            sent[timestamp] = (frame_index, sent_ns, scheduled_ns, len(frame))
            schedule_lags_ms.append((sent_ns - scheduled_ns) / 1_000_000)
            yield VideoChunk(data=frame, timestamp=timestamp)

    samples: list[dict[str, Any]] = []
    status_errors = 0
    call = stub.ProcessVideo(requests(), timeout=timeout)
    response_index = 0
    async for response in call:
        received_ns = time.perf_counter_ns()
        details = sent.get(response.timestamp)
        if details is None:
            status_errors += 1
            continue
        frame_index, sent_ns, scheduled_ns, payload_bytes = details
        samples.append(
            {
                "stream": stream_id,
                "frame": frame_index,
                "payload_bytes": payload_bytes,
                "latency_ms": (received_ns - sent_ns) / 1_000_000,
                "send_schedule_lag_ms": (sent_ns - scheduled_ns) / 1_000_000,
                "sent_ns": sent_ns,
                "received_ns": received_ns,
            }
        )
        response_index += 1
    sent_times = sorted(value[1] for value in sent.values())
    effective_fps = (
        (len(sent_times) - 1) * 1_000_000_000 / (sent_times[-1] - sent_times[0])
        if len(sent_times) > 1
        else 0.0
    )
    return {
        "samples": samples,
        "sent": len(sent),
        "received": response_index,
        "status_errors": status_errors,
        "effective_send_fps": effective_fps,
        "schedule_lags_ms": schedule_lags_ms,
    }


async def measure(target: str, frames: list[bytes], fps: float, concurrency: int, timeout: float):
    channel = grpc.aio.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ],
    )
    await asyncio.wait_for(channel.channel_ready(), timeout=5)
    stub = ai_processor_pb2_grpc.AiProcessorStub(channel)
    try:
        return await asyncio.gather(
            *(run_stream(stub, index, frames, fps, timeout) for index in range(concurrency))
        )
    finally:
        await channel.close()


def build_result(args: argparse.Namespace, video: dict[str, Any], streams):
    samples = [sample for stream in streams for sample in stream["samples"]]
    latencies = [sample["latency_ms"] for sample in samples]
    schedule_lags = [value for stream in streams for value in stream["schedule_lags_ms"]]
    first_sent = min(sample["sent_ns"] for sample in samples)
    last_received = max(sample["received_ns"] for sample in samples)
    duration_seconds = (last_received - first_sent) / 1_000_000_000
    total_payload = sum(sample["payload_bytes"] for sample in samples)
    deadline_ms = 1000 / args.fps
    received = len(samples)
    sent = sum(stream["sent"] for stream in streams)
    errors = sent - received + sum(stream["status_errors"] for stream in streams)
    metrics = {
        "sent_frames": sent,
        "received_frames": received,
        "errors": errors,
        "wall_duration_seconds": duration_seconds,
        "aggregate_frames_per_second": received / duration_seconds,
        "effective_send_fps_per_stream": statistics.mean(
            stream["effective_send_fps"] for stream in streams
        ),
        "app_payload_mib_per_second": total_payload / duration_seconds / 1024 / 1024,
        "latency_ms": {
            "average": statistics.mean(latencies),
            "minimum": min(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "maximum": max(latencies),
            "standard_deviation": statistics.pstdev(latencies),
        },
        "send_schedule_lag_ms": {
            "average": statistics.mean(schedule_lags),
            "p95": percentile(schedule_lags, 95),
            "maximum": max(schedule_lags),
        },
        "frame_deadline_ms": deadline_ms,
        "responses_within_frame_deadline_ratio": (
            sum(value <= deadline_ms for value in latencies) / len(latencies)
        ),
    }
    return {
        "schema_version": 1,
        "test": "grpc_bidi_per_frame_communication_latency",
        "recorded_at": dt.datetime.now().astimezone().isoformat(),
        "target": args.target,
        "server_processing_mode": "passthrough",
        "configuration": {
            "concurrency": args.concurrency,
            "target_fps_per_stream": args.fps,
            "timeout_seconds": args.timeout,
        },
        "input": video,
        "environment": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "total_memory_bytes": total_memory_bytes(),
            "python": sys.version,
            "grpcio": package_version("grpcio"),
            "opencv_python": package_version("opencv-python"),
        },
        "metrics": metrics,
        "samples": samples,
    }


def write_summary(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    latency = metrics["latency_ms"]
    input_data = result["input"]
    lines = [
        "# 1080p 30 FPS gRPC communication latency",
        "",
        f"- Input: {input_data['width']}x{input_data['height']} @ {input_data['sample_fps']:.0f} FPS, "
        f"{input_data['sampled_frame_count']} frames",
        f"- Streams: {result['configuration']['concurrency']} concurrent, `passthrough` server",
        f"- Received: {metrics['received_frames']}/{metrics['sent_frames']} frames, errors {metrics['errors']}",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Effective send FPS / stream | {metrics['effective_send_fps_per_stream']:.2f} |",
        f"| Aggregate receive FPS | {metrics['aggregate_frames_per_second']:.2f} |",
        f"| Application payload | {metrics['app_payload_mib_per_second']:.2f} MiB/s |",
        f"| RTT average | {latency['average']:.2f} ms |",
        f"| RTT p50 | {latency['p50']:.2f} ms |",
        f"| RTT p95 | {latency['p95']:.2f} ms |",
        f"| RTT p99 | {latency['p99']:.2f} ms |",
        f"| RTT maximum | {latency['maximum']:.2f} ms |",
        f"| Responses within {metrics['frame_deadline_ms']:.2f} ms | "
        f"{metrics['responses_within_frame_deadline_ratio']:.1%} |",
    ]
    lines.extend(
        [
            "",
            "> RTT is measured per frame from immediately before the client sends the protobuf "
            "message until its matching response is received. Image processing is excluded.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_chart(path: Path, result: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    samples = result["samples"]
    latencies = [sample["latency_ms"] for sample in samples]
    metrics = result["metrics"]
    latency = metrics["latency_ms"]
    width, height = 1400, 760
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    foreground, muted, grid = "#172033", "#64748b", "#d9e0ea"
    blue, orange, red = "#2878d0", "#e69f00", "#d95f59"
    title_font = load_font(34, bold=True)
    heading_font = load_font(23, bold=True)
    label_font = load_font(18)
    value_font = load_font(18, bold=True)
    draw.text((60, 40), "1080p 30 FPS · per-frame gRPC RTT", fill=foreground, font=title_font)
    draw.text(
        (60, 88),
        f"{result['configuration']['concurrency']} streams · {len(samples)} responses · passthrough",
        fill=muted,
        font=label_font,
    )

    plot = (80, 170, 920, 650)
    max_y = max(max(latencies), metrics["frame_deadline_ms"]) * 1.1 or 1
    for tick in range(6):
        y = plot[3] - (plot[3] - plot[1]) * tick / 5
        value = max_y * tick / 5
        draw.line((plot[0], y, plot[2], y), fill=grid, width=1)
        draw.text((20, y - 10), f"{value:.1f}", fill=muted, font=label_font)
    points = []
    for index, value in enumerate(latencies):
        x = plot[0] + (plot[2] - plot[0]) * index / max(len(latencies) - 1, 1)
        y = plot[3] - (plot[3] - plot[1]) * value / max_y
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=blue, width=2)
    deadline_y = plot[3] - (plot[3] - plot[1]) * metrics["frame_deadline_ms"] / max_y
    draw.line((plot[0], deadline_y, plot[2], deadline_y), fill=red, width=3)
    draw.text((plot[0] + 10, deadline_y - 28), "33.33 ms frame deadline", fill=red, font=value_font)
    draw.text((plot[0], 670), "Response sequence", fill=muted, font=label_font)
    draw.text((20, 135), "RTT ms", fill=muted, font=label_font)

    right_x = 1010
    draw.text((right_x, 170), "Latency percentiles", fill=foreground, font=heading_font)
    values = [("avg", latency["average"], blue), ("p50", latency["p50"], blue), ("p95", latency["p95"], orange), ("p99", latency["p99"], red)]
    max_bar = max(value for _, value, _ in values) or 1
    for index, (label, value, color) in enumerate(values):
        y = 230 + index * 90
        draw.text((right_x, y), label, fill=foreground, font=value_font)
        bar_width = int(280 * value / max_bar)
        draw.rounded_rectangle((right_x, y + 32, right_x + max(bar_width, 4), y + 60), 7, fill=color)
        draw.text((right_x + bar_width + 10, y + 34), f"{value:.2f} ms", fill=foreground, font=value_font)
    draw.text(
        (right_x, 620),
        f"{metrics['responses_within_frame_deadline_ratio']:.1%} within deadline",
        fill=foreground,
        font=heading_font,
    )
    image.save(path, format="PNG", optimize=True)


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    frames, video = load_frames(
        args.video.resolve(), args.fps, args.max_frames, args.jpeg_quality
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    host, port = split_target(args.target)
    server_process = None
    try:
        if args.start_server:
            server_process = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "ai_processor_server.py"),
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--max-workers",
                    str(args.max_workers),
                    "--processing-mode",
                    "passthrough",
                ],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wait_for_target(args.target, 15)
        else:
            wait_for_target(args.target, 3)
        streams = asyncio.run(
            measure(args.target, frames, args.fps, args.concurrency, args.timeout)
        )
    finally:
        if server_process:
            terminate_process(server_process)
    result = build_result(args, video, streams)

    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_summary(output_dir / "summary.md", result)
    write_chart(output_dir / "summary.png", result)
    print(f"Reports written to {output_dir}")
    return 0 if result["metrics"]["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
