#!/usr/bin/env python3
"""Run the B1-640-Q90-W5 gRPC ProcessVideo acceptance gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import grpc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grpc_client import VideoFrame, VideoProcessorClient
from scripts.benchmark_utils import distribution, load_jpegs, percentile, sha256_file

WINDOW = 5
MIN_FRAMES = 120
MIN_RESULT_FPS = 30.0
MAX_SERVER_P95_MS = 33.3


@dataclass(frozen=True, slots=True)
class Sample:
    sequence: int
    sent_at: float
    received_at: float
    input_jpeg_bytes: int
    processed_data_bytes: int
    response_bytes: int
    server_total_ms: float

    @property
    def rtt_ms(self) -> float:
        return (self.received_at - self.sent_at) * 1000


async def run_stream(
    target: str,
    jpegs: list[bytes],
    timeout: float,
    session_id: str,
) -> tuple[list[Sample], int]:
    sent_at: dict[int, float] = {}

    async def frames():
        for sequence, jpeg in enumerate(jpegs, start=1):
            timestamp = time.perf_counter_ns()
            sent_at[sequence] = timestamp / 1_000_000_000
            yield VideoFrame(
                data=jpeg,
                timestamp=timestamp,
                frame_id=sequence,
            )

    samples: list[Sample] = []
    async with VideoProcessorClient(
        target,
        connect_timeout=min(timeout, 10.0),
    ) as client:
        async for result in client.process_video(
            frames(),
            session_id=session_id,
            window=WINDOW,
            timeout=timeout,
        ):
            received_at = time.perf_counter()
            response = result.response
            sequence = int(response.frame_id)
            expected = len(samples) + 1
            if sequence != expected:
                raise RuntimeError(
                    f"terminal sequence mismatch: expected {expected}, got {sequence}"
                )
            if result.source_jpeg != jpegs[sequence - 1]:
                raise RuntimeError(f"source JPEG mismatch for frame {sequence}")
            if not response.data:
                raise RuntimeError("server did not return processed data")
            samples.append(
                Sample(
                    sequence=sequence,
                    sent_at=sent_at[sequence],
                    received_at=received_at,
                    input_jpeg_bytes=len(result.source_jpeg),
                    processed_data_bytes=len(response.data),
                    response_bytes=int(response.ByteSize()),
                    server_total_ms=_server_total_ms(response),
                )
            )
        max_inflight = client.max_inflight_observed

    if len(samples) != len(jpegs):
        raise RuntimeError(f"only {len(samples)}/{len(jpegs)} frames received a terminal result")
    return samples, max_inflight


def summarize(
    samples: list[Sample],
    max_inflight: int,
    input_path: Path,
    target: str,
) -> dict[str, object]:
    receive_times = [sample.received_at for sample in samples]
    duration = receive_times[-1] - receive_times[0] if len(samples) > 1 else 0
    result_fps = (len(samples) - 1) / duration if duration > 0 else 0
    rtt = [sample.rtt_ms for sample in samples]
    server = [sample.server_total_ms for sample in samples]
    first_quarter = rtt[: max(1, len(rtt) // 4)]
    last_quarter = rtt[-max(1, len(rtt) // 4) :]
    first_p50 = percentile(first_quarter, 50)
    last_p50 = percentile(last_quarter, 50)
    latency_growth_ms = last_p50 - first_p50
    sequences = [sample.sequence for sample in samples]
    gates = {
        "frames_at_least_120": len(samples) >= MIN_FRAMES,
        "result_fps_at_least_30": result_fps >= MIN_RESULT_FPS,
        "server_total_present": bool(server) and min(server) > 0,
        "server_total_p95_at_most_33_3_ms": percentile(server, 95) <= MAX_SERVER_P95_MS,
        "max_inflight_at_most_5": max_inflight <= WINDOW,
        "terminal_sequence_complete": sequences == list(range(1, len(samples) + 1)),
        "server_processed_data": all(sample.processed_data_bytes > 0 for sample in samples),
        "latency_not_continuously_growing": latency_growth_ms <= max(20.0, first_p50 * 0.20),
    }
    return {
        "profile": "B1-640-Q90-W5",
        "transport": "grpc.aio bidi ProcessVideo",
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {
            "frames": len(samples),
            "result_fps": round(result_fps, 2),
            "max_inflight": max_inflight,
            "rtt_ms": distribution(rtt),
            "server_total_ms": distribution(server),
            "latency_growth_ms": round(latency_growth_ms, 2),
            "input_jpeg_bytes": distribution([sample.input_jpeg_bytes for sample in samples]),
            "processed_data_bytes": distribution(
                [sample.processed_data_bytes for sample in samples]
            ),
            "response_bytes": distribution([sample.response_bytes for sample in samples]),
        },
        "provenance": {
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "target": target,
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "grpcio": grpc.__version__,
        },
        "samples": [
            {
                **asdict(sample),
                "rtt_ms": round(sample.rtt_ms, 2),
            }
            for sample in samples
        ],
    }


def _server_total_ms(response: object) -> float:
    timing = getattr(response, "timing", None)
    if timing is not None:
        for field in ("server_total_ms", "server_total"):
            if hasattr(timing, field):
                value = float(getattr(timing, field))
                if value > 0:
                    return value
    value = float(getattr(response, "processing_ms", 0.0))
    return value if value > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input video not found: {input_path}")
    if args.frames < MIN_FRAMES:
        raise SystemExit(f"--frames must be at least {MIN_FRAMES}")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    jpegs = load_jpegs(input_path, args.frames)
    samples, max_inflight = asyncio.run(
        run_stream(args.target, jpegs, args.timeout, args.session_id)
    )
    report = summarize(
        samples,
        max_inflight,
        input_path,
        args.target,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
