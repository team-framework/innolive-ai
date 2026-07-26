#!/usr/bin/env python3
"""Run the B1-640-Q90-W5 ILF1 transport and latency acceptance gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_utils import (
    distribution,
    load_jpegs,
    percentile,
    sha256_file,
)
from service.protocol import (
    MAX_GRPC_RESPONSE_BYTES,
    RESULT_HEADER,
    decode_response,
    decode_result,
    encode_request,
)

WINDOW = 5
MIN_FRAMES = 120
MIN_RESULT_FPS = 30.0
MAX_SERVER_P95_MS = 33.3


@dataclass(frozen=True)
class Sample:
    sequence: int
    sent_at: float
    received_at: float
    input_jpeg_bytes: int
    mosaic_jpeg_bytes: int
    response_bytes: int
    server_total_ms: float

    @property
    def rtt_ms(self) -> float:
        return (self.received_at - self.sent_at) * 1000


async def run_stream(url: str, jpegs: list[bytes]) -> tuple[list[Sample], int]:
    next_index = 0
    pending: dict[int, tuple[float, int]] = {}
    samples: list[Sample] = []
    last_terminal = 0
    max_inflight = 0

    async with websockets.connect(
        url,
        max_size=MAX_GRPC_RESPONSE_BYTES + RESULT_HEADER.size,
        max_queue=WINDOW,
        compression=None,
    ) as websocket:
        while len(samples) < len(jpegs):
            while next_index < len(jpegs) and len(pending) < WINDOW:
                sequence = next_index + 1
                jpeg = jpegs[next_index]
                sent_at = time.perf_counter()
                await websocket.send(encode_request(sequence, jpeg))
                pending[sequence] = (sent_at, len(jpeg))
                next_index += 1
                max_inflight = max(max_inflight, len(pending))

            raw = await websocket.recv()
            received_at = time.perf_counter()
            if isinstance(raw, str):
                error = decode_response(raw)
                raise RuntimeError(
                    f"server error seq={error['seq']} code={error.get('code')}: "
                    f"{error.get('message')}"
                )
            metadata, mosaic_jpeg = decode_result(raw)
            sequence = metadata["seq"]
            if sequence not in pending:
                raise RuntimeError(f"unknown or duplicate terminal sequence: {sequence}")
            if sequence <= last_terminal:
                raise RuntimeError(f"terminal sequence regressed: {sequence}")
            last_terminal = sequence
            sent_at, jpeg_bytes = pending.pop(sequence)
            if _contains_pixels(metadata):
                raise RuntimeError("result contains forbidden JPEG/raw pixel fields")
            server_total = float(metadata.get("timing_ms", {}).get("server_total", 0))
            samples.append(
                Sample(
                    sequence,
                    sent_at,
                    received_at,
                    jpeg_bytes,
                    len(mosaic_jpeg),
                    len(raw),
                    server_total,
                )
            )

    if pending or len(samples) != len(jpegs):
        raise RuntimeError("not every accepted sequence received one terminal result")
    return samples, max_inflight


def summarize(
    samples: list[Sample],
    max_inflight: int,
    health: dict[str, Any] | None,
    input_path: Path,
) -> dict[str, Any]:
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
    gates = {
        "frames_at_least_120": len(samples) >= MIN_FRAMES,
        "result_fps_at_least_30": result_fps >= MIN_RESULT_FPS,
        "server_total_p95_at_most_33_3_ms": percentile(server, 95) <= MAX_SERVER_P95_MS,
        "max_inflight_at_most_5": max_inflight <= WINDOW,
        "terminal_sequence_complete": [sample.sequence for sample in samples]
        == list(range(1, len(samples) + 1)),
        "server_mosaic_jpeg": all(sample.mosaic_jpeg_bytes > 0 for sample in samples),
        "latency_not_continuously_growing": latency_growth_ms <= max(20.0, first_p50 * 0.20),
    }
    return {
        "profile": "B1-640-Q90-W5",
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
            "mosaic_jpeg_bytes": distribution([sample.mosaic_jpeg_bytes for sample in samples]),
            "response_bytes": distribution([sample.response_bytes for sample in samples]),
        },
        "provenance": {
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "health": health,
        },
        "samples": [
            {
                **asdict(sample),
                "rtt_ms": round(sample.rtt_ms, 2),
            }
            for sample in samples
        ],
    }


def _contains_pixels(value: Any) -> bool:
    forbidden = {"jpeg", "image", "pixels", "data", "frame"}
    if isinstance(value, dict):
        if any(str(key).lower() in forbidden for key in value):
            return True
        return any(_contains_pixels(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_pixels(item) for item in value)
    return isinstance(value, (bytes, bytearray, memoryview))


def health_url(websocket_url: str) -> str:
    parsed = urlsplit(websocket_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunsplit((scheme, parsed.netloc, "/healthz", "", ""))


def read_health(websocket_url: str) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(health_url(websocket_url), timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8001/ws")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input video not found: {input_path}")
    if args.frames < MIN_FRAMES:
        raise SystemExit(f"--frames must be at least {MIN_FRAMES}")
    jpegs = load_jpegs(input_path, args.frames)
    samples, max_inflight = asyncio.run(run_stream(args.url, jpegs))
    report = summarize(samples, max_inflight, read_health(args.url), input_path)
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
