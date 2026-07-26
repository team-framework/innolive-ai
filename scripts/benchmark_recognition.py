#!/usr/bin/env python3
"""Measure the Python metadata cost added by session and track recognition."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.recognition import RecognitionConfig, SessionRegistry, StreamRecognition


@dataclass(frozen=True, slots=True)
class Result:
    scenario: str
    frames: int
    streams: int
    elapsed_ms: float
    frame_latency_ms: float
    fps: float
    adaface_calls: int
    queue_overflow: int
    peak_memory_kib: float


class ImmediateRuntime:
    ready = True

    def __init__(self) -> None:
        self.calls = 0

    def submit(self, _image: np.ndarray, *, owner: str):
        del owner
        self.calls += 1
        future = asyncio.get_running_loop().create_future()
        future.set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        return future


def face() -> dict[str, object]:
    return {
        "track_id": 1,
        "bbox": [16.0, 16.0, 80.0, 80.0],
        "held": False,
    }


async def measured(scenario: str, frames: int, streams: int, operation) -> Result:
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    calls, overflow = await operation()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_frames = frames * streams
    return Result(
        scenario=scenario,
        frames=total_frames,
        streams=streams,
        elapsed_ms=round(elapsed * 1_000, 3),
        frame_latency_ms=round(elapsed * 1_000 / total_frames, 6),
        fps=round(total_frames / elapsed, 2),
        adaface_calls=calls,
        queue_overflow=overflow,
        peak_memory_kib=round(peak / 1_024, 2),
    )


async def benchmark(frames: int, stream_count: int) -> list[Result]:
    image = np.zeros((96, 96, 3), dtype=np.uint8)

    async def baseline():
        item = face()
        for _ in range(frames):
            item["whitelisted"] = False
            await asyncio.sleep(0)
        return 0, 0

    async def recognition_case(*, streams: int, with_whitelist: bool):
        session = SessionRegistry().get_or_create("benchmark-session")
        if with_whitelist:
            session.append(np.asarray([1.0, 0.0], dtype=np.float32))
        runtime = ImmediateRuntime()
        config = RecognitionConfig(revalidate_frames=frames + 1)
        recognizers = [
            StreamRecognition(runtime, config, owner="benchmark-session") for _ in range(streams)
        ]
        objects = [face() for _ in range(streams)]

        async def run_stream(recognizer, item):
            overflow = 0
            for frame_sequence in range(1, frames + 1):
                snapshot = session.snapshot()
                metrics = recognizer.process(image, [item], snapshot, frame_sequence)
                overflow += metrics["adaface_queue_overflow"]
                await asyncio.sleep(0)
            return overflow

        overflow = sum(
            await asyncio.gather(
                *(
                    run_stream(recognizer, item)
                    for recognizer, item in zip(recognizers, objects, strict=True)
                )
            )
        )
        for recognizer in recognizers:
            recognizer.close()
        return runtime.calls, overflow

    return [
        await measured("all_blur_baseline", frames, 1, baseline),
        await measured(
            "empty_whitelist",
            frames,
            1,
            lambda: recognition_case(streams=1, with_whitelist=False),
        ),
        await measured(
            "one_whitelist_same_track",
            frames,
            1,
            lambda: recognition_case(streams=1, with_whitelist=True),
        ),
        await measured(
            "concurrent_stream_state",
            frames,
            stream_count,
            lambda: recognition_case(streams=stream_count, with_whitelist=True),
        ),
    ]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=positive_int, default=10_000)
    parser.add_argument("--streams", type=positive_int, default=4)
    args = parser.parse_args()
    results = asyncio.run(benchmark(args.frames, args.streams))
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
