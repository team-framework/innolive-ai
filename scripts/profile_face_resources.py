#!/usr/bin/env python3
"""Sample GPU-wide NVML and per-process CPU/RAM during actual gRPC measurements.

Requires psutil and nvidia-ml-py on the GPU server. GPU utilization means sampled
busy time, not SM occupancy. Other resident processes are reported separately.
CPU percentages use 100% per logical CPU. Warmup/input preparation are excluded
using the benchmark's monotonic measurement window. No clocks or limits change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_face_capacity import portraits, run_case


def summary(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (float, int))})
    return {
        key: {
            "mean": round(float(np.mean(values)), 3),
            "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "max": round(float(max(values)), 3),
        }
        for key in keys
        if key != "at" and (values := [r[key] for r in rows if key in r])
    }


async def main(args):
    import psutil
    import pynvml as nv

    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(args.gpu)
    processes = {}
    for pid in {args.server_pid, os.getpid(), *args.extra_pid} | {
        p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)
    }:
        process = psutil.Process(pid)
        process.cpu_percent()
        processes[pid] = process
    args.output.mkdir(parents=True, exist_ok=True)
    images = portraits(args.manifest) if not args.observe else None
    report = {
        "gpu": nv.nvmlDeviceGetName(handle),
        "driver": nv.nvmlSystemGetDriverVersion(),
        "cpu_logical_count": psutil.cpu_count(),
        "server_pid": args.server_pid,
        "monitor_pid": os.getpid(),
        "processes": {pid: p.name() for pid, p in processes.items()},
        "interval_seconds": args.interval,
        "cases": [],
    }

    def sample():
        util = nv.nvmlDeviceGetUtilizationRates(handle)
        memory = nv.nvmlDeviceGetMemoryInfo(handle)
        reasons = nv.nvmlDeviceGetCurrentClocksEventReasons(handle)
        row = {
            "at": time.perf_counter(),
            "gpu_busy_pct": util.gpu,
            "dram_busy_pct": util.memory,
            "gpu_memory_mib": memory.used / 2**20,
            "gpu_memory_total_mib": memory.total / 2**20,
            "power_w": nv.nvmlDeviceGetPowerUsage(handle) / 1000,
            "power_limit_w": nv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
            "temperature_c": nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
            "memory_clock_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
            "power_capped": int(bool(reasons & nv.nvmlClocksEventReasonSwPowerCap)),
            "thermal_slowdown": int(
                bool(
                    reasons
                    & (
                        nv.nvmlClocksEventReasonSwThermalSlowdown
                        | nv.nvmlClocksEventReasonHwThermalSlowdown
                    )
                )
            ),
            "system_cpu_pct": psutil.cpu_percent(),
            "system_available_ram_mib": psutil.virtual_memory().available / 2**20,
        }
        for name, getter in (
            ("encoder_busy_pct", nv.nvmlDeviceGetEncoderUtilization),
            ("decoder_busy_pct", nv.nvmlDeviceGetDecoderUtilization),
        ):
            with suppress(nv.NVMLError):
                row[name] = getter(handle)[0]
        for name, counter in (
            ("pcie_tx_kib_s", nv.NVML_PCIE_UTIL_TX_BYTES),
            ("pcie_rx_kib_s", nv.NVML_PCIE_UTIL_RX_BYTES),
        ):
            with suppress(nv.NVMLError):
                row[name] = nv.nvmlDeviceGetPcieThroughput(handle, counter)
        for pid, process in processes.items():
            with suppress(psutil.NoSuchProcess):
                row[f"pid_{pid}_cpu_pct"] = process.cpu_percent()
                row[f"pid_{pid}_rss_mib"] = process.memory_info().rss / 2**20
        for process in nv.nvmlDeviceGetComputeRunningProcesses(handle):
            if process.usedGpuMemory is not None and process.usedGpuMemory < 2**50:
                row[f"pid_{process.pid}_vram_mib"] = process.usedGpuMemory / 2**20
        return row

    async def monitor(samples):
        while True:
            # NVML calls may block for tens of milliseconds. Keep them off the
            # gRPC client's event loop so sampling cannot stall frame submission.
            samples.append(await asyncio.to_thread(sample))
            await asyncio.sleep(args.interval)

    try:
        for case in ["observe"] if args.observe else args.cases.split(","):
            samples = []
            sampler = asyncio.create_task(monitor(samples))
            try:
                if args.observe:
                    await asyncio.sleep(args.warmup)
                    start = time.perf_counter()
                    await asyncio.sleep(args.seconds)
                    result = {"measurement_monotonic": [start, time.perf_counter()]}
                else:
                    faces, sessions = map(int, case.split("x"))
                    if faces not in (1, 2, 4, 8, 16) or not 1 <= sessions <= 16:
                        raise ValueError("invalid face/session count")
                    result = await run_case(args, images, faces, sessions)
            finally:
                sampler.cancel()
                outcome = await asyncio.gather(sampler, return_exceptions=True)
                if isinstance(outcome[0], Exception):
                    raise RuntimeError("resource sampler failed") from outcome[0]
            start, end = result["measurement_monotonic"]
            measured = [r for r in samples if start <= r["at"] < end]
            if not measured:
                raise RuntimeError("no resource samples during measurement")
            case_report = {"benchmark": result, "resources": summary(measured), "samples": measured}
            report["cases"].append(case_report)
            (args.output / "resources.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "case": case,
                        "fps": result.get("aggregate_fps"),
                        "resources": case_report["resources"],
                    }
                ),
                flush=True,
            )
    finally:
        nv.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:50071")
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--extra-pid", type=int, action="append", default=[])
    parser.add_argument(
        "--observe", action="store_true", help="observe an externally driven workload"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("config/face_presets.json"))
    parser.add_argument("--cases", default="1x1,4x1,8x1,1x4,4x2,4x4")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--transport", choices=("raw", "jpeg"), default="raw")
    parser.add_argument("--window", type=int, choices=(1, 2, 5), default=1)
    parser.add_argument("--mode", choices=("face_swap", "blur"), default="face_swap")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120 or not 0 <= args.warmup <= 30 or not 0.1 <= args.interval <= 2:
        parser.error("invalid measurement duration or sample interval")
    if (args.width, args.height) not in ((640, 480), (1280, 720), (1920, 1080)):
        parser.error("unsupported resolution")
    asyncio.run(main(args))
