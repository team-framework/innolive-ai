#!/usr/bin/env python3
"""Config-driven ghz runner with reproducible reports and server monitoring."""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "benchmarks" / "config.toml"
LOCAL_GHZ = ROOT / ".benchmark-tools" / "bin" / "ghz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--target", help="Override host:port from the config")
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start and stop the configured local server around the benchmark",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        help="PID of an already running local server to monitor",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Only run the named scenario; may be specified more than once",
    )
    parser.add_argument("--ghz-bin", type=Path, help="Path to the ghz binary")
    parser.add_argument(
        "--video",
        type=Path,
        help="Use sampled frames from this video as bidi-stream request data",
    )
    parser.add_argument("--video-sample-fps", type=float)
    parser.add_argument("--video-max-frames", type=int)
    parser.add_argument("--video-jpeg-quality", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.resolve().open("rb") as file:
        return tomllib.load(file)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def find_ghz(explicit: Path | None) -> Path:
    candidates = [explicit, Path(os.environ["GHZ_BIN"]) if "GHZ_BIN" in os.environ else None]
    candidates.extend([Path(found) if (found := shutil.which("ghz")) else None, LOCAL_GHZ])
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RuntimeError(
        "ghz was not found. Run benchmarks/scripts/install_ghz.sh first, "
        "or pass --ghz-bin."
    )


def split_target(target: str) -> tuple[str, int]:
    host, separator, port = target.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(f"Invalid target {target!r}; expected host:port")
    return host.strip("[]"), int(port)


def wait_for_target(target: str, timeout: float) -> None:
    host, port = split_target(target)
    deadline = time.monotonic() + timeout
    error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError as exc:
            error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Server {target} was not ready after {timeout}s: {error}")


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return (result.stdout.strip() or result.stderr.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = command_output(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def package_version(distribution: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata(ghz: Path, config_path: Path, target: str) -> dict[str, Any]:
    status = command_output(["git", "status", "--porcelain"])
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "target": target,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes(),
        "python": sys.version,
        "dependencies": {
            "grpcio": package_version("grpcio"),
            "Pillow": package_version("Pillow"),
        },
        "ghz": command_output([str(ghz), "--version"]),
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "branch": command_output(["git", "branch", "--show-current"]),
            "dirty": bool(status),
        },
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path.resolve()),
    }


class ProcessMonitor:
    def __init__(self, pid: int, output: Path, interval: float):
        self.pid = pid
        self.output = output
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(2, self.interval * 4))

    def _run(self) -> None:
        started = time.monotonic()
        with self.output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["elapsed_seconds", "cpu_percent", "rss_bytes"])
            while not self.stop_event.is_set():
                result = subprocess.run(
                    ["ps", "-p", str(self.pid), "-o", "%cpu=", "-o", "rss="],
                    capture_output=True,
                    text=True,
                )
                values = result.stdout.split()
                if result.returncode != 0 or len(values) < 2:
                    break
                writer.writerow(
                    [f"{time.monotonic() - started:.3f}", values[0], int(values[1]) * 1024]
                )
                file.flush()
                self.stop_event.wait(self.interval)


def build_image_message(scenario: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    payload = resolve_path(scenario["payload_file"])
    payload_bytes = payload.read_bytes()
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    message = {scenario.get("payload_field", "data"): encoded}
    timestamp_field = scenario.get("timestamp_field")
    if timestamp_field:
        message[timestamp_field] = "{{.TimestampUnixMilli}}"
    return message, {
        "payload_source": str(payload.relative_to(ROOT)),
        "payload_source_sha256": file_sha256(payload),
        "payload_bytes_average": len(payload_bytes),
        "payload_bytes_min": len(payload_bytes),
        "payload_bytes_max": len(payload_bytes),
        "payload_bytes_per_call": len(payload_bytes),
        "messages_per_call": 1,
    }


def build_video_messages(
    scenario: dict[str, Any], video_path: Path, video_config: dict[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required when --video is used") from exc

    video_path = video_path.resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    requested_fps = float(video_config["sample_fps"])
    sample_fps = min(requested_fps, source_fps) if source_fps > 0 else requested_fps
    if sample_fps <= 0:
        capture.release()
        raise ValueError("video.sample_fps must be greater than zero")
    max_frames = int(video_config["max_frames"])
    quality = int(video_config["jpeg_quality"])
    if max_frames <= 0 or not 1 <= quality <= 100:
        capture.release()
        raise ValueError("video.max_frames must be positive and jpeg_quality must be 1..100")

    payload_field = scenario.get("payload_field", "data")
    timestamp_field = scenario.get("timestamp_field")
    messages: list[dict[str, str]] = []
    byte_sizes: list[int] = []
    frame_index = 0
    next_sample_time = 0.0
    while len(messages) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_time = frame_index / source_fps if source_fps > 0 else len(messages) / sample_fps
        frame_index += 1
        if frame_time + 1e-9 < next_sample_time:
            continue
        encoded_ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not encoded_ok:
            capture.release()
            raise RuntimeError(f"Failed to JPEG-encode frame {frame_index - 1}")
        frame_bytes = encoded.tobytes()
        message = {payload_field: base64.b64encode(frame_bytes).decode("ascii")}
        if timestamp_field:
            message[timestamp_field] = "{{.TimestampUnixMilli}}"
        messages.append(message)
        byte_sizes.append(len(frame_bytes))
        next_sample_time += 1.0 / sample_fps
    capture.release()
    if not messages:
        raise RuntimeError(f"No frames were read from video: {video_path}")

    return messages, {
        "payload_source": str(video_path),
        "payload_source_sha256": file_sha256(video_path),
        "payload_bytes_average": sum(byte_sizes) / len(byte_sizes),
        "payload_bytes_min": min(byte_sizes),
        "payload_bytes_max": max(byte_sizes),
        "payload_bytes_per_call": sum(byte_sizes),
        "messages_per_call": len(messages),
        "video": {
            "source_fps": source_fps,
            "source_frame_count": source_frames,
            "source_duration_seconds": (
                source_frames / source_fps if source_fps > 0 else None
            ),
            "width": width,
            "height": height,
            "sample_fps": sample_fps,
            "sampled_frame_count": len(messages),
            "jpeg_quality": quality,
        },
    }


def percentile_ns(report: dict[str, Any], percentile: int) -> int | None:
    for item in report.get("latencyDistribution", []):
        if item.get("percentage") == percentile:
            return item.get("latency")
    return None


def monitor_summary(path: Path) -> dict[str, float | int] | None:
    if not path.exists():
        return None
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    cpus = [float(row["cpu_percent"]) for row in rows]
    rss = [int(row["rss_bytes"]) for row in rows]
    return {
        "samples": len(rows),
        "cpu_percent_average": sum(cpus) / len(cpus),
        "cpu_percent_max": max(cpus),
        "rss_bytes_max": max(rss),
    }


def error_count(report: dict[str, Any]) -> int:
    statuses = report.get("statusCodeDistribution", {})
    return sum(count for status, count in statuses.items() if status != "OK")


def run_scenario(
    ghz: Path,
    proto: Path,
    target: str,
    config: dict[str, Any],
    profile: dict[str, Any],
    scenario: dict[str, Any],
    work_dir: Path,
    video_path: Path | None,
    video_config: dict[str, Any],
    server_pid: int | None,
    monitor_interval: float,
) -> tuple[dict[str, Any], list[str]]:
    if video_path and scenario["kind"] == "bidi_stream":
        request_data, payload_metadata = build_video_messages(
            scenario, video_path, video_config
        )
    else:
        request_data, payload_metadata = build_image_message(scenario)
    data_file = work_dir / f"{scenario['name']}.request.json"
    data_file.write_text(json.dumps(request_data), encoding="utf-8")
    raw_output = work_dir / f"{scenario['name']}.ghz.json"

    command = [
        str(ghz),
        "--proto", str(proto),
        "--call", scenario["call"],
        "--concurrency", str(scenario.get("concurrency", profile["concurrency"])),
        "--connections", str(scenario.get("connections", profile["connections"])),
        "--timeout", str(scenario.get("timeout", config["timeout"])),
        "--duration-stop", str(config.get("duration_stop", "wait")),
        "--data-file", str(data_file),
        "--format", "pretty",
        "--output", str(raw_output),
        "--name", scenario["name"],
    ]
    if "total" in scenario or "total" in profile:
        command.extend(["--total", str(scenario.get("total", profile.get("total")))])
    else:
        command.extend(["--duration", str(scenario.get("duration", profile["duration"]))])
    if config.get("insecure", True):
        command.append("--insecure")
    if scenario["kind"] == "bidi_stream":
        stream_call_count = payload_metadata["messages_per_call"]
        stream_interval = str(scenario.get("stream_interval", "0s"))
        if video_path and profile.get("video_pacing") == "source":
            stream_interval = f"{1 / payload_metadata['video']['sample_fps']:.6f}s"
        command.extend(
            [
                "--stream-call-count",
                str(
                    stream_call_count
                    if video_path
                    else scenario.get("stream_call_count", profile["stream_call_count"])
                ),
                "--stream-interval",
                stream_interval,
                "--stream-dynamic-messages",
            ]
        )
    command.extend(str(value) for value in profile.get("extra_args", []))
    command.extend(str(value) for value in scenario.get("extra_args", []))
    command.append(target)

    warmup = int(config.get("warmup_total", 0))
    if warmup:
        warmup_command = command.copy()
        if "--duration" in warmup_command:
            duration_index = warmup_command.index("--duration")
            del warmup_command[duration_index : duration_index + 2]
        elif "--total" in warmup_command:
            total_index = warmup_command.index("--total")
            del warmup_command[total_index : total_index + 2]
        output_index = warmup_command.index("--output")
        del warmup_command[output_index : output_index + 2]
        format_index = warmup_command.index("--format")
        warmup_command[format_index + 1] = "summary"
        warmup_command.extend(["--total", str(warmup)])
        subprocess.run(warmup_command, cwd=ROOT, check=True, capture_output=True)

    scenario_monitor: ProcessMonitor | None = None
    resource_path = work_dir / f"{scenario['name']}.server-resources.csv"
    if server_pid:
        scenario_monitor = ProcessMonitor(server_pid, resource_path, monitor_interval)
        scenario_monitor.start()
    print(f"Running {scenario['name']}: {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        if scenario_monitor:
            scenario_monitor.stop()
    report = json.loads(raw_output.read_text())
    configured_messages = (
        payload_metadata["messages_per_call"]
        if video_path and scenario["kind"] == "bidi_stream"
        else int(scenario.get("stream_call_count", profile["stream_call_count"]))
        if scenario["kind"] == "bidi_stream"
        else 1
    )
    report["benchmarkMetadata"] = {
        **payload_metadata,
        "kind": scenario["kind"],
        "messages_per_call": configured_messages,
        "server_resources": monitor_summary(resource_path),
        "load": {
            "concurrency": int(scenario.get("concurrency", profile["concurrency"])),
            "connections": int(scenario.get("connections", profile["connections"])),
            "duration": scenario.get("duration", profile.get("duration")),
            "total": scenario.get("total", profile.get("total")),
            "video_pacing": profile.get("video_pacing"),
        },
    }

    failures: list[str] = []
    count = int(report.get("count", 0))
    errors = error_count(report)
    error_rate = errors / count if count else 1.0
    thresholds = scenario.get("thresholds", {})
    if error_rate > thresholds.get("max_error_rate", 1.0):
        failures.append(
            f"{scenario['name']}: error rate {error_rate:.2%} exceeds "
            f"{thresholds['max_error_rate']:.2%}"
        )
    p95 = percentile_ns(report, 95)
    if "max_p95_ms" in thresholds and p95 is not None:
        p95_ms = p95 / 1_000_000
        if p95_ms > thresholds["max_p95_ms"]:
            failures.append(
                f"{scenario['name']}: p95 {p95_ms:.2f} ms exceeds "
                f"{thresholds['max_p95_ms']:.2f} ms"
            )
    return report, failures


def nanoseconds_to_ms(value: int | float | None) -> float | None:
    return None if value is None else value / 1_000_000


def compact_scenario(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["benchmarkMetadata"]
    calls_per_second = float(report.get("rps", 0))
    messages_per_second = calls_per_second * metadata["messages_per_call"]
    compact = {
        "name": report.get("name", "unknown"),
        "kind": metadata["kind"],
        "calls": int(report.get("count", 0)),
        "messages_per_call": metadata["messages_per_call"],
        "load": metadata["load"],
        "metrics": {
            "calls_per_second": calls_per_second,
            "messages_per_second": messages_per_second,
            "app_payload_mib_per_second": (
                messages_per_second * metadata["payload_bytes_average"] / 1024 / 1024
            ),
            "latency_ms": {
                "average": nanoseconds_to_ms(report.get("average")),
                "p50": nanoseconds_to_ms(percentile_ns(report, 50)),
                "p95": nanoseconds_to_ms(percentile_ns(report, 95)),
                "p99": nanoseconds_to_ms(percentile_ns(report, 99)),
            },
            "errors": error_count(report),
            "status_codes": report.get("statusCodeDistribution", {}),
        },
        "payload": {
            "source": metadata["payload_source"],
            "source_sha256": metadata["payload_source_sha256"],
            "bytes_average": metadata["payload_bytes_average"],
            "bytes_min": metadata["payload_bytes_min"],
            "bytes_max": metadata["payload_bytes_max"],
            "bytes_per_call": metadata["payload_bytes_per_call"],
        },
        "server_resources": metadata["server_resources"],
    }
    if "video" in metadata:
        compact["video"] = metadata["video"]
        sample_fps = metadata["video"]["sample_fps"]
        concurrency = metadata["load"]["concurrency"]
        compact["metrics"]["effective_fps_per_stream"] = (
            messages_per_second / concurrency
        )
        compact["metrics"]["realtime_target_ratio"] = (
            messages_per_second / (sample_fps * concurrency)
        )
    return compact


def build_result(
    profile_name: str,
    target: str,
    reports: list[dict[str, Any]],
    environment: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recorded_at": dt.datetime.now().astimezone().isoformat(),
        "profile": profile_name,
        "target": target,
        "environment": environment,
        "thresholds": {"passed": not failures, "failures": failures},
        "scenarios": [compact_scenario(report) for report in reports],
    }


def format_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_summary(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# gRPC performance benchmark",
        "",
        f"- Profile: `{result['profile']}`",
        f"- Target: `{result['target']}`",
        f"- Recorded at: {result['recorded_at']}",
        "",
        "## Results",
        "",
        "| Scenario | Messages/s | App MiB/s | Avg ms | p95 ms | p99 ms | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in result["scenarios"]:
        metrics = scenario["metrics"]
        latency = metrics["latency_ms"]
        lines.append(
            "| {name} | {messages:.2f} | {mib:.2f} | {avg} | {p95} | {p99} | {errors} |".format(
                name=scenario["name"],
                messages=metrics["messages_per_second"],
                mib=metrics["app_payload_mib_per_second"],
                avg=format_number(latency["average"]),
                p95=format_number(latency["p95"]),
                p99=format_number(latency["p99"]),
                errors=metrics["errors"],
            )
        )
    lines.extend(
        [
            "",
            "> Streaming latency is measured per completed stream call by ghz; "
            "Messages/s is derived from calls/s × configured messages per call.",
        ]
    )
    video_scenarios = [item for item in result["scenarios"] if "video" in item]
    if video_scenarios:
        lines.extend(["", "## Video", ""])
        for scenario in video_scenarios:
            video = scenario["video"]
            metrics = scenario["metrics"]
            lines.append(
                f"- {video['width']}x{video['height']} @ {video['sample_fps']:.0f} FPS, "
                f"{video['sampled_frame_count']} frames; effective "
                f"{metrics['effective_fps_per_stream']:.2f} FPS/stream "
                f"({metrics['realtime_target_ratio']:.1%} of target)."
            )
    resource_reports = [
        (scenario["name"], scenario.get("server_resources"))
        for scenario in result["scenarios"]
    ]
    resource_reports = [(name, stats) for name, stats in resource_reports if stats]
    if resource_reports:
        lines.extend(
            [
                "",
                "## Server process resources during measured load",
                "",
                "| Scenario | Samples | Avg CPU | Peak CPU | Peak RSS |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, stats in resource_reports:
            lines.append(
                f"| {name} | {stats['samples']} | "
                f"{stats['cpu_percent_average']:.2f}% | "
                f"{stats['cpu_percent_max']:.2f}% | "
                f"{stats['rss_bytes_max'] / 1024 / 1024:.2f} MiB |"
            )
    lines.extend(["", "## Thresholds", ""])
    lines.extend(
        [f"- FAIL: {failure}" for failure in result["thresholds"]["failures"]]
        or ["- All configured thresholds passed."]
    )
    output.write_text("\n".join(lines) + "\n")


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def write_chart(output: Path, result: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw

    width, height = 1400, 760
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    heading_font = load_font(24, bold=True)
    label_font = load_font(19)
    value_font = load_font(19, bold=True)
    muted = "#64748b"
    foreground = "#172033"
    grid = "#d9e0ea"
    throughput_color = "#2878d0"
    latency_colors = ["#2f9e74", "#e69f00", "#d95f59"]

    draw.text((60, 42), f"gRPC benchmark · {result['profile']}", fill=foreground, font=title_font)
    draw.text((60, 92), result["recorded_at"], fill=muted, font=label_font)
    scenarios = result["scenarios"]
    chart_top = 180
    panel_height = 470
    left_x, left_width = 70, 560
    right_x, right_width = 760, 570
    draw.text((left_x, 140), "Message throughput", fill=foreground, font=heading_font)
    draw.text((right_x, 140), "Stream-call latency", fill=foreground, font=heading_font)

    max_messages = max((item["metrics"]["messages_per_second"] for item in scenarios), default=1) or 1
    row_height = max(90, panel_height // max(len(scenarios), 1))
    for index, scenario in enumerate(scenarios):
        y = chart_top + index * row_height
        metrics = scenario["metrics"]
        draw.text((left_x, y), scenario["name"], fill=foreground, font=label_font)
        bar_y = y + 34
        bar_width = int((left_width - 30) * metrics["messages_per_second"] / max_messages)
        draw.rounded_rectangle((left_x, bar_y, left_x + max(bar_width, 4), bar_y + 28), 7, fill=throughput_color)
        draw.text(
            (left_x, bar_y + 38),
            f"{metrics['messages_per_second']:.2f} messages/s  ·  {metrics['app_payload_mib_per_second']:.2f} MiB/s",
            fill=foreground,
            font=value_font,
        )

    latency_values = []
    for scenario in scenarios:
        latency = scenario["metrics"]["latency_ms"]
        latency_values.extend(value for value in (latency["average"], latency["p95"], latency["p99"]) if value is not None)
    max_latency = max(latency_values, default=1) or 1
    for index, scenario in enumerate(scenarios):
        base_y = chart_top + index * row_height
        draw.text((right_x, base_y), scenario["name"], fill=foreground, font=label_font)
        latency = scenario["metrics"]["latency_ms"]
        for metric_index, (label, key) in enumerate((("avg", "average"), ("p95", "p95"), ("p99", "p99"))):
            value = latency[key]
            if value is None:
                continue
            y = base_y + 34 + metric_index * 31
            draw.text((right_x, y), label, fill=muted, font=label_font)
            start_x = right_x + 55
            bar_width = int((right_width - 165) * value / max_latency)
            draw.rounded_rectangle((start_x, y + 2, start_x + max(bar_width, 4), y + 23), 5, fill=latency_colors[metric_index])
            draw.text((start_x + bar_width + 10, y), f"{value:.2f} ms", fill=foreground, font=value_font)

    draw.line((60, 680, width - 60, 680), fill=grid, width=2)
    total_errors = sum(item["metrics"]["errors"] for item in scenarios)
    draw.text((60, 704), f"Errors: {total_errors}", fill=foreground, font=value_font)
    video_items = [item for item in scenarios if "video" in item]
    if video_items:
        video = video_items[0]["video"]
        draw.text(
            (300, 704),
            f"Input: {video['width']}×{video['height']} · {video['sample_fps']:.0f} FPS · {video['sampled_frame_count']} frames",
            fill=foreground,
            font=value_font,
        )
    image.save(output, format="PNG", optimize=True)


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    try:
        profile = config["profiles"][args.profile]
    except KeyError:
        available = ", ".join(config.get("profiles", {}))
        raise SystemExit(f"Unknown profile {args.profile!r}. Available: {available}")

    scenarios = config["scenarios"]
    if args.scenario:
        selected = set(args.scenario)
        scenarios = [item for item in scenarios if item["name"] in selected]
        missing = selected - {item["name"] for item in scenarios}
        if missing:
            raise SystemExit(f"Unknown scenarios: {', '.join(sorted(missing))}")

    ghz = find_ghz(args.ghz_bin)
    video_path = args.video.resolve() if args.video else None
    video_config = dict(config.get("video", {}))
    if args.video_sample_fps is not None:
        video_config["sample_fps"] = args.video_sample_fps
    if args.video_max_frames is not None:
        video_config["max_frames"] = args.video_max_frames
    if args.video_jpeg_quality is not None:
        video_config["jpeg_quality"] = args.video_jpeg_quality
    target = args.target or config["target"]
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_dir = resolve_path(config["results_dir"]) / f"{timestamp}-{args.profile}"
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = environment_metadata(ghz, config_path, target)
    work_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"grpc-benchmark-{os.getpid()}"
    work_dir.mkdir(parents=True, exist_ok=False)

    server_process: subprocess.Popen[Any] | None = None
    server_pid: int | None = None
    try:
        if args.start_server:
            if args.server_pid:
                raise SystemExit("--start-server and --server-pid cannot be used together")
            server_config = config["server"]
            command = [part.format(python=sys.executable) for part in server_config["command"]]
            command.extend(str(value) for value in profile.get("server_args", []))
            log = (work_dir / "server.log").open("w")
            server_process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            wait_for_target(target, float(server_config.get("ready_timeout_seconds", 15)))
            server_pid = server_process.pid
        else:
            wait_for_target(target, 3)
            server_pid = args.server_pid

        proto = resolve_path(config["proto"])
        reports: list[dict[str, Any]] = []
        failures: list[str] = []
        for scenario in scenarios:
            report, scenario_failures = run_scenario(
                ghz,
                proto,
                target,
                config,
                profile,
                scenario,
                work_dir,
                video_path,
                video_config,
                server_pid,
                float(config.get("monitor_interval_seconds", 0.5)),
            )
            reports.append(report)
            failures.extend(scenario_failures)
    finally:
        if server_process:
            terminate_process(server_process)
        shutil.rmtree(work_dir, ignore_errors=True)

    result = build_result(
        args.profile,
        target,
        reports,
        environment,
        failures,
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_summary(output_dir / "summary.md", result)
    write_chart(output_dir / "summary.png", result)
    print(f"Reports written to {output_dir}")
    if failures:
        print("Threshold failures:", *failures, sep="\n- ", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
