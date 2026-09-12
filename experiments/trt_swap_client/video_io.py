"""Optional NVDEC/NVENC bridge for the independent TensorRT swap lab.

Frames intentionally enter the Python compositor as BGR arrays because the
existing tracker, AdaFace and InSwapper APIs operate on OpenCV arrays.  This
uses NVDEC/NVENC for compressed video I/O, but is *not* a zero-copy inference
path: one device-to-host transfer occurs after decoding and one host-to-device
transfer occurs before encoding.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class VideoSpec:
    width: int
    height: int
    fps: float


def require_nvcodec_ffmpeg() -> str:
    """Return ffmpeg only when its installed encoder supports NVENC."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --input-video")
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or "h264_nvenc" not in completed.stdout:
        raise RuntimeError("ffmpeg must be built with the h264_nvenc encoder")
    return ffmpeg


def probe_video(path: Path) -> VideoSpec:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for --input-video")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        stream = json.loads(result.stdout)["streams"][0]
        fps = float(Fraction(stream["avg_frame_rate"]))
        spec = VideoSpec(int(stream["width"]), int(stream["height"]), fps)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"could not determine video dimensions for {path}") from error
    if spec.width < 1 or spec.height < 1 or not 0 < spec.fps <= 240:
        raise RuntimeError(f"invalid video spec: {spec}")
    return spec


class NvdecReader:
    """Decode a compressed file with NVDEC and expose BGR frames to OpenCV."""

    def __init__(self, path: Path, spec: VideoSpec, *, ffmpeg: str):
        self.spec = spec
        self._frame_bytes = spec.width * spec.height * 3
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(path),
                "-vf",
                "hwdownload,format=bgr24",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self) -> np.ndarray | None:
        if self._process.stdout is None:
            raise RuntimeError("NVDEC stdout is unavailable")
        raw = self._process.stdout.read(self._frame_bytes)
        if not raw:
            return None
        if len(raw) != self._frame_bytes:
            raise RuntimeError("NVDEC returned a partial frame")
        return (
            np.frombuffer(raw, dtype=np.uint8)
            .reshape((self.spec.height, self.spec.width, 3))
            .copy()
        )

    def close(self) -> None:
        _close_process(self._process)


class NvencHlsWriter:
    """Encode processed BGR frames with NVENC into a short low-latency HLS playlist."""

    def __init__(self, output_dir: Path, spec: VideoSpec, *, ffmpeg: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.playlist = output_dir / "live.m3u8"
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-video_size",
                f"{spec.width}x{spec.height}",
                "-framerate",
                f"{spec.fps:.6f}",
                "-i",
                "pipe:0",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-tune",
                "ll",
                "-rc",
                "cbr",
                "-b:v",
                "12M",
                "-maxrate",
                "12M",
                "-bufsize",
                "6M",
                "-g",
                str(max(1, round(spec.fps))),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_list_size",
                "3",
                "-hls_flags",
                "delete_segments+append_list+independent_segments",
                str(self.playlist),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("NVENC stdin is unavailable")
        if frame.shape != (self.spec.height, self.spec.width, 3):
            raise ValueError("NVENC frame dimensions changed")
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        _close_process(self._process)


def _close_process(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            stream.close()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    if process.returncode not in (0, None):
        stderr = (
            process.stderr.read().decode("utf-8", errors="replace").strip()
            if process.stderr
            else ""
        )
        raise RuntimeError(f"ffmpeg failed ({process.returncode}): {stderr}")
