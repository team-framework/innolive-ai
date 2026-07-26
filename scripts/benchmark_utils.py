"""Shared video loading and statistics for transport acceptance tools."""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

import cv2
import numpy as np

LONG_EDGE = 640
JPEG_QUALITY = 90


def resize_long_edge(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, LONG_EDGE / max(width, height))
    target = (max(32, round(width * scale)), max(32, round(height * scale)))
    if target == (width, height):
        return image
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def load_jpegs(video_path: Path, frame_limit: int) -> list[bytes]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frames: list[bytes] = []
    try:
        while len(frames) < frame_limit:
            ok, image = capture.read()
            if not ok:
                break
            encoded, jpeg = cv2.imencode(
                ".jpg",
                resize_long_edge(image),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if not encoded:
                raise RuntimeError(f"JPEG encode failed at frame {len(frames) + 1}")
            frames.append(jpeg.tobytes())
    finally:
        capture.release()
    if len(frames) < frame_limit:
        raise RuntimeError(f"video supplied {len(frames)} frames; {frame_limit} are required")
    return frames


def distribution(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(float(statistics.fmean(values)), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "max": round(float(max(values)), 2),
    }


def percentile(values: list[float | int], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(np.ceil(len(ordered) * percent / 100)) - 1))
    return ordered[index]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
