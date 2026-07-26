"""Transport-neutral JPEG boundary validation for B1-640 frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from service.protocol import MAX_JPEG_BYTES

MIN_FRAME_DIMENSION = 32
MAX_LONG_EDGE = 640
MAX_PIXELS = MAX_LONG_EDGE * MAX_LONG_EDGE
_JPEG_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_JPEG_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD9)})


@dataclass(frozen=True, slots=True)
class FrameLimits:
    """Decoded-frame and encoded-payload limits shared by every transport."""

    max_jpeg_bytes: int = MAX_JPEG_BYTES
    min_dimension: int = MIN_FRAME_DIMENSION
    max_long_edge: int = MAX_LONG_EDGE
    max_pixels: int = MAX_PIXELS

    def __post_init__(self) -> None:
        if self.max_jpeg_bytes < 1:
            raise ValueError("max_jpeg_bytes must be positive")
        if self.min_dimension < 1:
            raise ValueError("min_dimension must be positive")
        if self.max_long_edge < self.min_dimension:
            raise ValueError("max_long_edge must be at least min_dimension")
        if self.max_pixels < 1:
            raise ValueError("max_pixels must be positive")


DEFAULT_FRAME_LIMITS = FrameLimits()


def decode_jpeg(jpeg: bytes, limits: FrameLimits = DEFAULT_FRAME_LIMITS) -> np.ndarray:
    """Validate and decode exactly one complete JPEG within the serving profile."""

    if not jpeg:
        raise ValueError("frame could not be decoded as a complete JPEG")
    if len(jpeg) > limits.max_jpeg_bytes:
        raise ValueError(f"frame exceeds {limits.max_jpeg_bytes} byte limit")
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("frame could not be decoded as a complete JPEG")

    declared_width, declared_height = _jpeg_dimensions(jpeg)
    _validate_dimensions(declared_width, declared_height, limits)
    try:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("frame could not be decoded as JPEG") from error
    if image is None:
        raise ValueError("frame could not be decoded as JPEG")

    height, width = image.shape[:2]
    _validate_dimensions(width, height, limits)
    return image


def _validate_dimensions(width: int, height: int, limits: FrameLimits) -> None:
    if width < limits.min_dimension or height < limits.min_dimension:
        raise ValueError(
            f"decoded dimensions must be at least {limits.min_dimension}x{limits.min_dimension}"
        )
    if max(width, height) > limits.max_long_edge:
        raise ValueError(f"decoded frame exceeds long-edge {limits.max_long_edge} limit")
    if width * height > limits.max_pixels:
        raise ValueError(f"decoded frame exceeds {limits.max_pixels} pixel limit")


def _jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Read SOF dimensions before OpenCV can allocate a decoded image."""

    offset = 2
    payload_end = len(jpeg) - 2
    while offset < payload_end:
        marker, offset = _read_marker(jpeg, offset, payload_end)
        if marker in _JPEG_STANDALONE_MARKERS:
            continue
        segment_length, segment_end = _read_segment(jpeg, offset, payload_end)
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            return _read_frame_dimensions(jpeg, offset, segment_length)
        if marker == 0xDA:
            break
        offset = segment_end

    raise ValueError("frame does not contain a JPEG frame header")


def _read_marker(jpeg: bytes, offset: int, payload_end: int) -> tuple[int, int]:
    if jpeg[offset] != 0xFF:
        raise ValueError("frame contains an invalid JPEG marker")
    while offset < payload_end and jpeg[offset] == 0xFF:
        offset += 1
    if offset >= payload_end or jpeg[offset] == 0x00:
        raise ValueError("frame contains an invalid JPEG marker")
    return jpeg[offset], offset + 1


def _read_segment(jpeg: bytes, offset: int, payload_end: int) -> tuple[int, int]:
    if offset + 2 > payload_end:
        raise ValueError("frame contains a truncated JPEG segment")
    length = int.from_bytes(jpeg[offset : offset + 2], "big")
    if length < 2:
        raise ValueError("frame contains an invalid JPEG segment")
    end = offset + length
    if end > len(jpeg):
        raise ValueError("frame contains a truncated JPEG segment")
    return length, end


def _read_frame_dimensions(jpeg: bytes, offset: int, segment_length: int) -> tuple[int, int]:
    if segment_length < 8:
        raise ValueError("frame contains an invalid JPEG frame header")
    height = int.from_bytes(jpeg[offset + 3 : offset + 5], "big")
    width = int.from_bytes(jpeg[offset + 5 : offset + 7], "big")
    if width == 0 or height == 0:
        raise ValueError("frame contains invalid JPEG dimensions")
    return width, height
