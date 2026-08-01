"""Bounded image decoding for video frames and face enrollment."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import cv2
import numpy as np

from service.protocol import MAX_JPEG_BYTES

MIN_FRAME_DIMENSION = 32
# Long-edge ceiling for a decoded frame. FHD (1920x1080) is the largest source
# the serving profile accepts: the detector letterboxes to imgsz=640 internally
# ("detect small") while the mosaic blur runs at full resolution ("blur big"),
# so raising this from the original 640 preserves output quality up to FHD.
MAX_LONG_EDGE = 1920
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


def decode_image(encoded: bytes, limits: FrameLimits = DEFAULT_FRAME_LIMITS) -> np.ndarray:
    """Decode one bounded JPEG, PNG, or WebP enrollment image."""

    if encoded.startswith(b"\xff\xd8"):
        return decode_jpeg(encoded, limits)
    if not encoded:
        raise ValueError("encoded image must not be empty")
    if len(encoded) > limits.max_jpeg_bytes:
        raise ValueError(f"encoded image exceeds {limits.max_jpeg_bytes} byte limit")

    width, height = _non_jpeg_dimensions(encoded)
    _validate_dimensions(width, height, limits)
    try:
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError("encoded image could not be decoded") from error
    if image is None:
        raise ValueError("encoded image could not be decoded")
    decoded_height, decoded_width = image.shape[:2]
    _validate_dimensions(decoded_width, decoded_height, limits)
    return image


def _non_jpeg_dimensions(encoded: bytes) -> tuple[int, int]:
    if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png_dimensions(encoded)
    if encoded.startswith(b"RIFF") and encoded[8:12] == b"WEBP":
        return _webp_dimensions(encoded)
    raise ValueError("unsupported image format; expected JPEG, PNG, or WebP")


def _png_dimensions(encoded: bytes) -> tuple[int, int]:
    offset = 8
    dimensions = None
    while offset + 12 <= len(encoded):
        chunk_size = int.from_bytes(encoded[offset : offset + 4], "big")
        chunk_type = encoded[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        chunk_end = payload_end + 4
        if chunk_end > len(encoded):
            raise ValueError("encoded image contains a truncated PNG chunk")
        expected_crc = int.from_bytes(encoded[payload_end:chunk_end], "big")
        actual_crc = zlib.crc32(encoded[offset + 4 : payload_end]) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("encoded image contains an invalid PNG checksum")
        if dimensions is None:
            if chunk_type != b"IHDR" or chunk_size != 13:
                raise ValueError("encoded image contains an invalid PNG header")
            dimensions = (
                int.from_bytes(encoded[payload_start : payload_start + 4], "big"),
                int.from_bytes(encoded[payload_start + 4 : payload_start + 8], "big"),
            )
        elif chunk_type == b"IHDR":
            raise ValueError("encoded image contains more than one PNG header")
        if chunk_type == b"acTL":
            raise ValueError("animated PNG is not supported for face enrollment")
        if chunk_type == b"IEND":
            if chunk_size != 0 or chunk_end != len(encoded):
                raise ValueError("encoded image contains an invalid PNG ending")
            if dimensions is None:
                raise ValueError("encoded image contains no PNG header")
            return dimensions
        offset = chunk_end
    raise ValueError("encoded image contains no complete PNG image")


def _webp_dimensions(encoded: bytes) -> tuple[int, int]:
    if len(encoded) < 20 or int.from_bytes(encoded[4:8], "little") + 8 != len(encoded):
        raise ValueError("encoded image contains an invalid WebP container")
    offset = 12
    canvas_dimensions = None
    frame_dimensions = None
    while offset + 8 <= len(encoded):
        chunk_type = encoded[offset : offset + 4]
        chunk_size = int.from_bytes(encoded[offset + 4 : offset + 8], "little")
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        padded_end = payload_end + (chunk_size & 1)
        if padded_end > len(encoded):
            raise ValueError("encoded image contains a truncated WebP chunk")
        payload = encoded[payload_start:payload_end]
        if chunk_type == b"VP8X":
            if len(payload) != 10 or canvas_dimensions is not None:
                raise ValueError("encoded image contains an invalid WebP canvas")
            if payload[0] & 0x02:
                raise ValueError("animated WebP is not supported for face enrollment")
            canvas_dimensions = (
                1 + int.from_bytes(payload[4:7], "little"),
                1 + int.from_bytes(payload[7:10], "little"),
            )
        elif chunk_type in (b"ANIM", b"ANMF"):
            raise ValueError("animated WebP is not supported for face enrollment")
        elif chunk_type == b"VP8L" and len(payload) >= 5 and payload[0] == 0x2F:
            bits = int.from_bytes(payload[1:5], "little")
            dimensions = (1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF))
            if frame_dimensions is not None:
                raise ValueError("encoded image contains more than one WebP frame")
            frame_dimensions = dimensions
        elif chunk_type == b"VP8 " and len(payload) >= 10 and payload[3:6] == b"\x9d\x01\x2a":
            dimensions = (
                int.from_bytes(payload[6:8], "little") & 0x3FFF,
                int.from_bytes(payload[8:10], "little") & 0x3FFF,
            )
            if frame_dimensions is not None:
                raise ValueError("encoded image contains more than one WebP frame")
            frame_dimensions = dimensions
        offset = padded_end
    if offset != len(encoded) or frame_dimensions is None:
        raise ValueError("encoded image contains no complete WebP frame")
    if canvas_dimensions is not None and canvas_dimensions != frame_dimensions:
        raise ValueError("WebP canvas dimensions do not match its frame")
    return frame_dimensions


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
