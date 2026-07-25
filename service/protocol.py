"""ILF1 single-frame request and metadata-only terminal response codec."""

from __future__ import annotations

import json
import struct
from typing import Any


MAGIC = b"ILF1"
VERSION = 1
HEADER = struct.Struct("!4sI")
MAX_SEQUENCE = 2**32 - 1
MAX_JPEG_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024


def encode_request(sequence: int, jpeg: bytes) -> bytes:
    if not 0 <= sequence <= MAX_SEQUENCE:
        raise ValueError(f"sequence must be in 0..{MAX_SEQUENCE}")
    _validate_jpeg(jpeg, MAX_JPEG_BYTES)
    return HEADER.pack(MAGIC, sequence) + jpeg


def decode_request(
    payload: bytes,
    *,
    max_jpeg_bytes: int = MAX_JPEG_BYTES,
) -> tuple[int, bytes]:
    if len(payload) < HEADER.size:
        raise ValueError("request is shorter than the ILF1 header")
    magic, sequence = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError("request has an unknown protocol magic")
    jpeg = payload[HEADER.size :]
    _validate_jpeg(jpeg, max_jpeg_bytes)
    return sequence, jpeg


def recover_sequence(payload: bytes) -> int | None:
    """Recover a sequence only when a complete eight-byte header exists."""
    if len(payload) < HEADER.size:
        return None
    return HEADER.unpack_from(payload)[1]


def encode_response(metadata: dict[str, Any]) -> str:
    terminal_type = metadata.get("type")
    sequence = metadata.get("seq")
    if terminal_type not in {"result", "error"}:
        raise ValueError("response type must be result or error")
    if not isinstance(sequence, int) or not 0 <= sequence <= MAX_SEQUENCE:
        raise ValueError("response must contain an unsigned 32-bit seq")
    payload = {"v": VERSION, **metadata}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError(f"response exceeds {MAX_RESPONSE_BYTES} byte limit")
    return encoded


def decode_response(payload: str) -> dict[str, Any]:
    try:
        metadata = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("response is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("response must be a JSON object")
    if metadata.get("v") != VERSION:
        raise ValueError("response has an unsupported protocol version")
    if metadata.get("type") not in {"result", "error"}:
        raise ValueError("response has an unknown terminal type")
    sequence = metadata.get("seq")
    if not isinstance(sequence, int) or not 0 <= sequence <= MAX_SEQUENCE:
        raise ValueError("response has an invalid seq")
    return metadata


def _validate_jpeg(jpeg: bytes, byte_limit: int) -> None:
    if not jpeg:
        raise ValueError("frame is empty")
    if len(jpeg) > byte_limit:
        raise ValueError(f"frame exceeds {byte_limit} byte limit")
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("frame is not a complete JPEG")
