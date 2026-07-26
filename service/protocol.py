"""ILF1 request and ILR1 result codecs for the browser WebSocket gateway."""

from __future__ import annotations

import json
import struct
from typing import Any

MAGIC = b"ILF1"
RESULT_MAGIC = b"ILR1"
VERSION = 2
HEADER = struct.Struct("!4sI")
RESULT_HEADER = struct.Struct("!4sII")
MAX_SEQUENCE = 2**32 - 1
MAX_JPEG_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_GRPC_RESPONSE_BYTES = MAX_JPEG_BYTES + MAX_RESPONSE_BYTES


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
    return _encode_metadata(metadata, terminal_type="error").decode("utf-8")


def decode_response(payload: str) -> dict[str, Any]:
    return _decode_metadata(payload.encode("utf-8"), terminal_type="error")


def encode_result(metadata: dict[str, Any], jpeg: bytes) -> bytes:
    encoded_metadata = _encode_metadata(metadata, terminal_type="result")
    _validate_jpeg(jpeg, MAX_JPEG_BYTES)
    sequence = metadata["seq"]
    return (
        RESULT_HEADER.pack(RESULT_MAGIC, sequence, len(encoded_metadata)) + encoded_metadata + jpeg
    )


def decode_result(
    payload: bytes,
    *,
    max_jpeg_bytes: int = MAX_JPEG_BYTES,
) -> tuple[dict[str, Any], bytes]:
    if len(payload) < RESULT_HEADER.size:
        raise ValueError("result is shorter than the ILR1 header")
    magic, sequence, metadata_length = RESULT_HEADER.unpack_from(payload)
    if magic != RESULT_MAGIC:
        raise ValueError("result has an unknown protocol magic")
    if metadata_length > MAX_RESPONSE_BYTES:
        raise ValueError(f"result metadata exceeds {MAX_RESPONSE_BYTES} byte limit")

    metadata_end = RESULT_HEADER.size + metadata_length
    if metadata_end > len(payload):
        raise ValueError("result metadata is truncated")
    metadata = _decode_metadata(
        payload[RESULT_HEADER.size : metadata_end],
        terminal_type="result",
    )
    if metadata["seq"] != sequence:
        raise ValueError("result header and metadata seq do not match")

    jpeg = payload[metadata_end:]
    _validate_jpeg(jpeg, max_jpeg_bytes)
    return metadata, jpeg


def _encode_metadata(metadata: dict[str, Any], *, terminal_type: str) -> bytes:
    _validate_metadata(metadata, terminal_type=terminal_type, require_version=False)
    payload = {**metadata, "v": VERSION}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("response metadata is not JSON serializable") from error
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError(f"response exceeds {MAX_RESPONSE_BYTES} byte limit")
    return encoded


def _decode_metadata(payload: bytes, *, terminal_type: str) -> dict[str, Any]:
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("response is not valid JSON") from error
    _validate_metadata(metadata, terminal_type=terminal_type, require_version=True)
    return metadata


def _validate_metadata(
    metadata: Any,
    *,
    terminal_type: str,
    require_version: bool,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("response must be a JSON object")
    if require_version and metadata.get("v") != VERSION:
        raise ValueError("response has an unsupported protocol version")
    if metadata.get("type") != terminal_type:
        raise ValueError(f"response type must be {terminal_type}")
    sequence = metadata.get("seq")
    if type(sequence) is not int or not 0 <= sequence <= MAX_SEQUENCE:
        raise ValueError("response has an invalid seq")


def _validate_jpeg(jpeg: bytes, byte_limit: int) -> None:
    if not jpeg:
        raise ValueError("frame is empty")
    if len(jpeg) > byte_limit:
        raise ValueError(f"frame exceeds {byte_limit} byte limit")
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("frame is not a complete JPEG")
