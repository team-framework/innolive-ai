from __future__ import annotations

import json
import struct
from dataclasses import dataclass


BATCH_SIZE = 4
MAX_HEADER_BYTES = 128 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_PACKET_BYTES = 4 + MAX_HEADER_BYTES + BATCH_SIZE * MAX_FRAME_BYTES


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    frame_id: int
    captured_at: float
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame: EncodedFrame
    jpeg: bytes
    width: int
    height: int
    faces: tuple[dict, ...]
    timing: dict


def decode_batch(message: bytes) -> list[EncodedFrame]:
    if len(message) < 4:
        raise ValueError("missing packet header")
    if len(message) > MAX_PACKET_BYTES:
        raise ValueError("packet is too large")

    header_size = struct.unpack_from(">I", message)[0]
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ValueError("invalid packet header size")

    payload_offset = 4 + header_size
    if payload_offset > len(message):
        raise ValueError("truncated packet header")

    header = json.loads(message[4:payload_offset])
    if not isinstance(header, dict):
        raise ValueError("packet header must be an object")
    frames = header.get("frames", [])
    if header.get("v") != 1 or not 0 < len(frames) <= BATCH_SIZE:
        raise ValueError(f"a v1 packet must contain 1 to {BATCH_SIZE} frames")

    decoded: list[EncodedFrame] = []
    for frame in frames:
        size = int(frame["size"])
        if not 0 < size <= MAX_FRAME_BYTES:
            raise ValueError("invalid JPEG size")

        end = payload_offset + size
        if end > len(message):
            raise ValueError("truncated JPEG payload")

        decoded.append(
            EncodedFrame(
                frame_id=int(frame["id"]),
                captured_at=float(frame["capturedAt"]),
                jpeg=message[payload_offset:end],
            )
        )
        payload_offset = end

    if payload_offset != len(message):
        raise ValueError("unexpected packet payload")
    return decoded


def encode_result(result: FrameResult, processing_ms: float) -> bytes:
    metadata = {
        "v": 1,
        "processingMs": round(processing_ms, 2),
        "frames": [
            {
                "id": result.frame.frame_id,
                "capturedAt": result.frame.captured_at,
                "size": len(result.jpeg),
                "width": result.width,
                "height": result.height,
                "faces": result.faces,
                "timing": result.timing,
            }
        ],
    }
    header = json.dumps(metadata, separators=(",", ":")).encode()
    return b"".join((struct.pack(">I", len(header)), header, result.jpeg))
