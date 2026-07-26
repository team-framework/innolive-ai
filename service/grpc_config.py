"""Bounded gRPC transport settings for the B1 ProcessVideo service."""

from __future__ import annotations

import os

from service.protocol import MAX_JPEG_BYTES, MAX_RESPONSE_BYTES

PROTOBUF_OVERHEAD_BYTES = 64 * 1024


def server_options() -> tuple[tuple[str, int], ...]:
    minimum_ping_ms = _positive_env("GRPC_MIN_RECV_PING_MS", 30_000)
    return (
        ("grpc.max_receive_message_length", MAX_JPEG_BYTES + PROTOBUF_OVERHEAD_BYTES),
        ("grpc.max_send_message_length", MAX_RESPONSE_BYTES + PROTOBUF_OVERHEAD_BYTES),
        ("grpc.so_reuseport", 0),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.min_ping_interval_without_data_ms", minimum_ping_ms),
        ("grpc.http2.max_ping_strikes", 2),
    )


def channel_options() -> tuple[tuple[str, int], ...]:
    minimum_ping_ms = _positive_env("GRPC_MIN_RECV_PING_MS", 30_000)
    keepalive_ms = _positive_env("GRPC_KEEPALIVE_MS", 60_000)
    if keepalive_ms < minimum_ping_ms:
        raise ValueError("GRPC_KEEPALIVE_MS must be at least GRPC_MIN_RECV_PING_MS")
    return (
        ("grpc.max_send_message_length", MAX_JPEG_BYTES + PROTOBUF_OVERHEAD_BYTES),
        (
            "grpc.max_receive_message_length",
            MAX_RESPONSE_BYTES + PROTOBUF_OVERHEAD_BYTES,
        ),
        ("grpc.keepalive_time_ms", keepalive_ms),
        ("grpc.keepalive_timeout_ms", 20_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    )


def listen_address(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"


def _positive_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
