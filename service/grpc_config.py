from __future__ import annotations

import os


def _message_options() -> tuple[tuple[str, int], ...]:
    max_message_bytes = int(os.getenv("GRPC_MAX_MESSAGE_MB", "16")) * 1024 * 1024
    return (
        ("grpc.max_receive_message_length", max_message_bytes),
        ("grpc.max_send_message_length", max_message_bytes),
    )


def channel_options() -> tuple[tuple[str, int], ...]:
    return _message_options() + (
        ("grpc.keepalive_time_ms", int(os.getenv("GRPC_KEEPALIVE_MS", "60000"))),
        ("grpc.keepalive_timeout_ms", 20_000),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", 0),
    )


def server_options() -> tuple[tuple[str, int], ...]:
    return _message_options() + (
        ("grpc.keepalive_permit_without_calls", 1),
        (
            "grpc.http2.min_ping_interval_without_data_ms",
            int(os.getenv("GRPC_MIN_RECV_PING_MS", "30000")),
        ),
        ("grpc.http2.max_ping_strikes", 2),
    )


def listen_address(host: str, port: int) -> str:
    return (
        f"[{host}]:{port}"
        if ":" in host and not host.startswith("[")
        else f"{host}:{port}"
    )
