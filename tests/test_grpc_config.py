from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from service.grpc_config import channel_options, listen_address, server_options
from service.protocol import MAX_JPEG_BYTES, MAX_RESPONSE_BYTES


class GrpcConfigTests(unittest.TestCase):
    def test_transport_limits_keepalive_and_exclusive_bind_are_fixed(self):
        server = dict(server_options())
        channel = dict(channel_options())

        self.assertGreater(server["grpc.max_receive_message_length"], MAX_JPEG_BYTES)
        self.assertGreater(channel["grpc.max_send_message_length"], MAX_JPEG_BYTES)
        self.assertGreater(server["grpc.max_send_message_length"], MAX_RESPONSE_BYTES)
        self.assertGreater(
            channel["grpc.max_receive_message_length"],
            MAX_RESPONSE_BYTES,
        )
        self.assertEqual(server["grpc.so_reuseport"], 0)
        self.assertGreaterEqual(
            channel["grpc.keepalive_time_ms"],
            server["grpc.http2.min_ping_interval_without_data_ms"],
        )

    def test_invalid_keepalive_environment_fails_at_startup(self):
        for value in ("0", "-1", "not-an-integer"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"GRPC_KEEPALIVE_MS": value}),
                self.assertRaises(ValueError),
            ):
                channel_options()

        with (
            patch.dict(
                os.environ,
                {
                    "GRPC_KEEPALIVE_MS": "1000",
                    "GRPC_MIN_RECV_PING_MS": "2000",
                },
            ),
            self.assertRaises(ValueError),
        ):
            channel_options()

    def test_ipv4_and_ipv6_listen_addresses_are_unambiguous(self):
        self.assertEqual(listen_address("127.0.0.1", 50051), "127.0.0.1:50051")
        self.assertEqual(listen_address("::1", 50051), "[::1]:50051")


if __name__ == "__main__":
    unittest.main()
