import unittest

from service.grpc_config import channel_options, server_options


class GrpcConfigTest(unittest.TestCase):
    def test_keepalive_policies_are_compatible(self):
        channel = dict(channel_options())
        server = dict(server_options())

        self.assertGreaterEqual(
            channel["grpc.keepalive_time_ms"],
            server["grpc.http2.min_ping_interval_without_data_ms"],
        )
        self.assertEqual(channel["grpc.http2.max_pings_without_data"], 0)


if __name__ == "__main__":
    unittest.main()
