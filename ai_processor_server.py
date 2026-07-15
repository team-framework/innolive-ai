import sys
from pathlib import Path

GENERATED_DIR = Path(__file__).resolve().parent / "__generated__"
sys.path.insert(0, str(GENERATED_DIR))

import logging
import time
from concurrent import futures

import grpc
from __generated__ import ai_processor_pb2_grpc
from __generated__.ai_processor_pb2 import WhitelistResponse

class AiProcessorServicer(ai_processor_pb2_grpc.AiProcessorServicer):

    def ProcessVideo(self, request, context):
        return

    def AddWhitelist(self, request, context):
        return WhitelistResponse(
            status_message="테스트 성공",
            timestamp=int(time.time()),
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ai_processor_pb2_grpc.add_AiProcessorServicer_to_server(
        AiProcessorServicer(),
        server,
    )
    listen_addr = "localhost:50051"
    server.add_insecure_port(listen_addr)
    print(f"Starting server on {listen_addr}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()
