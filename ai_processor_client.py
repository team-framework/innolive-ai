import sys
from pathlib import Path

import numpy as np

GENERATED_DIR = Path(__file__).resolve().parent / "__generated__"
sys.path.insert(0, str(GENERATED_DIR))

import time
import cv2

import grpc

from __generated__ import ai_processor_pb2_grpc
from __generated__.ai_processor_pb2 import VideoChunk

def make_image_generator():
    images = ['yongin.jpg', 'dh.jpg', 'jb.jpg']

    for file_path in images:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
        _, buffer = cv2.imencode('.jpg', image)
        yield VideoChunk(data=buffer.tobytes(), timestamp=int(time.time()))

        time.sleep(1)

def transform_to_grayscale(stub: ai_processor_pb2_grpc.AiProcessorStub):
    processed_images = stub.ProcessVideo(make_image_generator())
    for processed_image in processed_images:
        image = cv2.imdecode(
            np.frombuffer(processed_image.data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        cv2.imshow('Processed Image', image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

with grpc.insecure_channel("localhost:50051") as channel:
    stub = ai_processor_pb2_grpc.AiProcessorStub(channel)
    transform_to_grayscale(stub)