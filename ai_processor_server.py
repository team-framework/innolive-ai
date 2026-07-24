import argparse
import logging
import os
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Callable, Iterator

import grpc

from service.transform_image import to_grayscale


GENERATED_DIR = Path(__file__).resolve().parent / "__generated__"
sys.path.insert(0, str(GENERATED_DIR))

from __generated__ import ai_processor_pb2_grpc  # noqa: E402
from __generated__.ai_processor_pb2 import (  # noqa: E402
    ProcessedVideoChunk,
    VideoChunk,
    WhitelistResponse,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best.pt"
DEFAULT_TRACKER_CONFIG_PATH = ROOT_DIR / "config" / "botsort.yaml"


def passthrough(data: bytes) -> bytes:
    return data


class AiProcessorServicer(ai_processor_pb2_grpc.AiProcessorServicer):
    def __init__(
        self,
        image_processor: Callable[[bytes], bytes] = to_grayscale,
        image_processor_factory: Callable[[], Callable[[bytes], bytes]] | None = None,
    ):
        self._image_processor_factory = image_processor_factory or (
            lambda: image_processor
        )

    def ProcessVideo(self, request_iterator: Iterator[VideoChunk], context):
        # A factory gives every bidirectional stream its own BoT-SORT state while
        # the expensive detector/TensorRT engine remains shared by the server.
        image_processor = self._image_processor_factory()
        for request in request_iterator:
            try:
                processed_image = image_processor(request.data)
                yield ProcessedVideoChunk(
                    data=processed_image,
                    status_message="success",
                    timestamp=request.timestamp,
                )
            except Exception as e:
                print(f"error occurred while processing video: {e}")
                yield ProcessedVideoChunk(
                    data=b"",
                    status_message="failed",
                    timestamp=request.timestamp,
                )

    def AddWhitelist(self, request, context):
        return WhitelistResponse(
            status_message="테스트 성공",
            timestamp=int(time.time()),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the InnoLive AI gRPC server")
    parser.add_argument("--host", default=os.getenv("GRPC_HOST", "localhost"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRPC_PORT", "50051")),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("GRPC_MAX_WORKERS", "10")),
    )
    parser.add_argument(
        "--processing-mode",
        choices=("anonymize", "grayscale", "passthrough"),
        default=os.getenv("GRPC_PROCESSING_MODE", "anonymize"),
        help="Use anonymize in production; passthrough isolates gRPC performance",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("AI_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="Path to best.pt or best.engine (a fresh sibling engine is preferred)",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path(os.getenv("AI_TRACKER_CONFIG", str(DEFAULT_TRACKER_CONFIG_PATH))),
    )
    parser.add_argument(
        "--device",
        default=os.getenv("AI_DEVICE") or None,
        help="Ultralytics device, for example 0, cuda:0, cpu, or mps",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.getenv("AI_IMAGE_SIZE", "640")),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=float(os.getenv("AI_CONFIDENCE", "0.10")),
        help="Must be low enough to retain BoT-SORT second-stage detections",
    )
    parser.add_argument(
        "--mask-hold-frames",
        type=int,
        default=int(os.getenv("AI_MASK_HOLD_FRAMES", "8")),
        help="Render BoT-SORT-predicted masks across short detection gaps",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=int(os.getenv("AI_JPEG_QUALITY", "90")),
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        default=os.getenv("AI_PREFER_TENSORRT", "1").lower() in {"0", "false", "no"},
        help="Do not prefer a fresh best.engine next to the requested .pt model",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=os.getenv("AI_WARMUP", "1").lower() in {"0", "false", "no"},
        help="Skip startup inference warmup",
    )
    return parser.parse_args()


def serve(
    host: str = "localhost",
    port: int = 50051,
    max_workers: int = 10,
    processing_mode: str = "anonymize",
    model_path: Path = DEFAULT_MODEL_PATH,
    tracker_config_path: Path = DEFAULT_TRACKER_CONFIG_PATH,
    device: str | None = None,
    imgsz: int = 640,
    confidence: float = 0.10,
    mask_hold_frames: int = 8,
    jpeg_quality: int = 90,
    prefer_tensorrt: bool = True,
    warmup: bool = True,
):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    processor_factory = None
    backend_description = processing_mode
    if processing_mode == "anonymize":
        # Keep heavy optional imports out of transport-only benchmark startup.
        from service.anonymizer import AnonymizerConfig, HeadAnonymizerRuntime

        runtime = HeadAnonymizerRuntime(
            AnonymizerConfig(
                model_path=model_path,
                tracker_config_path=tracker_config_path,
                device=device,
                imgsz=imgsz,
                confidence=confidence,
                mask_hold_frames=mask_hold_frames,
                jpeg_quality=jpeg_quality,
                prefer_tensorrt=prefer_tensorrt,
                warmup=warmup,
            )
        )

        def processor_factory():
            return runtime.create_stream().process_bytes

        image_processor = passthrough  # unused when processor_factory is supplied
        backend_description = (
            f"anonymize/{'TensorRT' if runtime.using_tensorrt else 'PyTorch'}"
        )
    else:
        image_processor = (
            to_grayscale if processing_mode == "grayscale" else passthrough
        )

    ai_processor_pb2_grpc.add_AiProcessorServicer_to_server(
        AiProcessorServicer(
            image_processor=image_processor,
            image_processor_factory=processor_factory,
        ),
        server,
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    print(
        f"Starting server on {listen_addr} with {max_workers} workers "
        f"(processing_mode={backend_description})",
        flush=True,
    )
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    args = parse_args()
    serve(
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        processing_mode=args.processing_mode,
        model_path=args.model,
        tracker_config_path=args.tracker_config,
        device=args.device,
        imgsz=args.imgsz,
        confidence=args.confidence,
        mask_hold_frames=args.mask_hold_frames,
        jpeg_quality=args.jpeg_quality,
        prefer_tensorrt=not args.no_tensorrt,
        warmup=not args.no_warmup,
    )
