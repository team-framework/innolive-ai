#!/usr/bin/env python3
"""Preview the actual gRPC anonymization pipeline with a camera, video, or image."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from contextlib import aclosing
from pathlib import Path

import cv2
import grpc
import numpy as np
from google.protobuf.json_format import MessageToDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpc_client import VideoFrame, VideoProcessorClient
from service.frame import resize_long_edge


async def run(args: argparse.Namespace) -> int:
    capture = None
    image = None
    stop = False
    processed = 0
    metadata_file = None
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"could not read image: {args.image}")
    else:
        capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        if not capture.isOpened():
            capture.release()
            raise ValueError("could not open camera/video")

    async def frames():
        for frame_id in range(args.frames):
            if stop:
                return
            if capture is not None:
                success, frame = await asyncio.to_thread(capture.read)
                if not success:
                    return
            else:
                frame = image
            frame = resize_long_edge(frame, args.long_edge)
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise ValueError("JPEG encoding failed")
            yield VideoFrame(
                encoded.tobytes(), time.time_ns(), frame_id, anonymization_mode=args.mode
            )

    credentials = (
        grpc.ssl_channel_credentials(args.root_cert.read_bytes()) if args.root_cert else None
    )
    try:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8")
        async with VideoProcessorClient(
            args.target, credentials=credentials, connect_timeout=args.connect_timeout
        ) as client:
            if not await client.is_serving(timeout=args.connect_timeout):
                raise RuntimeError("AiProcessor is not serving")
            async with aclosing(
                client.process_video(
                    frames(), session_id=args.session_id, window=1, timeout=args.timeout
                )
            ) as results:
                async for result in results:
                    processed += 1
                    response = result.response
                    payload = MessageToDict(response, preserving_proto_field_name=True)
                    payload.pop("data", None)
                    payload["requested_mode"] = args.mode
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    if metadata_file:
                        metadata_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        metadata_file.flush()
                        (args.output_dir / "last.jpg").write_bytes(result.processed_jpeg)
                    if not args.headless:
                        preview = cv2.imdecode(
                            np.frombuffer(result.processed_jpeg, np.uint8), cv2.IMREAD_COLOR
                        )
                        if preview is None:
                            raise ValueError("server returned an undecodable JPEG")
                        for face in response.faces:
                            x, y = int(face.bbox.x1), int(face.bbox.y1)
                            cv2.rectangle(
                                preview,
                                (x, y),
                                (int(face.bbox.x2), int(face.bbox.y2)),
                                (0, 220, 0),
                                1,
                            )
                            info = face.anonymization
                            text = f"T{face.track_id} {info.identity_key} {info.attributes.glasses}/{info.attributes.exp} {info.status or args.mode} {info.fallback_reason}"
                            cv2.putText(
                                preview,
                                text,
                                (max(0, x), max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0, 220, 0),
                                1,
                            )
                        cv2.imshow("InnoLive gRPC face presets - q/Esc to stop", preview)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                            stop = True
                            break
    finally:
        if metadata_file:
            metadata_file.close()
        if capture is not None:
            capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if processed == 0:
        raise RuntimeError("input produced no frames")
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int)
    source.add_argument("--video", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--session-id", default=f"preset-test-{uuid.uuid4().hex[:12]}")
    parser.add_argument("--mode", choices=("blur", "face_metadata", "face_swap"), default="blur")
    parser.add_argument(
        "--frames", type=int, default=300, help="maximum frames; image input repeats in one RPC"
    )
    parser.add_argument("--long-edge", type=int, default=640)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, help="save metadata.jsonl and latest processed last.jpg"
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="whole RPC deadline; includes lazy model loading",
    )
    parser.add_argument(
        "--root-cert", type=Path, help="enable TLS with this server root certificate"
    )
    args = parser.parse_args()
    if args.frames < 1 or not 36 <= args.long_edge <= 1920:
        parser.error("frames must be positive and long-edge must be in 36..1920")
    return args


def main() -> None:
    try:
        count = asyncio.run(run(parse_args()))
        print(f"processed {count} frame(s)", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from error


if __name__ == "__main__":
    main()
