"""Development/test CLI for the VOD anonymization core (`python -m service.vod_cli`).

Not used in production (the server drives the worker over gRPC). It loads the same runtime and
AdaFace the worker uses, runs one file-in/file-out job, and streams JSON-line events on stdout
while logging to stderr. SIGTERM/SIGINT pause the job at a frame boundary and exit 75.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from service import vod
from service.adaface_model import (
    DEFAULT_ADAFACE_ARCHITECTURE,
    DEFAULT_ADAFACE_WEIGHTS,
    DEFAULT_FACE_DETECTOR,
    AdaFaceConfig,
    AdaFaceRuntime,
    AdaFaceUnavailable,
)
from service.frame import DEFAULT_FRAME_LIMITS, decode_image
from service.runtime import DEFAULT_CHECKPOINT, DEFAULT_ENGINE, RuntimeConfig, RuntimeManager
from service.tracking import StreamTracker

LOGGER = logging.getLogger("innolive.vod_cli")
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKER = ROOT / "config" / "botsort.yaml"

# Exit codes per interface-A §9.
EXIT_DONE = 0
EXIT_ARGS = 2
EXIT_INPUT = 3
EXIT_CHECKPOINT = 4
EXIT_ENCODE = 5
EXIT_RUNTIME = 6
EXIT_PAUSED = 75
EXIT_OTHER = 1

_EXIT_BY_CODE = {
    "INPUT_NOT_FOUND": EXIT_INPUT,
    "INPUT_UNREADABLE": EXIT_INPUT,
    "INPUT_UNSUPPORTED": EXIT_INPUT,
    "INPUT_TOO_LARGE": EXIT_INPUT,
    "INPUT_TOO_LONG": EXIT_INPUT,
    "PATH_OUTSIDE_ROOT": EXIT_INPUT,
    "CHECKPOINT_INVALID": EXIT_CHECKPOINT,
    "CHECKPOINT_MISMATCH": EXIT_CHECKPOINT,
    "DECODER_FAILED": EXIT_ENCODE,
    "ENCODER_FAILED": EXIT_ENCODE,
    "MUX_FAILED": EXIT_ENCODE,
    "DISK_FULL": EXIT_ENCODE,
    "RUNTIME_FAILED": EXIT_RUNTIME,
}


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _load_whitelist(args: argparse.Namespace, adaface: AdaFaceRuntime | None) -> vod.VodWhitelist:
    if args.whitelist_json:
        data = json.loads(Path(args.whitelist_json).read_text(encoding="utf-8"))
        entries = tuple(
            (entry["entry_id"], np.asarray(entry["values"], dtype=np.float32))
            for entry in data.get("entries", [])
        )
        return vod.VodWhitelist(str(data.get("embedding_model", "")), entries)
    if args.reference_image:
        if adaface is None or not adaface.ready:
            raise SystemExit("AdaFace is required for --reference-image but is unavailable")
        entries: list[tuple[str, np.ndarray]] = []
        for image_path in args.reference_image:
            image = decode_image(Path(image_path).read_bytes(), DEFAULT_FRAME_LIMITS)
            embedding = adaface._embedding(image)  # offline enrollment; mirrors the worker path
            entries.append((Path(image_path).stem, np.asarray(embedding, dtype=np.float32)))
        return vod.VodWhitelist(adaface.embedding_model or "", tuple(entries))
    return vod.VodWhitelist("", ())


def _build_adaface(args: argparse.Namespace, device: str) -> AdaFaceRuntime | None:
    if not (args.whitelist_json or args.reference_image):
        return None
    # Only --reference-image needs AdaFace to be loaded; --whitelist-json carries vectors already.
    if args.whitelist_json and not args.reference_image:
        return None
    try:
        return AdaFaceRuntime(
            AdaFaceConfig(
                architecture=DEFAULT_ADAFACE_ARCHITECTURE,
                weights=DEFAULT_ADAFACE_WEIGHTS,
                detector=DEFAULT_FACE_DETECTOR,
                device=args.device,
            ),
            fallback_device=device,
        )
    except AdaFaceUnavailable as error:
        raise SystemExit(f"AdaFace unavailable: {error}") from error


async def _run(args: argparse.Namespace) -> int:
    cv2.setNumThreads(max(1, (cv2.getNumberOfCPUs() or 2) - 1))
    try:
        runtime = await asyncio.to_thread(
            RuntimeManager,
            RuntimeConfig(
                checkpoint=DEFAULT_CHECKPOINT,
                engine=DEFAULT_ENGINE,
                backend="auto",
                device=args.device,
            ),
        )
    except Exception as error:  # model/runtime load failures are fatal for the CLI.
        _emit({"event": "failed", "error_code": "RUNTIME_FAILED", "error_message": str(error)})
        return EXIT_RUNTIME

    adaface = None
    try:
        adaface = _build_adaface(args, runtime.device)
        whitelist = _load_whitelist(args, adaface)
    except SystemExit as error:
        _emit({"event": "failed", "error_code": "INTERNAL", "error_message": str(error)})
        _close(runtime, adaface)
        return EXIT_OTHER

    config = vod.VodConfig(
        segment_frames=args.segment_frames,
        warmup_frames=args.warmup_frames,
        output=vod.VodOutputOptions(video_encoder=args.encoder, quality=args.quality),
    )
    limits = vod.VodLimits(
        vod_root=Path(args.vod_root) if args.vod_root else None,
        max_input_bytes=args.max_input_bytes,
        max_duration_seconds=args.max_duration_seconds,
    )

    def progress(snapshot: vod.VodProgressSnapshot, committed: bool) -> None:
        _emit(
            {
                "event": "checkpoint" if committed else "progress",
                "frames_done": snapshot.frames_done,
                "frames_total": snapshot.frames_total,
                "fraction": round(snapshot.fraction, 4),
                "segments_done": snapshot.segments_done,
                "processing_fps": snapshot.processing_fps,
                "eta_seconds": snapshot.eta_seconds,
                "checkpoint_token": snapshot.checkpoint_token,
            }
        )

    job = vod.VodJob(
        job_id=args.job_id,
        input_path=args.input,
        output_path=args.output,
        work_dir=args.work_dir,
        runtime=runtime,
        adaface=adaface,
        config=config,
        limits=limits,
        whitelist=whitelist,
        tracker_factory=StreamTracker,
        tracker_config=args.tracker_config,
        device=runtime.device,
        progress_callback=progress,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, job.request_pause, vod.PAUSE_REQUESTED)

    checkpoint_token = vod.read_checkpoint_token(Path(args.work_dir)) if args.resume else ""
    try:
        probe = await job.validate(overwrite_output=args.overwrite)
        _emit(
            {
                "event": "started",
                "frames_total_estimated": probe.nb_frames is None,
                "width": probe.width,
                "height": probe.height,
                "avg_frame_rate": probe.avg_frame_rate,
                "duration_seconds": probe.duration_seconds,
                "has_audio": probe.has_audio,
            }
        )
        result = await job.run(
            resume=args.resume,
            checkpoint_token=checkpoint_token,
            overwrite_output=args.overwrite,
        )
    except vod.VodError as error:
        _emit({"event": "failed", "error_code": error.code, "error_message": error.message})
        return _EXIT_BY_CODE.get(error.code, EXIT_OTHER)
    finally:
        _close(runtime, adaface)

    if result.state == vod.STATE_PAUSED:
        _emit({"event": "paused", "checkpoint_token": result.checkpoint_token})
        return EXIT_PAUSED
    _emit(
        {
            "event": "done",
            "output_path": result.output_path,
            "checkpoint_token": result.checkpoint_token,
        }
    )
    return EXIT_DONE


def _close(runtime: Any, adaface: Any | None) -> None:
    if adaface is not None:
        with contextlib.suppress(Exception):
            adaface.close()
    if runtime is not None:
        with contextlib.suppress(Exception):
            runtime.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InnoLive VOD anonymization CLI")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--job-id", default="cli")
    parser.add_argument("--whitelist-json")
    parser.add_argument("--reference-image", action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--segment-frames", type=int, default=vod.DEFAULT_SEGMENT_FRAMES)
    parser.add_argument("--warmup-frames", type=int, default=vod.DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--encoder", choices=("auto", "h264_nvenc", "libx264"), default="auto")
    parser.add_argument("--quality", type=int, default=0)
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--vod-root", default=None)
    parser.add_argument("--max-input-bytes", type=int, default=vod.DEFAULT_MAX_INPUT_BYTES)
    parser.add_argument(
        "--max-duration-seconds", type=float, default=vod.DEFAULT_MAX_DURATION_SECONDS
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
