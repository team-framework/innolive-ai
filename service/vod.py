"""File-in/file-out VOD anonymization core, reusing the live inference pipeline.

No gRPC imports: ``VodJob`` takes an already-loaded ``RuntimeManager``-like runtime and an
optional AdaFace runtime, decodes the input through an ffmpeg child process, runs the same
detection→tracking→recognition→mosaic pipeline the live ProcessVideo path uses, encodes one
MP4 segment per checkpoint, and muxes the original audio back at DONE. Checkpoints are the
source of truth so a paused job can resume on any worker over shared disk.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from service.detection import is_number_plate_object
from service.frame import MAX_LONG_EDGE
from service.mosaic import (
    DEFAULT_BLUR_RADIUS,
    DEFAULT_MASK_MARGIN,
    DEFAULT_PIXEL_SIZE,
    mosaic_bgr,
    validate_mask_margin,
    validate_mosaic_params,
)
from service.recognition import (
    RecognitionConfig,
    StreamRecognition,
    WhitelistEntry,
    WhitelistSnapshot,
)
from service.runtime import InferenceFailure, TrackingFailure, detector_mask_pad

# Quality modes mirror the proto VodQuality numbering; the core stays gRPC-free.
QUALITY_STANDARD = 0
QUALITY_HIGH = 1
# Anonymization modes mirror FaceAnonymizationMode; v1 accepts BLUR only.
ANONYMIZATION_BLUR = 0

DEFAULT_SEGMENT_FRAMES = 300
DEFAULT_WARMUP_FRAMES = 30
DEFAULT_MAX_INPUT_BYTES = 2_147_483_648  # 2 GiB (MVP)
DEFAULT_MAX_DURATION_SECONDS = 3600.0  # 60 min (MVP)
DEFAULT_ENCODE_QUALITY = 20
CHECKPOINT_VERSION = 1
# Audio codecs MP4 can carry without re-encoding.
COPYABLE_AUDIO_CODECS = frozenset({"aac", "mp3", "opus", "ac3", "eac3", "alac"})
# Codecs we can copy or re-encode to AAC. Apple spatial (apac) and similar are skipped.
REENCODEABLE_AUDIO_CODECS = COPYABLE_AUDIO_CODECS | {
    "pcm_s16le",
    "pcm_s16be",
    "pcm_s24le",
    "pcm_s24be",
    "pcm_f32le",
    "pcm_f32be",
}
_FPS_EWMA_ALPHA = 0.2

# Run states returned by VodJob.run; FAILED surfaces as a raised VodError.
STATE_DONE = "done"
STATE_PAUSED = "paused"

# Pause reasons mirror the proto VodPauseReason numbering.
PAUSE_UNSPECIFIED = 0
PAUSE_REQUESTED = 1
PAUSE_WORKER_DRAINING = 2
PAUSE_RUNTIME_RECOVERY = 3


class VodError(Exception):
    """A job-level failure carrying a §6 error code (surfaced as state=FAILED)."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


class VodPreconditionError(Exception):
    """A request the worker cannot serve (HIGH quality, non-BLUR); maps to FAILED_PRECONDITION."""


@dataclass(frozen=True, slots=True)
class VodLimits:
    vod_root: Path | None = None
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    max_long_edge: int = MAX_LONG_EDGE


@dataclass(frozen=True, slots=True)
class VodOutputOptions:
    video_encoder: str = "auto"
    quality: int = 0  # 0 => default (20)
    force_audio_aac: bool = False


@dataclass(frozen=True, slots=True)
class VodConfig:
    segment_frames: int = DEFAULT_SEGMENT_FRAMES
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    quality: int = QUALITY_STANDARD
    anonymization_mode: int = ANONYMIZATION_BLUR
    blur_radius: float = DEFAULT_BLUR_RADIUS
    pixel_size: int = DEFAULT_PIXEL_SIZE
    mask_margin: int = DEFAULT_MASK_MARGIN
    output: VodOutputOptions = field(default_factory=VodOutputOptions)


@dataclass(frozen=True, slots=True)
class VodWhitelist:
    embedding_model: str
    entries: tuple[tuple[str, np.ndarray], ...]


@dataclass(slots=True)
class VodProbe:
    width: int
    height: int
    avg_frame_rate: float
    duration_seconds: float
    nb_frames: int | None
    has_audio: bool
    audio_codecs: tuple[str, ...]
    size_bytes: int


@dataclass(slots=True)
class VodProgressSnapshot:
    frames_done: int
    frames_total: int
    frames_total_estimated: bool
    fraction: float
    segments_done: int
    segments_total: int
    processing_fps: float
    eta_seconds: float
    checkpoint_token: str


@dataclass(slots=True)
class VodRunResult:
    state: str
    checkpoint_token: str
    output_path: str = ""
    pause_reason: int = PAUSE_UNSPECIFIED


def _normalize_embedding(value: Any) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise VodError("INTERNAL", "whitelist embedding must contain finite values")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 0:
        raise VodError("INTERNAL", "whitelist embedding norm must be positive")
    embedding /= norm
    embedding.setflags(write=False)
    return embedding


def _whitelist_fingerprint(
    embedding_model: str, entries: tuple[tuple[str, np.ndarray], ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(embedding_model.encode("utf-8"))
    for entry_id, embedding in sorted(entries, key=lambda item: item[0]):
        digest.update(entry_id.encode("utf-8"))
        digest.update(np.asarray(embedding, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _params_sha256(config: VodConfig) -> str:
    payload = {
        "quality": config.quality,
        "anonymization_mode": config.anonymization_mode,
        "blur_radius": round(float(config.blur_radius), 6),
        "pixel_size": int(config.pixel_size),
        "mask_margin": int(config.mask_margin),
        "segment_frames": int(config.segment_frames),
        "output": {
            "video_encoder": config.output.video_encoder,
            "quality": int(config.output.quality),
            "force_audio_aac": bool(config.output.force_audio_aac),
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _input_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1 << 20))
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "head_sha256": digest.hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"), sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _checkpoint_token(path: Path, segments_done: int) -> str:
    return f"v1:{segments_done}:{_sha256_file(path)[:16]}"


def read_checkpoint_token(work_dir: Path) -> str:
    """Recompute the token for an on-disk checkpoint (for CLI resume); empty if absent."""

    checkpoint_path = Path(work_dir) / "checkpoint.json"
    if not checkpoint_path.is_file():
        return ""
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        segments = len(data.get("segments", []))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    return _checkpoint_token(checkpoint_path, segments)


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise VodError("INTERNAL", f"{name} is required for VOD processing")
    return binary


def _realpath_under(path: Path, root: Path | None) -> Path:
    resolved = path.resolve()
    if root is None:
        return resolved
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise VodError("PATH_OUTSIDE_ROOT", f"path escapes the VOD root: {path}")
    return resolved


def probe_input(path: Path) -> VodProbe:
    """Run ffprobe and return source facts, raising VodError with a §6 code on failure."""

    ffprobe = _require_binary("ffprobe")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise VodError("INPUT_UNREADABLE", f"ffprobe failed: {completed.stderr.strip()}")
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise VodError("INPUT_UNREADABLE", "ffprobe returned invalid JSON") from error

    streams = parsed.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    duration = _probe_duration(parsed, video_streams)
    if duration is None:
        raise VodError("INPUT_UNREADABLE", "could not determine input duration")

    if len(video_streams) != 1:
        raise VodError(
            "INPUT_UNSUPPORTED", f"expected exactly one video stream, found {len(video_streams)}"
        )
    video = video_streams[0]
    width, height = _displayed_dimensions(video)
    if width < 1 or height < 1:
        raise VodError("INPUT_UNREADABLE", "invalid video dimensions")

    try:
        avg_frame_rate = float(Fraction(video.get("avg_frame_rate", "0")))
    except (ValueError, ZeroDivisionError):
        avg_frame_rate = 0.0
    if not math.isfinite(avg_frame_rate) or avg_frame_rate <= 0:
        try:
            avg_frame_rate = float(Fraction(video.get("r_frame_rate", "0")))
        except (ValueError, ZeroDivisionError):
            avg_frame_rate = 0.0
    if not math.isfinite(avg_frame_rate) or avg_frame_rate <= 0:
        raise VodError("INPUT_UNREADABLE", "could not determine frame rate")

    nb_frames = _probe_nb_frames(video)
    return VodProbe(
        width=width,
        height=height,
        avg_frame_rate=avg_frame_rate,
        duration_seconds=duration,
        nb_frames=nb_frames,
        has_audio=bool(audio_streams),
        audio_codecs=tuple(str(s.get("codec_name", "")) for s in audio_streams),
        size_bytes=path.stat().st_size,
    )


def _probe_duration(parsed: dict[str, Any], video_streams: list[dict[str, Any]]) -> float | None:
    for source in (
        parsed.get("format", {}).get("duration"),
        *(s.get("duration") for s in video_streams),
    ):
        if source in (None, "N/A"):
            continue
        try:
            value = float(source)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _probe_nb_frames(video: dict[str, Any]) -> int | None:
    value = video.get("nb_frames")
    if value in (None, "N/A"):
        return None
    try:
        frames = int(value)
    except (TypeError, ValueError):
        return None
    return frames if frames > 0 else None


def _displayed_dimensions(video: dict[str, Any]) -> tuple[int, int]:
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    rotation = 0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                rotation = round(float(side_data["rotation"]))
            except (TypeError, ValueError):
                rotation = 0
    tags = video.get("tags", {})
    if not rotation and "rotate" in tags:
        try:
            rotation = round(float(tags["rotate"]))
        except (TypeError, ValueError):
            rotation = 0
    if rotation % 180 == 90:
        return height, width
    return width, height


class _FfmpegDecoder:
    """Decode the input to a sequential bgr24 pipe (autorotate applied)."""

    def __init__(self, ffmpeg: str, path: Path, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 3
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self) -> np.ndarray | None:
        assert self._process.stdout is not None
        raw = self._process.stdout.read(self._frame_bytes)
        if not raw:
            return None
        if len(raw) != self._frame_bytes:
            raise VodError("DECODER_FAILED", "decoder returned a partial frame")
        return np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()

    def finish(self) -> None:
        """Wait for a clean EOF; raise DECODER_FAILED on a non-zero exit."""

        if self._process.stdout is not None:
            self._process.stdout.close()
        returncode = self._process.wait()
        stderr = self._read_stderr()
        if returncode not in (0, None):
            _raise_ffmpeg("DECODER_FAILED", returncode, stderr)

    def close(self) -> None:
        _terminate(self._process)

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return ""
        return self._process.stderr.read().decode("utf-8", errors="replace")


class _FfmpegSegmentEncoder:
    """Encode bgr24 frames to one video-only MP4 segment beginning with an IDR frame."""

    def __init__(
        self,
        ffmpeg: str,
        path: Path,
        width: int,
        height: int,
        fps: float,
        config: VodConfig,
        encoder: str,
    ) -> None:
        self.path = path
        self._frame_shape = (height, width, 3)
        quality = config.output.quality or DEFAULT_ENCODE_QUALITY
        if encoder == "h264_nvenc":
            codec_args = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(quality)]
        else:
            codec_args = ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality)]
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                "pipe:0",
                "-an",
                *codec_args,
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "cfr",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self._process.stdin is not None
        if frame.shape != self._frame_shape:
            raise VodError("ENCODER_FAILED", "encoder frame dimensions changed")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as error:
            raise _ffmpeg_error(
                "ENCODER_FAILED", self._process.returncode, self._read_stderr()
            ) from error

    def finish(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        returncode = self._process.wait()
        if returncode not in (0, None):
            _raise_ffmpeg("ENCODER_FAILED", returncode, self._read_stderr())

    def close(self) -> None:
        _terminate(self._process)

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return ""
        return self._process.stderr.read().decode("utf-8", errors="replace")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _ffmpeg_error(code: str, returncode: int | None, stderr: str) -> VodError:
    if "No space left on device" in stderr or "ENOSPC" in stderr:
        return VodError("DISK_FULL", stderr.strip())
    return VodError(code, f"ffmpeg exited {returncode}: {stderr.strip()}")


def _raise_ffmpeg(code: str, returncode: int | None, stderr: str) -> None:
    raise _ffmpeg_error(code, returncode, stderr)


class VodJob:
    """One VOD anonymization job: file in, file out, checkpointed per segment."""

    def __init__(
        self,
        *,
        job_id: str,
        input_path: str,
        output_path: str,
        work_dir: str,
        runtime: Any,
        adaface: Any | None,
        config: VodConfig,
        limits: VodLimits,
        whitelist: VodWhitelist,
        tracker_factory: Callable[..., Any],
        tracker_config: Path,
        device: str,
        recognition_config: RecognitionConfig | None = None,
        worker_target: str = "",
        progress_callback: Callable[[VodProgressSnapshot, bool], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if config.quality != QUALITY_STANDARD:
            raise VodPreconditionError("only VOD_QUALITY_STANDARD is implemented")
        if config.anonymization_mode != ANONYMIZATION_BLUR:
            raise VodPreconditionError("VOD anonymization_mode v1 accepts BLUR only")
        if config.segment_frames < 1:
            raise VodError("INTERNAL", "segment_frames must be at least one")
        validate_mosaic_params(config.blur_radius, config.pixel_size)
        validate_mask_margin(config.mask_margin)

        self.job_id = job_id
        self.input_path = input_path
        self.output_path = output_path
        self.work_dir = Path(work_dir)
        self.runtime = runtime
        self.adaface = adaface
        self.config = config
        self.limits = limits
        self.whitelist = whitelist
        self.tracker_factory = tracker_factory
        self.tracker_config = tracker_config
        self.device = device
        self.recognition_config = recognition_config or RecognitionConfig()
        self.worker_target = worker_target
        self._progress_callback = progress_callback
        self._clock = clock

        self.probe: VodProbe | None = None
        self._input_resolved: Path | None = None
        self._output_resolved: Path | None = None
        self._stop = asyncio.Event()
        self._pause_reason = PAUSE_UNSPECIFIED
        self._whitelist_fingerprint = _whitelist_fingerprint(
            whitelist.embedding_model, whitelist.entries
        )
        self._params_sha = _params_sha256(config)
        self._fps = 0.0
        self._last_frame_time: float | None = None

    def request_pause(self, reason: int = PAUSE_REQUESTED) -> None:
        """Cooperatively ask the running job to pause at the next frame boundary."""

        self._pause_reason = reason
        self._stop.set()

    def _validate_sync(self, *, overwrite_output: bool) -> VodProbe:
        """Resolve paths and probe the input in §6 order; raise VodError on any problem.

        Blocking (stat + ffprobe); run via a thread. Caches the resolved paths and probe so a
        later ``run`` reuses them instead of probing twice.
        """

        _require_binary("ffmpeg")
        _require_binary("ffprobe")
        input_path = _realpath_under(Path(self.input_path), self.limits.vod_root)
        output_path = _realpath_under(Path(self.output_path), self.limits.vod_root)
        _realpath_under(self.work_dir, self.limits.vod_root)

        if not input_path.is_file():
            raise VodError("INPUT_NOT_FOUND", f"input not found: {input_path}")
        if input_path.stat().st_size > self.limits.max_input_bytes:
            raise VodError("INPUT_TOO_LARGE", "input exceeds the configured byte limit")

        probe = probe_input(input_path)
        if probe.duration_seconds > self.limits.max_duration_seconds:
            raise VodError("INPUT_TOO_LONG", "input exceeds the configured duration limit")
        if max(probe.width, probe.height) > self.limits.max_long_edge:
            raise VodError("INPUT_UNSUPPORTED", "input long edge exceeds the supported maximum")

        if output_path.exists() and not overwrite_output:
            raise VodError("OUTPUT_EXISTS", f"output already exists: {output_path}")

        self._input_resolved = input_path
        self._output_resolved = output_path
        self.probe = probe
        return probe

    async def validate(self, *, overwrite_output: bool = False) -> VodProbe:
        """Run the admission probe off the event loop and return the source facts."""

        return await asyncio.to_thread(self._validate_sync, overwrite_output=overwrite_output)

    async def run(
        self, *, resume: bool = False, checkpoint_token: str = "", overwrite_output: bool = False
    ) -> VodRunResult:
        ffmpeg = _require_binary("ffmpeg")
        if self.probe is None:
            await self.validate(overwrite_output=overwrite_output)
        assert self._input_resolved is not None and self._output_resolved is not None
        input_path = self._input_resolved
        output_path = self._output_resolved
        probe = self.probe
        assert probe is not None

        segments_dir = self.work_dir / "segments"
        checkpoint_path = self.work_dir / "checkpoint.json"

        # Paused before the first segment committed: nothing durable exists, so start over.
        if resume and not checkpoint_token and not checkpoint_path.exists():
            resume = False
        if resume:
            state = self._load_checkpoint(
                checkpoint_path, checkpoint_token, input_path, segments_dir
            )
        else:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            if segments_dir.exists():
                shutil.rmtree(segments_dir)
            segments_dir.mkdir(parents=True, exist_ok=True)
            state = _CheckpointState.new(input_path, probe, self.config)
            state.job_id = self.job_id
            state.whitelist_fingerprint = self._whitelist_fingerprint
            state.embedding_model = self.whitelist.embedding_model

        try:
            return await self._process(
                ffmpeg,
                input_path,
                output_path,
                segments_dir,
                checkpoint_path,
                state,
            )
        except VodError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:  # map unexpected failures to a job code.
            raise VodError("INTERNAL", f"unexpected VOD failure: {error}") from error

    # -- checkpoint -------------------------------------------------------

    def _load_checkpoint(
        self, checkpoint_path: Path, token: str, input_path: Path, segments_dir: Path
    ) -> _CheckpointState:
        if not checkpoint_path.is_file():
            raise VodError("CHECKPOINT_INVALID", "checkpoint.json is missing")
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VodError("CHECKPOINT_INVALID", "checkpoint.json is unreadable") from error
        try:
            state = _CheckpointState.from_dict(data)
        except (KeyError, TypeError, ValueError) as error:
            raise VodError("CHECKPOINT_INVALID", "checkpoint.json is malformed") from error

        expected_token = _checkpoint_token(checkpoint_path, len(state.segments))
        if token != expected_token:
            raise VodError("CHECKPOINT_MISMATCH", "checkpoint token does not match")
        current_input = _input_fingerprint(input_path)
        if state.input != current_input:
            raise VodError("CHECKPOINT_MISMATCH", "input fingerprint changed since checkpoint")
        if state.params_sha256 != self._params_sha:
            raise VodError("CHECKPOINT_MISMATCH", "job parameters changed since checkpoint")
        if state.whitelist_fingerprint != self._whitelist_fingerprint:
            raise VodError("CHECKPOINT_MISMATCH", "whitelist changed since checkpoint")
        for segment in state.segments:
            if not (segments_dir.parent / segment["file"]).is_file():
                raise VodError("CHECKPOINT_INVALID", f"segment file is missing: {segment['file']}")
        return state

    def _commit_checkpoint(self, checkpoint_path: Path, state: _CheckpointState) -> str:
        _atomic_write_json(checkpoint_path, state.to_dict())
        return _checkpoint_token(checkpoint_path, len(state.segments))

    # -- processing -------------------------------------------------------

    async def _process(
        self,
        ffmpeg: str,
        input_path: Path,
        output_path: Path,
        segments_dir: Path,
        checkpoint_path: Path,
        state: _CheckpointState,
    ) -> VodRunResult:
        # Runs on the caller's event loop so VOD inference shares the live lane (and loses to
        # live frames). ffmpeg I/O and mosaic composition are offloaded to threads so they never
        # block the loop; detection/tracking/recognition stay on the loop with the shared runtime.
        probe = self.probe
        assert probe is not None
        encoder_name = await asyncio.to_thread(
            _select_encoder, ffmpeg, self.config.output.video_encoder
        )
        recognition = StreamRecognition(self.adaface, self.recognition_config, owner=self.job_id)
        snapshot = self._whitelist_snapshot()
        tracker = await asyncio.to_thread(self.tracker_factory, self.tracker_config, self.device)

        decoder = _FfmpegDecoder(ffmpeg, input_path, probe.width, probe.height)
        encoder: _FfmpegSegmentEncoder | None = None
        committed_token = (
            _checkpoint_token(checkpoint_path, len(state.segments)) if state.segments else ""
        )
        try:
            start_frame = state.next_segment_start_frame
            warmup = min(self.config.warmup_frames, start_frame)
            frame_sequence = 0
            frames_done = start_frame
            self._fps = 0.0
            self._last_frame_time = None

            # Skip frames fully before the warmup window without running the model.
            for _ in range(start_frame - warmup):
                if await asyncio.to_thread(decoder.read) is None:
                    break

            # Re-warm the tracker/recognition over the warmup window without output.
            for _ in range(warmup):
                image = await asyncio.to_thread(decoder.read)
                if image is None:
                    break
                frame_sequence += 1
                await self._run_frame(
                    image, tracker, recognition, snapshot, frame_sequence, encode=False
                )

            segment_index = len(state.segments)
            segment_start = start_frame
            segment_count = 0
            self._emit_progress(frames_done, state, committed_token, committed=False)

            while True:
                if self._stop.is_set():
                    if encoder is not None:
                        encoder.close()
                        _unlink(segments_dir / f"{segment_index:06d}.mp4")
                    await asyncio.to_thread(decoder.close)
                    return VodRunResult(
                        STATE_PAUSED, committed_token, pause_reason=self._pause_reason
                    )
                image = await asyncio.to_thread(decoder.read)
                if image is None:
                    break
                if encoder is None:
                    encoder = _FfmpegSegmentEncoder(
                        ffmpeg,
                        segments_dir / f"{segment_index:06d}.mp4",
                        probe.width,
                        probe.height,
                        probe.avg_frame_rate,
                        self.config,
                        encoder_name,
                    )
                    segment_start = frames_done
                    segment_count = 0
                frame_sequence += 1
                output_frame = await self._run_frame(
                    image, tracker, recognition, snapshot, frame_sequence
                )
                await asyncio.to_thread(encoder.write, output_frame)
                frames_done += 1
                segment_count += 1
                self._update_fps()

                if segment_count >= self.config.segment_frames:
                    committed_token = await asyncio.to_thread(
                        self._finish_segment,
                        encoder,
                        checkpoint_path,
                        state,
                        segment_index,
                        segment_start,
                        segment_count,
                    )
                    encoder = None
                    segment_index += 1
                    self._emit_progress(frames_done, state, committed_token, committed=True)
                elif frames_done % 30 == 0:
                    self._emit_progress(frames_done, state, committed_token, committed=False)

            await asyncio.to_thread(decoder.finish)
            if encoder is not None and segment_count > 0:
                committed_token = await asyncio.to_thread(
                    self._finish_segment,
                    encoder,
                    checkpoint_path,
                    state,
                    segment_index,
                    segment_start,
                    segment_count,
                )
                encoder = None
                self._emit_progress(frames_done, state, committed_token, committed=True)
            elif encoder is not None:
                encoder.close()

            await asyncio.to_thread(
                self._finalize, ffmpeg, input_path, output_path, segments_dir, state
            )
            # Keep checkpoint.json; drop the now-muxed segment slices.
            await asyncio.to_thread(shutil.rmtree, segments_dir, True)
            self._emit_progress(
                state.frames_total or frames_done,
                state,
                committed_token,
                committed=False,
                done=True,
            )
            return VodRunResult(STATE_DONE, committed_token, output_path=str(output_path))
        finally:
            if encoder is not None:
                encoder.close()
            await asyncio.to_thread(decoder.close)
            recognition.close()
            with contextlib.suppress(Exception):
                await asyncio.to_thread(tracker.reset)

    def _finish_segment(
        self,
        encoder: _FfmpegSegmentEncoder,
        checkpoint_path: Path,
        state: _CheckpointState,
        segment_index: int,
        segment_start: int,
        segment_count: int,
    ) -> str:
        encoder.finish()
        relative = f"segments/{segment_index:06d}.mp4"
        state.segments.append(
            {
                "index": segment_index,
                "start_frame": segment_start,
                "frames": segment_count,
                "file": relative,
                "sha256": _sha256_file(encoder.path),
            }
        )
        state.next_segment_start_frame = segment_start + segment_count
        state.frames_done = state.next_segment_start_frame
        state.updated_at_unix_ms = int(time.time() * 1000)
        return self._commit_checkpoint(checkpoint_path, state)

    async def _run_frame(
        self,
        image: np.ndarray,
        tracker: Any,
        recognition: StreamRecognition,
        snapshot: WhitelistSnapshot,
        frame_sequence: int,
        *,
        encode: bool = True,
    ) -> np.ndarray | None:
        result = await self._infer(image, tracker)
        objects = result.get("objects", [])
        # Recognition schedules AdaFace on its own executor but wraps the result into a future on
        # this loop, so it must stay on the loop (not inside to_thread).
        recognition.process(
            image,
            objects,
            snapshot,
            frame_sequence,
            live_track_ids=getattr(tracker, "live_track_ids", None),
        )
        if not encode:
            return None
        if any(
            item.get("whitelisted") is not True or is_number_plate_object(item) for item in objects
        ):
            mask_pad = detector_mask_pad(*image.shape[:2])
            return await asyncio.to_thread(
                mosaic_bgr,
                image,
                objects,
                blur_radius=self.config.blur_radius,
                pixel_size=self.config.pixel_size,
                mask_margin=self.config.mask_margin,
                mask_pad=mask_pad,
            )
        return image

    async def _infer(self, image: np.ndarray, tracker: Any) -> dict[str, Any]:
        """Run one VOD-priority inference on the shared runtime lane (loses to live frames)."""

        try:
            return await self.runtime.infer(image, tracker, live=False)
        except VodError:
            raise
        except (InferenceFailure, TrackingFailure) as error:
            raise VodError("RUNTIME_FAILED", str(error)) from error
        except Exception as error:  # treat inference faults as retryable.
            raise VodError("RUNTIME_FAILED", f"inference failed: {error}") from error

    def _whitelist_snapshot(self) -> WhitelistSnapshot:
        entries = tuple(
            WhitelistEntry(entry_id, _normalize_embedding(values))
            for entry_id, values in self.whitelist.entries
        )
        return WhitelistSnapshot(entries, version=1)

    def _finalize(
        self,
        ffmpeg: str,
        input_path: Path,
        output_path: Path,
        segments_dir: Path,
        state: _CheckpointState,
    ) -> None:
        if not state.segments:
            raise VodError("INTERNAL", "no segments to finalize")
        concat_path = segments_dir / "concat.txt"
        concat_path.write_text(
            "".join(f"file '{Path(segment['file']).name}'\n" for segment in state.segments),
            encoding="utf-8",
        )
        part_path = output_path.with_suffix(output_path.suffix + ".part")
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
        ]
        command.extend(self._audio_args())
        command.extend(
            ["-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", str(part_path)]
        )
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            _unlink(part_path)
            _raise_ffmpeg("MUX_FAILED", completed.returncode, completed.stderr)
        os.replace(part_path, output_path)

    def _audio_args(self) -> list[str]:
        probe = self.probe
        assert probe is not None
        args: list[str] = []
        out_index = 0
        for index, codec in enumerate(probe.audio_codecs):
            codec_name = str(codec or "").lower()
            # Drop Apple spatial/APAC and other tracks ffmpeg cannot remux/re-encode here.
            if codec_name not in REENCODEABLE_AUDIO_CODECS:
                continue
            args.extend(["-map", f"1:a:{index}"])
            if self.config.output.force_audio_aac or codec_name not in COPYABLE_AUDIO_CODECS:
                args.extend([f"-c:a:{out_index}", "aac", f"-b:a:{out_index}", "192k"])
            else:
                args.extend([f"-c:a:{out_index}", "copy"])
            out_index += 1
        return args

    # -- progress ---------------------------------------------------------

    def _update_fps(self) -> None:
        now = self._clock()
        if self._last_frame_time is not None:
            delta = now - self._last_frame_time
            if delta > 0:
                instant = 1.0 / delta
                self._fps = (
                    instant
                    if self._fps <= 0
                    else _FPS_EWMA_ALPHA * instant + (1 - _FPS_EWMA_ALPHA) * self._fps
                )
        self._last_frame_time = now

    def _emit_progress(
        self,
        frames_done: int,
        state: _CheckpointState,
        token: str,
        *,
        committed: bool,
        done: bool = False,
    ) -> None:
        if self._progress_callback is None:
            return
        total = state.frames_total or 0
        fraction = 1.0 if done else (min(1.0, frames_done / total) if total else 0.0)
        eta = (total - frames_done) / self._fps if self._fps > 0 and total else -1.0
        snapshot = VodProgressSnapshot(
            frames_done=frames_done,
            frames_total=total,
            frames_total_estimated=state.frames_total_estimated,
            fraction=fraction,
            segments_done=len(state.segments),
            segments_total=_segments_total(total, self.config.segment_frames),
            processing_fps=round(self._fps, 3),
            eta_seconds=round(eta, 3) if eta >= 0 else -1.0,
            checkpoint_token=token,
        )
        self._progress_callback(snapshot, committed)


def _segments_total(frames_total: int, segment_frames: int) -> int:
    if frames_total <= 0:
        return 0
    return -(-frames_total // segment_frames)


def _unlink(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


_ENCODER_PROBE_CACHE: dict[str, str] = {}


def _select_encoder(ffmpeg: str, requested: str) -> str:
    requested = (requested or "auto").strip().casefold()
    if requested == "libx264":
        return "libx264"
    if requested == "h264_nvenc":
        return "h264_nvenc"
    cached = _ENCODER_PROBE_CACHE.get(ffmpeg)
    if cached is not None:
        return cached
    encoder = "libx264"
    if _nvenc_available(ffmpeg):
        encoder = "h264_nvenc"
    _ENCODER_PROBE_CACHE[ffmpeg] = encoder
    return encoder


def _nvenc_available(ffmpeg: str) -> bool:
    listed = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], check=False, capture_output=True, text=True
    )
    if listed.returncode != 0 or "h264_nvenc" not in listed.stdout:
        return False
    probe = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


@dataclass(slots=True)
class _CheckpointState:
    job_id: str
    input: dict[str, Any]
    params_sha256: str
    whitelist_fingerprint: str
    embedding_model: str
    frames_total: int
    frames_total_estimated: bool
    segment_frames: int
    segments: list[dict[str, Any]]
    next_segment_start_frame: int
    frames_done: int
    updated_at_unix_ms: int

    @classmethod
    def new(cls, input_path: Path, probe: VodProbe, config: VodConfig) -> _CheckpointState:
        estimated = probe.nb_frames is None
        frames_total = probe.nb_frames or round(probe.duration_seconds * probe.avg_frame_rate)
        return cls(
            job_id="",
            input=_input_fingerprint(input_path),
            params_sha256=_params_sha256(config),
            whitelist_fingerprint="",
            embedding_model="",
            frames_total=int(frames_total),
            frames_total_estimated=estimated,
            segment_frames=int(config.segment_frames),
            segments=[],
            next_segment_start_frame=0,
            frames_done=0,
            updated_at_unix_ms=int(time.time() * 1000),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _CheckpointState:
        if int(data["version"]) != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        return cls(
            job_id=str(data["job_id"]),
            input=dict(data["input"]),
            params_sha256=str(data["params_sha256"]),
            whitelist_fingerprint=str(data["whitelist_fingerprint"]),
            embedding_model=str(data.get("embedding_model", "")),
            frames_total=int(data["frames_total"]),
            frames_total_estimated=bool(data["frames_total_estimated"]),
            segment_frames=int(data["segment_frames"]),
            segments=list(data["segments"]),
            next_segment_start_frame=int(data["next_segment_start_frame"]),
            frames_done=int(data["frames_done"]),
            updated_at_unix_ms=int(data["updated_at_unix_ms"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "job_id": self.job_id,
            "input": self.input,
            "params_sha256": self.params_sha256,
            "whitelist_fingerprint": self.whitelist_fingerprint,
            "embedding_model": self.embedding_model,
            "frames_total": self.frames_total,
            "frames_total_estimated": self.frames_total_estimated,
            "segment_frames": self.segment_frames,
            "segments": self.segments,
            "next_segment_start_frame": self.next_segment_start_frame,
            "frames_done": self.frames_done,
            "updated_at_unix_ms": self.updated_at_unix_ms,
        }
