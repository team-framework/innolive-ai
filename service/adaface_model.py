"""Bounded AdaFace inference with YuNet five-point face preparation."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from service.adaface_backbones import (
    ADAFACE_ARCHITECTURES,
    build_adaface_backbone,
    checkpoint_backbone_state,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAFACE_ARCHITECTURE = "vit_base_kprpe"
DEFAULT_ADAFACE_WEIGHTS = ROOT / "models" / "adaface_vit_base_kprpe_webface12m.ckpt"
DEFAULT_FACE_DETECTOR = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
_REFERENCE_LANDMARKS = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
_ENROLLMENT_SCORE_THRESHOLD = 0.9
_QUERY_SCORE_THRESHOLD = 0.6
_FLIPPED_LANDMARK_ORDER = (1, 0, 2, 4, 3)
_VIT_FACE_MARGIN = 0.25


class AdaFaceUnavailable(RuntimeError):
    """The configured recognition model could not be loaded."""


class FaceCountError(ValueError):
    """Enrollment or recognition input did not contain exactly one face."""


class FaceTooSmallError(ValueError):
    """The detected face is below the configured quality floor."""


class FaceAlignmentError(ValueError):
    """Five-point face alignment could not produce a valid crop."""


@dataclass(frozen=True, slots=True)
class AdaFaceConfig:
    architecture: str = DEFAULT_ADAFACE_ARCHITECTURE
    weights: Path = DEFAULT_ADAFACE_WEIGHTS
    detector: Path = DEFAULT_FACE_DETECTOR
    device: str = "auto"
    min_face_size: int = 40
    query_min_face_size: int = 24
    queue_capacity: int = 8
    warmup_runs: int = 1

    def __post_init__(self) -> None:
        architecture = self.architecture.strip().casefold()
        device = self.device.strip().casefold()
        if architecture not in ADAFACE_ARCHITECTURES:
            choices = ", ".join(ADAFACE_ARCHITECTURES)
            raise ValueError(f"AdaFace architecture must be one of: {choices}")
        if not device:
            raise ValueError("AdaFace device must not be empty")
        if self.min_face_size < 16:
            raise ValueError("AdaFace min_face_size must be at least 16")
        if self.query_min_face_size < 16:
            raise ValueError("AdaFace query_min_face_size must be at least 16")
        if self.queue_capacity < 1:
            raise ValueError("AdaFace queue_capacity must be at least one")
        if self.warmup_runs < 0:
            raise ValueError("AdaFace warmup_runs must not be negative")
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "weights", self.weights.expanduser().resolve())
        object.__setattr__(self, "detector", self.detector.expanduser().resolve())
        object.__setattr__(self, "device", device)


class AdaFaceRuntime:
    """Own one AdaFace model, one YuNet detector, and one bounded worker."""

    def __init__(self, config: AdaFaceConfig, *, fallback_device: str):
        self.config = config
        self.device = torch.device("cpu")
        self.ready = False
        self.load_error: str | None = None
        self.load_ms: float | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adaface")
        self._lock = threading.Lock()
        self._inflight = 0
        self._inflight_by_owner: dict[str, int] = {}
        self._owner_capacity = max(1, config.queue_capacity // 2)
        self._max_inflight = 0
        self._calls = 0
        self._overflow = 0
        self._failures = 0
        self._queue_wait_ms_total = 0.0
        self._queue_wait_ms_max = 0.0
        started = time.perf_counter()
        try:
            self.device = _resolve_device(config.device, fallback_device)
            self._executor.submit(self._load).result()
            self.load_ms = (time.perf_counter() - started) * 1_000
            self.ready = True
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"

    def _load(self) -> None:
        if not self.config.weights.is_file():
            raise AdaFaceUnavailable(f"AdaFace weights not found: {self.config.weights}")
        if not self.config.detector.is_file():
            raise AdaFaceUnavailable(f"YuNet detector not found: {self.config.detector}")

        model = build_adaface_backbone(self.config.architecture)
        checkpoint = _load_checkpoint(self.config.weights)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise AdaFaceUnavailable("AdaFace checkpoint does not contain a state dictionary")
        model_state = checkpoint_backbone_state(self.config.architecture, state)
        if not model_state:
            raise AdaFaceUnavailable(
                f"AdaFace checkpoint does not match {self.config.architecture}"
            )
        model.load_state_dict(model_state, strict=True)
        self._model = model.to(self.device).eval()
        self._detector = cv2.FaceDetectorYN.create(
            str(self.config.detector),
            "",
            (320, 320),
            score_threshold=_QUERY_SCORE_THRESHOLD,
            nms_threshold=0.3,
            top_k=5_000,
        )
        self._warmup()

    def _warmup(self) -> None:
        if self.config.warmup_runs == 0:
            return
        inputs = torch.zeros((2, 3, 112, 112), dtype=torch.float32, device=self.device)
        keypoints = torch.from_numpy((_REFERENCE_LANDMARKS / 112.0).copy())
        keypoints = keypoints.unsqueeze(0).repeat(2, 1, 1).to(self.device)
        with torch.inference_mode():
            for _ in range(self.config.warmup_runs):
                self._model(inputs, keypoints)
        _synchronize(self.device)

    def submit(self, image: np.ndarray, *, owner: str) -> asyncio.Future[np.ndarray] | None:
        return self._submit(self._queued_embedding, image, owner)

    def submit_enrollment(
        self,
        image: np.ndarray,
        *,
        owner: str,
    ) -> asyncio.Future[np.ndarray] | None:
        return self._submit(self._queued_enrollment_embedding, image, owner)

    def _submit(
        self,
        worker: Callable[[np.ndarray, float], np.ndarray],
        image: np.ndarray,
        owner: str,
    ) -> asyncio.Future[np.ndarray] | None:
        if not self.ready:
            return None
        with self._lock:
            owner_inflight = self._inflight_by_owner.get(owner, 0)
            if (
                self._inflight >= self.config.queue_capacity
                or owner_inflight >= self._owner_capacity
            ):
                self._overflow += 1
                return None
            self._inflight += 1
            self._inflight_by_owner[owner] = owner_inflight + 1
            self._calls += 1
            self._max_inflight = max(self._max_inflight, self._inflight)

        concurrent = self._executor.submit(worker, image.copy(), time.perf_counter())
        concurrent.add_done_callback(partial(self._completed, owner))
        wrapped = asyncio.wrap_future(concurrent)
        wrapped.add_done_callback(_observe_future)
        return wrapped

    def _queued_embedding(self, image: np.ndarray, submitted_at: float) -> np.ndarray:
        self._record_queue_wait(submitted_at)
        return self._embedding(
            image,
            score_threshold=_QUERY_SCORE_THRESHOLD,
            min_face_size=min(self.config.min_face_size, self.config.query_min_face_size),
        )

    def _queued_enrollment_embedding(
        self,
        image: np.ndarray,
        submitted_at: float,
    ) -> np.ndarray:
        self._record_queue_wait(submitted_at)
        return self._embedding(
            image,
            score_threshold=_ENROLLMENT_SCORE_THRESHOLD,
            min_face_size=self.config.min_face_size,
        )

    def _record_queue_wait(self, submitted_at: float) -> None:
        queue_wait_ms = (time.perf_counter() - submitted_at) * 1_000
        with self._lock:
            self._queue_wait_ms_total += queue_wait_ms
            self._queue_wait_ms_max = max(self._queue_wait_ms_max, queue_wait_ms)

    def _completed(self, owner: str, future: Future[np.ndarray]) -> None:
        failed = future.cancelled()
        if not failed:
            try:
                failed = future.exception() is not None
            except BaseException:
                failed = True
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            owner_inflight = self._inflight_by_owner.get(owner, 0) - 1
            if owner_inflight > 0:
                self._inflight_by_owner[owner] = owner_inflight
            else:
                self._inflight_by_owner.pop(owner, None)
            if failed:
                self._failures += 1

    def _embedding(
        self,
        image: np.ndarray,
        *,
        score_threshold: float = _ENROLLMENT_SCORE_THRESHOLD,
        min_face_size: int | None = None,
    ) -> np.ndarray:
        prepared, keypoints = self._prepared_face(
            image,
            score_threshold=score_threshold,
            min_face_size=self.config.min_face_size if min_face_size is None else min_face_size,
        )
        if self.config.architecture == "vit_base_kprpe":
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        normalized = prepared.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
        tensor = tensor.to(self.device)
        inputs = torch.cat((tensor, torch.flip(tensor, dims=(3,))), dim=0)
        keypoint_inputs = None
        if keypoints is not None:
            mirrored = keypoints[np.asarray(_FLIPPED_LANDMARK_ORDER)].copy()
            mirrored[:, 0] = 1.0 - mirrored[:, 0]
            keypoint_inputs = torch.from_numpy(np.stack((keypoints, mirrored)))
            keypoint_inputs = keypoint_inputs.to(self.device)
        with torch.inference_mode():
            embeddings, norms = self._model(inputs, keypoint_inputs)
        fused = (embeddings * norms).sum(dim=0)
        output = fused.detach().to("cpu", dtype=torch.float32).numpy()
        norm = float(np.linalg.norm(output))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("AdaFace returned an invalid embedding")
        return output / norm

    def _prepared_face(
        self,
        image: np.ndarray,
        *,
        score_threshold: float,
        min_face_size: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        face = self._detected_face(
            image,
            score_threshold=score_threshold,
            min_face_size=min_face_size,
        )
        if self.config.architecture == "vit_base_kprpe":
            return _square_face_crop(image, face)
        return _align_face(image, face), None

    def _aligned_face(
        self,
        image: np.ndarray,
        *,
        score_threshold: float = _ENROLLMENT_SCORE_THRESHOLD,
        min_face_size: int | None = None,
    ) -> np.ndarray:
        face = self._detected_face(
            image,
            score_threshold=score_threshold,
            min_face_size=self.config.min_face_size if min_face_size is None else min_face_size,
        )
        return _align_face(image, face)

    def _detected_face(
        self,
        image: np.ndarray,
        *,
        score_threshold: float,
        min_face_size: int,
    ) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise FaceAlignmentError("face image must be BGR HxWx3")
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is not None:
            faces = faces[faces[:, 14] >= score_threshold]
        count = 0 if faces is None else len(faces)
        if count != 1:
            raise FaceCountError(f"expected exactly one face, found {count}")

        face = faces[0]
        if min(float(face[2]), float(face[3])) < min_face_size:
            raise FaceTooSmallError(f"face must be at least {min_face_size}px on its short side")
        return face

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self.ready,
                "device": str(self.device),
                "architecture": self.config.architecture,
                "weights": self.config.weights.name,
                "detector": self.config.detector.name,
                "load_ms": round(self.load_ms, 2) if self.load_ms is not None else None,
                "load_error": self.load_error,
                "inflight": self._inflight,
                "queue_capacity": self.config.queue_capacity,
                "owner_capacity": self._owner_capacity,
                "max_inflight": self._max_inflight,
                "calls": self._calls,
                "queue_overflow": self._overflow,
                "queue_wait_ms_total": round(self._queue_wait_ms_total, 2),
                "queue_wait_ms_max": round(self._queue_wait_ms_max, 2),
                "failures": self._failures,
            }

    def close(self) -> None:
        self.ready = False
        self._executor.shutdown(wait=True, cancel_futures=True)
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_detector"):
            del self._detector


def _resolve_device(requested: str, fallback: str) -> torch.device:
    value = fallback if requested == "auto" else requested
    if value.isdigit():
        value = f"cuda:{value}"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise AdaFaceUnavailable(f"AdaFace CUDA device is unavailable: {value}")
    if value == "mps" and not torch.backends.mps.is_available():
        raise AdaFaceUnavailable("AdaFace MPS device is unavailable")
    try:
        return torch.device(value)
    except (RuntimeError, ValueError) as error:
        raise AdaFaceUnavailable(f"invalid AdaFace device: {value}") from error


def _observe_future(future: asyncio.Future[np.ndarray]) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except BaseException:
        return


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def _align_face(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks,
        _REFERENCE_LANDMARKS,
        method=cv2.LMEDS,
    )
    if transform is None or not np.isfinite(transform).all():
        raise FaceAlignmentError("five-point alignment transform failed")
    aligned = cv2.warpAffine(
        image,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if aligned.shape != (112, 112, 3):
        raise FaceAlignmentError("aligned face has an invalid shape")
    return aligned


def _square_face_crop(image: np.ndarray, face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = (float(value) for value in face[:4])
    side = max(width, height) * (1.0 + _VIT_FACE_MARGIN * 2.0)
    if not math.isfinite(side) or side <= 0:
        raise FaceAlignmentError("face crop has invalid dimensions")
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    scale = 112.0 / side
    transform = np.asarray(
        [
            [scale, 0.0, 56.0 - center_x * scale],
            [0.0, scale, 56.0 - center_y * scale],
        ],
        dtype=np.float32,
    )
    cropped = cv2.warpAffine(
        image,
        transform,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    landmarks = cv2.transform(landmarks[None], transform)[0] / 112.0
    if cropped.shape != (112, 112, 3) or not np.isfinite(landmarks).all():
        raise FaceAlignmentError("ViT face preparation failed")
    return cropped, np.clip(landmarks, 0.0, 1.0).astype(np.float32)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
