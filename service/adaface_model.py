"""Bounded AdaFace IR-18 inference with YuNet five-point alignment."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAFACE_WEIGHTS = ROOT / "models" / "adaface_ir18_casia.ckpt"
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
    weights: Path = DEFAULT_ADAFACE_WEIGHTS
    detector: Path = DEFAULT_FACE_DETECTOR
    device: str = "auto"
    min_face_size: int = 40
    queue_capacity: int = 8

    def __post_init__(self) -> None:
        device = self.device.strip().casefold()
        if not device:
            raise ValueError("AdaFace device must not be empty")
        if self.min_face_size < 16:
            raise ValueError("AdaFace min_face_size must be at least 16")
        if self.queue_capacity < 1:
            raise ValueError("AdaFace queue_capacity must be at least one")
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

        model = AdaFaceIR18()
        checkpoint = torch.load(self.config.weights, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        model_state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        if not model_state:
            model_state = dict(state)
        model.load_state_dict(model_state, strict=True)
        self._model = model.to(self.device).eval()
        self._detector = cv2.FaceDetectorYN.create(
            str(self.config.detector),
            "",
            (320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5_000,
        )

    def submit(self, image: np.ndarray, *, owner: str) -> asyncio.Future[np.ndarray] | None:
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

        concurrent = self._executor.submit(
            self._queued_embedding, image.copy(), time.perf_counter()
        )
        concurrent.add_done_callback(partial(self._completed, owner))
        wrapped = asyncio.wrap_future(concurrent)
        wrapped.add_done_callback(_observe_future)
        return wrapped

    def _queued_embedding(self, image: np.ndarray, submitted_at: float) -> np.ndarray:
        queue_wait_ms = (time.perf_counter() - submitted_at) * 1_000
        with self._lock:
            self._queue_wait_ms_total += queue_wait_ms
            self._queue_wait_ms_max = max(self._queue_wait_ms_max, queue_wait_ms)
        return self._embedding(image)

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

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        aligned = self._aligned_face(image)
        normalized = aligned.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            embedding, _ = self._model(tensor)
        output = embedding[0].detach().to("cpu", dtype=torch.float32).numpy()
        norm = float(np.linalg.norm(output))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("AdaFace returned an invalid embedding")
        return output / norm

    def _aligned_face(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise FaceAlignmentError("face image must be BGR HxWx3")
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        count = 0 if faces is None else len(faces)
        if count != 1:
            raise FaceCountError(f"expected exactly one face, found {count}")

        face = faces[0]
        if min(float(face[2]), float(face[3])) < self.config.min_face_size:
            raise FaceTooSmallError(
                f"face must be at least {self.config.min_face_size}px on its short side"
            )
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

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self.ready,
                "device": str(self.device),
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


class AdaFaceIR18(nn.Module):
    """IR-18 backbone compatible with the official AdaFace checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        blocks = (
            (64, 64, 2),
            (64, 128, 2),
            (128, 256, 2),
            (256, 512, 2),
        )
        body = []
        for input_channels, output_channels, units in blocks:
            body.append(BasicBlockIR(input_channels, output_channels, 2))
            body.extend(BasicBlockIR(output_channels, output_channels, 1) for _ in range(units - 1))
        self.body = nn.Sequential(*body)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.output_layer(self.body(self.input_layer(inputs)))
        norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
        return features / norms, norms


class BasicBlockIR(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        if input_channels == output_channels:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.res_layer(inputs) + self.shortcut_layer(inputs)


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
