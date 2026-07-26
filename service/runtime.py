"""Platform-aware YOLO runtime with one serialized inference lane."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import platform
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from service.tracking import (
    ACTIVATION_CONFIDENCE,
    CONTINUATION_CONFIDENCE,
    DETECTOR_CONFIDENCE,
    HOLD_CONFIDENCE_DECAY,
    MAX_MASK_HOLD_FRAMES,
)

if TYPE_CHECKING:
    from service.tracking import StreamTracker


IMAGE_SIZE = 640
MAX_DETECTIONS = 100
MAX_POLYGON_POINTS = 64
EXPECTED_CLASS_NAMES = {0: "face"}
BACKENDS = frozenset({"auto", "tensorrt", "pytorch"})
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "best.pt"
DEFAULT_ENGINE = ROOT / "models" / "best_b1.engine"


class InferenceFailure(RuntimeError):
    """The selected model backend could not process a frame."""


class TrackingFailure(RuntimeError):
    """Tracking or temporal mask stabilization failed."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    engine: Path = DEFAULT_ENGINE
    backend: str = "auto"
    device: str = "auto"
    warmup_runs: int = 3

    def __post_init__(self) -> None:
        backend = self.backend.strip().casefold()
        device = self.device.strip().casefold()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
        if not device:
            raise ValueError("device must not be empty")
        if self.warmup_runs < 1:
            raise ValueError("warmup_runs must be at least one")
        object.__setattr__(self, "checkpoint", self.checkpoint.expanduser().resolve())
        object.__setattr__(self, "engine", self.engine.expanduser().resolve())
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "device", device)


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    backend: str
    artifact: Path
    device: str
    manifest: dict[str, Any] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_engine(config: RuntimeConfig) -> dict[str, Any]:
    engine = config.engine
    if engine.suffix.lower() != ".engine" or not engine.is_file():
        raise RuntimeError(f"static TensorRT engine not found: {engine}")
    manifest_path = engine.with_suffix(engine.suffix + ".json")
    if not manifest_path.is_file():
        raise RuntimeError(f"TensorRT manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid TensorRT manifest: {manifest_path}") from error

    required = {
        "schema_version": 1,
        "standard_profile": "B1-640-Q90-W5",
        "precision": "fp16",
        "dynamic": False,
        "batch": 1,
        "image_size": IMAGE_SIZE,
        "class_names": {"0": "face"},
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"engine manifest {key}={manifest.get(key)!r}; expected {expected!r}"
            )
    expected_hash = manifest.get("engine_sha256")
    if not isinstance(expected_hash, str) or sha256_file(engine) != expected_hash:
        raise RuntimeError(f"TensorRT engine hash does not match manifest: {engine}")
    source_name = manifest.get("source_checkpoint")
    source_hash = manifest.get("source_sha256")
    if not isinstance(source_name, str) or not isinstance(source_hash, str):
        raise RuntimeError("engine manifest is missing checkpoint provenance")
    source = engine.parent / source_name
    if not source.is_file() or sha256_file(source) != source_hash:
        raise RuntimeError(f"checkpoint hash does not match manifest: {source}")
    return manifest


def select_runtime(config: RuntimeConfig) -> RuntimeSelection:
    """Resolve one explicit artifact and accelerator from the configured policy."""

    if config.backend == "tensorrt":
        return _select_tensorrt(config)
    if config.backend == "pytorch":
        return _select_pytorch(config)

    if _tensorrt_supported() and config.engine.is_file():
        return _select_tensorrt(config)
    return _select_pytorch(config)


def _select_tensorrt(config: RuntimeConfig) -> RuntimeSelection:
    if not _tensorrt_supported():
        raise RuntimeError(
            "TensorRT requires Linux x86_64, an NVIDIA CUDA device, and the tensorrt package"
        )
    device = _resolve_device(config.device, backend="tensorrt")
    return RuntimeSelection(
        backend="tensorrt",
        artifact=config.engine,
        device=device,
        manifest=validate_engine(config),
    )


def _select_pytorch(config: RuntimeConfig) -> RuntimeSelection:
    if not config.checkpoint.is_file() or config.checkpoint.suffix.lower() != ".pt":
        raise RuntimeError(f"PyTorch checkpoint not found: {config.checkpoint}")
    return RuntimeSelection(
        backend="pytorch",
        artifact=config.checkpoint,
        device=_resolve_device(config.device, backend="pytorch"),
    )


def _resolve_device(requested: str, *, backend: str) -> str:
    import torch

    if backend == "tensorrt":
        device = "0" if requested == "auto" else requested.removeprefix("cuda:")
        if not device.isdigit() or int(device) >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device is unavailable: {requested}")
        return device

    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if requested.startswith("cuda:"):
        requested = requested.removeprefix("cuda:")
    if requested.isdigit() and int(requested) >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is unavailable: {requested}")
    if requested not in {"cpu", "mps"} and not requested.isdigit():
        raise ValueError("device must be auto, cpu, mps, or a CUDA device index")
    return requested


def _tensorrt_supported() -> bool:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        return False
    if importlib.util.find_spec("tensorrt") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class RuntimeManager:
    """Own exactly one warmed model and one bounded execution thread."""

    _guard = threading.Lock()
    _live_instances = 0

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.selection = select_runtime(config)
        self.manifest = self.selection.manifest
        if self.selection.backend == "tensorrt":
            import tensorrt

            expected_version = self.manifest.get("tensorrt") if self.manifest else None
            if tensorrt.__version__ != expected_version:
                raise RuntimeError(
                    "TensorRT runtime version does not match the engine manifest: "
                    f"{tensorrt.__version__} != {expected_version}"
                )
        with self._guard:
            if type(self)._live_instances:
                raise RuntimeError("only one RuntimeManager may exist in a process")
            type(self)._live_instances += 1

        self.backend = self.selection.backend
        self.device = self.selection.device
        self.artifact_sha256 = sha256_file(self.selection.artifact)
        self.ready = False
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="b1-inference")
        self._lane: asyncio.Lock | None = None
        self._latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=300))
        self.frames = 0
        self.warmup_ms: float | None = None
        try:
            from ultralytics import YOLO

            started = time.perf_counter()
            self._model = YOLO(str(self.selection.artifact), task="segment")
            self.names = {int(key): str(value) for key, value in self._model.names.items()}
            if self.names != EXPECTED_CLASS_NAMES:
                raise RuntimeError(
                    f"model class names {self.names!r}; expected {EXPECTED_CLASS_NAMES!r}"
                )
            dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            for _ in range(self.config.warmup_runs):
                self._predict(dummy)
            self.warmup_ms = (time.perf_counter() - started) * 1000
            self.ready = True
        except Exception:
            self.close()
            raise

    @property
    def lane(self) -> asyncio.Lock:
        if self._lane is None:
            self._lane = asyncio.Lock()
        return self._lane

    def _predict(self, image: np.ndarray):
        results = list(
            self._model.predict(
                source=[image],
                batch=1,
                imgsz=IMAGE_SIZE,
                conf=DETECTOR_CONFIDENCE,
                iou=0.70,
                classes=[0],
                max_det=MAX_DETECTIONS,
                retina_masks=True,
                device=self.device,
                verbose=False,
            )
        )
        if len(results) != 1:
            raise RuntimeError(f"B1 inference returned {len(results)} results")
        return results[0]

    async def infer(self, image: np.ndarray, tracker: StreamTracker) -> dict[str, Any]:
        if not self.ready or self._closed:
            raise RuntimeError("runtime is not ready")
        loop = asyncio.get_running_loop()
        queued_at = time.perf_counter()
        async with self.lane:
            admitted_at = time.perf_counter()
            result = await loop.run_in_executor(
                self._executor,
                self._infer_sync,
                image,
                tracker,
            )
        result["timing_ms"]["queue"] = round((admitted_at - queued_at) * 1000, 2)
        self.frames += 1
        for stage, value in result["timing_ms"].items():
            self._latency[stage].append(float(value))
        return result

    def _infer_sync(self, image: np.ndarray, tracker: StreamTracker) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            prediction = self._predict(image)
        except Exception as error:
            raise InferenceFailure(f"{self.backend} inference failed") from error
        inferred_at = time.perf_counter()
        try:
            boxes = prediction.boxes.cpu().numpy()
            tracks = tracker.update(boxes, image)
        except Exception as error:
            raise TrackingFailure("BoT-SORT update failed") from error
        tracked_at = time.perf_counter()
        try:
            detector_objects = self._objects(
                prediction,
                tracks,
                image.shape[1],
                image.shape[0],
            )
            objects, temporal = tracker.stabilize(
                detector_objects,
                image.shape[1],
                image.shape[0],
            )
        except Exception as error:
            raise TrackingFailure("mask tracking failed") from error
        serialized_at = time.perf_counter()
        confidences = np.asarray(boxes.conf, dtype=np.float32)
        return {
            "objects": objects,
            "detections": int((confidences >= ACTIVATION_CONFIDENCE).sum()),
            "raw_detections": len(boxes),
            "continuation_candidates": int(
                (
                    (confidences >= CONTINUATION_CONFIDENCE) & (confidences < ACTIVATION_CONFIDENCE)
                ).sum()
            ),
            "detector_backed_tracks": temporal["detector_backed_tracks"],
            "low_confidence_continuations": temporal["low_confidence_continuations"],
            "held_tracks": temporal["held_tracks"],
            "tracks": len(objects),
            "tracker_frame": tracker.frame_id,
            "timing_ms": {
                "inference": round((inferred_at - started) * 1000, 2),
                "tracking": round((tracked_at - inferred_at) * 1000, 2),
                "serialize": round((serialized_at - tracked_at) * 1000, 2),
                "runtime_total": round((serialized_at - started) * 1000, 2),
            },
        }

    def _objects(
        self,
        prediction: Any,
        tracks: np.ndarray,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        polygons = prediction.masks.xy if prediction.masks is not None else []
        objects: list[dict[str, Any]] = []
        for row in tracks:
            if len(row) < 8:
                continue
            detection_index = int(row[-1])
            if not 0 <= detection_index < len(prediction.boxes):
                raise RuntimeError("tracker returned an invalid detection index")
            points: list[list[float]] = []
            if detection_index < len(polygons):
                polygon = np.asarray(polygons[detection_index], dtype=np.float32)
                if len(polygon):
                    stride = max(1, int(np.ceil(len(polygon) / MAX_POLYGON_POINTS)))
                    polygon = polygon[::stride][:MAX_POLYGON_POINTS]
                    points = [
                        [
                            round(float(np.clip(x, 0, width - 1)), 1),
                            round(float(np.clip(y, 0, height - 1)), 1),
                        ]
                        for x, y in polygon
                    ]
            x1, y1, x2, y2 = (float(value) for value in row[:4])
            area = (
                float(cv2.contourArea(np.asarray(points, dtype=np.float32)))
                if len(points) >= 3
                else 0.0
            )
            class_id = int(row[6])
            objects.append(
                {
                    "track_id": int(row[4]),
                    "class_id": class_id,
                    "class_name": self.names.get(class_id, str(class_id)),
                    "confidence": round(float(row[5]), 4),
                    "bbox": [
                        round(float(np.clip(x1, 0, width - 1)), 1),
                        round(float(np.clip(y1, 0, height - 1)), 1),
                        round(float(np.clip(x2, 0, width - 1)), 1),
                        round(float(np.clip(y2, 0, height - 1)), 1),
                    ],
                    "mask_polygon": points,
                    "mask_area_px": round(area, 1),
                }
            )
        return objects

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready and not self._closed,
            "runtime_instances": type(self)._live_instances,
            "backend": self.backend,
            "device": self.device,
            "artifact": self.selection.artifact.name,
            "artifact_sha256": self.artifact_sha256,
            "engine_sha256": (
                self.manifest["engine_sha256"] if self.manifest is not None else None
            ),
            "image_size": IMAGE_SIZE,
            "batch_size": 1,
            "scheduler": "serialized_b1",
            "frames": self.frames,
            "detector_ingress": DETECTOR_CONFIDENCE,
            "continuation_confidence": CONTINUATION_CONFIDENCE,
            "activation_confidence": ACTIVATION_CONFIDENCE,
            "mask_hold_frames": MAX_MASK_HOLD_FRAMES,
            "hold_confidence_decay": HOLD_CONFIDENCE_DECAY,
            "warmup_ms": round(self.warmup_ms, 2) if self.warmup_ms else None,
            "latency_ms": {
                stage: _percentiles(list(values)) for stage, values in self._latency.items()
            },
            "accelerator": _accelerator_health(self.device),
        }

    def close(self) -> None:
        if self._closed:
            return
        self.ready = False
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        if hasattr(self, "_model"):
            del self._model
        with self._guard:
            type(self)._live_instances = max(0, type(self)._live_instances - 1)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    return {
        "p50": round(float(np.percentile(values, 50)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
    }


def _accelerator_health(device: str) -> dict[str, Any]:
    if device == "mps":
        return {"type": "mps", "name": "Apple Metal Performance Shaders"}
    if device == "cpu":
        return {"type": "cpu", "name": platform.processor() or platform.machine()}
    try:
        import pynvml
    except ImportError:
        return _unavailable_cuda_health()
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "type": "cuda",
            "name": str(pynvml.nvmlDeviceGetName(handle)),
            "memory_used_bytes": int(memory.used),
            "memory_total_bytes": int(memory.total),
            "utilization_percent": int(utilization.gpu),
        }
    except (pynvml.NVMLError, ValueError):
        return _unavailable_cuda_health()


def _unavailable_cuda_health() -> dict[str, Any]:
    return {
        "type": "cuda",
        "name": None,
        "memory_used_bytes": None,
        "memory_total_bytes": None,
        "utilization_percent": None,
    }
