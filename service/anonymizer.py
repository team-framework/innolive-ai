"""Head segmentation, BoT-SORT tracking, and temporally stable anonymization."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, ops


LOGGER = logging.getLogger(__name__)
HEAD_CLASS_ID = 0
HEAD_CLASS_NAME = "head"


@dataclass(frozen=True)
class AnonymizerConfig:
    """Runtime settings shared by every gRPC video stream."""

    model_path: Path
    tracker_config_path: Path
    imgsz: int = 640
    confidence: float = 0.10
    iou: float = 0.70
    max_detections: int = 100
    device: str | None = None
    prefer_tensorrt: bool = True
    warmup: bool = True
    mask_hold_frames: int = 8
    unconfirmed_hold_frames: int = 2
    temporal_decay: float = 0.70
    mask_threshold: float = 0.35
    mask_dilation: int = 9
    mask_feather: int = 9
    blur_sigma: float = 25.0
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if self.imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.iou <= 1.0:
            raise ValueError("confidence and iou must be between 0 and 1")
        if self.mask_hold_frames < 0:
            raise ValueError("mask_hold_frames cannot be negative")
        if self.unconfirmed_hold_frames < 0:
            raise ValueError("unconfirmed_hold_frames cannot be negative")
        if not 0.0 <= self.temporal_decay <= 1.0:
            raise ValueError("temporal_decay must be between 0 and 1")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if self.mask_dilation < 0 or self.mask_feather < 0:
            raise ValueError("mask dilation/feather values cannot be negative")
        if self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass
class DetectionBatch:
    """CPU-side detections and full-resolution segmentation masks."""

    boxes: Any
    masks: np.ndarray


@dataclass
class _TrackedMask:
    mask: np.ndarray
    bbox: np.ndarray
    missed_frames: int = 0


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def resolve_model_path(model_path: Path, prefer_tensorrt: bool = True) -> Path:
    """Prefer a fresh sibling TensorRT engine and safely fall back to the PT file."""

    requested = model_path.expanduser().resolve()
    if requested.suffix == ".engine":
        if requested.is_file():
            return requested
        pt_fallback = requested.with_suffix(".pt")
        if pt_fallback.is_file():
            LOGGER.warning(
                "TensorRT engine %s was not found; falling back to %s",
                requested,
                pt_fallback,
            )
            return pt_fallback
        raise FileNotFoundError(f"Model does not exist: {requested}")

    if not requested.is_file():
        raise FileNotFoundError(f"Model does not exist: {requested}")

    engine_path = requested.with_suffix(".engine")
    if prefer_tensorrt and engine_path.is_file():
        if engine_path.stat().st_mtime >= requested.stat().st_mtime:
            return engine_path
        LOGGER.warning(
            "Ignoring stale TensorRT engine %s because %s is newer",
            engine_path,
            requested,
        )
    return requested


class HeadAnonymizerRuntime:
    """Thread-safe shared YOLO runtime; tracking remains private to each stream."""

    def __init__(self, config: AnonymizerConfig):
        self.config = config
        self.model_path = resolve_model_path(
            config.model_path,
            prefer_tensorrt=config.prefer_tensorrt,
        )
        self._inference_lock = threading.Lock()
        self._tracker_args = self._load_tracker_args(config.tracker_config_path)
        self.model = YOLO(str(self.model_path), task="segment")
        self._validate_model()

        LOGGER.info(
            "Loaded head segmentation model %s (backend=%s, class 0=%s)",
            self.model_path,
            "TensorRT" if self.using_tensorrt else "PyTorch",
            HEAD_CLASS_NAME,
        )
        if config.warmup:
            self.warmup()

    @property
    def using_tensorrt(self) -> bool:
        return self.model_path.suffix == ".engine"

    def _validate_model(self) -> None:
        if self.model.task != "segment":
            raise ValueError(
                f"Expected a segmentation model, but {self.model_path} is {self.model.task!r}"
            )
        names = _normalise_names(self.model.names)
        if names.get(HEAD_CLASS_ID, "").strip().lower() != HEAD_CLASS_NAME:
            raise ValueError(
                "The model class mapping must contain 0: 'head'; "
                f"received {names!r} from {self.model_path}"
            )

    @staticmethod
    def _load_tracker_args(path: Path) -> SimpleNamespace:
        tracker_path = path.expanduser().resolve()
        if not tracker_path.is_file():
            raise FileNotFoundError(f"BoT-SORT config does not exist: {tracker_path}")
        values = YAML.load(str(tracker_path))
        if values.get("tracker_type") != "botsort":
            raise ValueError(f"tracker_type must be 'botsort' in {tracker_path}")
        return SimpleNamespace(**values)

    def _prediction_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "imgsz": self.config.imgsz,
            # The predictor must retain BoT-SORT's low-confidence second-stage
            # candidates even if the public confidence setting is raised.
            "conf": min(
                self.config.confidence,
                float(self._tracker_args.track_low_thresh),
            ),
            "iou": self.config.iou,
            "classes": [HEAD_CLASS_ID],
            "max_det": self.config.max_detections,
            "retina_masks": False,
            "verbose": False,
        }
        if self.config.device:
            kwargs["device"] = self.config.device
        requested_device = (self.config.device or "").lower()
        cuda_requested = (
            not requested_device
            or requested_device.isdigit()
            or requested_device.startswith("cuda")
        )
        if not self.using_tensorrt and cuda_requested and torch.cuda.is_available():
            kwargs["half"] = True
        return kwargs

    def warmup(self) -> None:
        """Build CUDA/TensorRT execution state before the first real frame."""

        frame = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
        with self._inference_lock:
            self.model.predict(frame, **self._prediction_kwargs())

    def infer(self, frame: np.ndarray) -> DetectionBatch:
        """Run inference and detach everything needed by a per-stream tracker."""

        with self._inference_lock:
            result = self.model.predict(frame, **self._prediction_kwargs())[0]
            boxes = result.boxes.cpu().numpy()
            if result.masks is None or len(result.masks.data) == 0:
                masks = np.empty((0, frame.shape[0], frame.shape[1]), dtype=np.float32)
            else:
                scaled = ops.scale_masks(
                    result.masks.data[:, None],
                    frame.shape[:2],
                )[:, 0]
                masks = scaled.float().cpu().numpy()
        return DetectionBatch(boxes=boxes, masks=masks)

    def create_stream(self) -> "HeadAnonymizerStream":
        return HeadAnonymizerStream(self)


class HeadAnonymizerStream:
    """Stateful BoT-SORT and mask history for one ProcessVideo RPC stream."""

    def __init__(self, runtime: HeadAnonymizerRuntime):
        self.runtime = runtime
        self.config = runtime.config
        self.tracker = BOTSORT(runtime._tracker_args)
        self._tracked_masks: dict[int, _TrackedMask] = {}
        self._unconfirmed_masks: list[_TrackedMask] = []
        self._frame_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self._tracked_masks.clear()
        self._unconfirmed_masks.clear()
        self._frame_shape = None

    def process_bytes(self, image_bytes: bytes) -> bytes:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("VideoChunk.data is not a decodable image")

        output, changed = self.process_frame(frame)
        if not changed:
            return image_bytes

        extension = ".png" if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            if extension == ".jpg"
            else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
        ok, result = cv2.imencode(extension, output, params)
        if not ok:
            raise RuntimeError(f"Could not encode processed frame as {extension}")
        return result.tobytes()

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        shape = frame.shape[:2]
        if self._frame_shape is not None and self._frame_shape != shape:
            LOGGER.info(
                "Frame resolution changed from %s to %s; resetting BoT-SORT state",
                self._frame_shape,
                shape,
            )
            self.reset()
        self._frame_shape = shape

        detections = self.runtime.infer(frame)
        tracks = self.tracker.update(detections.boxes, frame)
        masks = self._stable_masks(detections, tracks, shape)
        if not masks:
            return frame, False
        return self._apply_blur(frame, masks), True

    def _stable_masks(
        self,
        detections: DetectionBatch,
        tracks: np.ndarray,
        shape: tuple[int, int],
    ) -> list[np.ndarray]:
        render_masks: list[np.ndarray] = []
        matched_detection_indices: set[int] = set()
        active_track_ids: set[int] = set()
        active_bboxes: list[np.ndarray] = []

        for track in tracks:
            if len(track) < 8 or int(track[6]) != HEAD_CLASS_ID:
                continue
            track_id = int(track[4])
            detection_index = int(track[-1])
            if not 0 <= detection_index < len(detections.masks):
                continue

            bbox = np.asarray(track[:4], dtype=np.float32)
            current = detections.masks[detection_index].astype(np.float32, copy=False)
            previous = self._tracked_masks.get(track_id)
            if previous is not None:
                previous_mask = self._warp_mask(
                    previous.mask, previous.bbox, bbox, shape
                )
                current = np.maximum(
                    current, previous_mask * self.config.temporal_decay
                )

            self._tracked_masks[track_id] = _TrackedMask(current, bbox)
            render_masks.append(current)
            active_track_ids.add(track_id)
            active_bboxes.append(bbox)
            matched_detection_indices.add(detection_index)

        # BoT-SORT deliberately suppresses new unconfirmed tracks for one frame. Mask
        # those raw detections immediately so anonymization never has a first-frame gap.
        scores = np.asarray(detections.boxes.conf, dtype=np.float32)
        detection_bboxes = np.asarray(detections.boxes.xyxy, dtype=np.float32)
        unconfirmed: list[_TrackedMask] = []
        for index, mask in enumerate(detections.masks):
            if (
                index not in matched_detection_indices
                and scores[index] >= self.runtime._tracker_args.new_track_thresh
            ):
                current = mask.astype(np.float32, copy=False)
                unconfirmed.append(_TrackedMask(current, detection_bboxes[index]))

        self._hold_unconfirmed_masks(
            unconfirmed,
            active_bboxes,
            render_masks,
            shape,
        )

        lost_tracks = {
            int(track.track_id): np.asarray(track.xyxy, dtype=np.float32)
            for track in self.tracker.lost_stracks
        }
        for track_id in list(self._tracked_masks):
            if track_id in active_track_ids:
                continue
            cached = self._tracked_masks[track_id]
            predicted_bbox = lost_tracks.get(track_id)
            if (
                predicted_bbox is None
                or cached.missed_frames >= self.config.mask_hold_frames
            ):
                del self._tracked_masks[track_id]
                continue
            cached.mask = self._warp_mask(
                cached.mask, cached.bbox, predicted_bbox, shape
            )
            cached.bbox = predicted_bbox
            cached.missed_frames += 1
            render_masks.append(cached.mask)

        return render_masks

    def _hold_unconfirmed_masks(
        self,
        current: list[_TrackedMask],
        active_bboxes: list[np.ndarray],
        render_masks: list[np.ndarray],
        shape: tuple[int, int],
    ) -> None:
        """Bridge the one-frame confirmation window of newly created tracks."""

        next_pending: list[_TrackedMask] = []
        matched_current: set[int] = set()
        for cached in self._unconfirmed_masks:
            if any(self._bbox_iou(cached.bbox, bbox) >= 0.30 for bbox in active_bboxes):
                continue

            best_index = -1
            best_iou = 0.0
            for index, candidate in enumerate(current):
                if index in matched_current:
                    continue
                iou = self._bbox_iou(cached.bbox, candidate.bbox)
                if iou > best_iou:
                    best_index, best_iou = index, iou
            if best_index >= 0 and best_iou >= 0.30:
                candidate = current[best_index]
                previous = self._warp_mask(
                    cached.mask,
                    cached.bbox,
                    candidate.bbox,
                    shape,
                )
                candidate.mask = np.maximum(
                    candidate.mask,
                    previous * self.config.temporal_decay,
                )
                next_pending.append(candidate)
                render_masks.append(candidate.mask)
                matched_current.add(best_index)
            elif cached.missed_frames < self.config.unconfirmed_hold_frames:
                cached.missed_frames += 1
                next_pending.append(cached)
                render_masks.append(cached.mask)

        for index, candidate in enumerate(current):
            if index not in matched_current:
                next_pending.append(candidate)
                render_masks.append(candidate.mask)
        self._unconfirmed_masks = next_pending

    @staticmethod
    def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        first_area = max(float(first[2] - first[0]), 0.0) * max(
            float(first[3] - first[1]), 0.0
        )
        second_area = max(float(second[2] - second[0]), 0.0) * max(
            float(second[3] - second[1]), 0.0
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _warp_mask(
        mask: np.ndarray,
        old_bbox: np.ndarray,
        new_bbox: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        old_width = max(float(old_bbox[2] - old_bbox[0]), 1.0)
        old_height = max(float(old_bbox[3] - old_bbox[1]), 1.0)
        new_width = max(float(new_bbox[2] - new_bbox[0]), 1.0)
        new_height = max(float(new_bbox[3] - new_bbox[1]), 1.0)
        scale_x = new_width / old_width
        scale_y = new_height / old_height
        transform = np.asarray(
            [
                [scale_x, 0.0, float(new_bbox[0]) - float(old_bbox[0]) * scale_x],
                [0.0, scale_y, float(new_bbox[1]) - float(old_bbox[1]) * scale_y],
            ],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            mask,
            transform,
            (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def _apply_blur(self, frame: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
        combined = np.maximum.reduce(masks)
        binary = (combined >= self.config.mask_threshold).astype(np.uint8)
        if self.config.mask_dilation:
            size = self.config.mask_dilation * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            binary = cv2.dilate(binary, kernel)
        if not np.any(binary):
            return frame

        alpha = binary.astype(np.float32)
        if self.config.mask_feather:
            size = self.config.mask_feather * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (size, size), 0)

        # Gaussian blur is the largest CPU post-processing cost. Restrict it to
        # the anonymized area plus a 3-sigma halo instead of filtering the frame.
        ys, xs = np.nonzero(binary)
        halo = int(np.ceil(self.config.blur_sigma * 3))
        x1 = max(int(xs.min()) - halo, 0)
        y1 = max(int(ys.min()) - halo, 0)
        x2 = min(int(xs.max()) + halo + 1, frame.shape[1])
        y2 = min(int(ys.max()) + halo + 1, frame.shape[0])
        source = frame[y1:y2, x1:x2]
        roi_alpha = alpha[y1:y2, x1:x2, None]
        blurred = cv2.GaussianBlur(
            source,
            (0, 0),
            sigmaX=self.config.blur_sigma,
            sigmaY=self.config.blur_sigma,
        )
        blended = source.astype(np.float32) * (1.0 - roi_alpha)
        blended += blurred.astype(np.float32) * roi_alpha
        output = frame.copy()
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return output
