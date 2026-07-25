"""Connection-local BoT-SORT with one-frame segmentation mask hold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.utils.stracks import parse_bboxes
from ultralytics.utils import IterableSimpleNamespace, YAML


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "botsort.yaml"
DETECTOR_CONFIDENCE = 0.01
CONTINUATION_CONFIDENCE = 0.05
ACTIVATION_CONFIDENCE = 0.25
MAX_MASK_HOLD_FRAMES = 1
HOLD_CONFIDENCE_DECAY = 0.90


@dataclass
class _MaskState:
    bbox: np.ndarray
    polygon: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    last_detection_frame: int


class _LocalBOTrack(BOTrack):
    _local_count = 0

    @classmethod
    def next_id(cls) -> int:
        cls._local_count += 1
        return cls._local_count

    @classmethod
    def reset_id(cls) -> None:
        cls._local_count = 0


class _ConnectionBOTSORT(BOTSORT):
    def __init__(self, args: Any):
        self._track_class = type(
            "ConnectionBOTrack",
            (_LocalBOTrack,),
            {"_local_count": 0},
        )
        super().__init__(args)

    def reset_id(self) -> None:
        self._track_class.reset_id()

    def init_track(self, results: Any, img: np.ndarray | None = None) -> list[BOTrack]:
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        if self.args.with_reid and self.encoder is not None and img is not None:
            features = self.encoder(img, bboxes)
            return [
                self._track_class(xywh, score, cls, feature)
                for xywh, score, cls, feature in zip(
                    bboxes,
                    results.conf,
                    results.cls,
                    features,
                )
            ]
        return [
            self._track_class(xywh, score, cls)
            for xywh, score, cls in zip(bboxes, results.conf, results.cls)
        ]


class StreamTracker:
    """One ordered tracker and mask cache owned by one WebSocket connection."""

    def __init__(self, config: Path = DEFAULT_CONFIG, device: str = "0"):
        config_path = config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"tracker config not found: {config_path}")
        values = YAML.load(str(config_path))
        if values.get("tracker_type") != "botsort":
            raise ValueError("tracker_type must be botsort")
        if float(values.get("track_low_thresh", -1)) != CONTINUATION_CONFIDENCE:
            raise ValueError("track_low_thresh must be 0.05")
        if float(values.get("track_high_thresh", -1)) != ACTIVATION_CONFIDENCE:
            raise ValueError("track_high_thresh must be 0.25")
        if float(values.get("new_track_thresh", -1)) != ACTIVATION_CONFIDENCE:
            raise ValueError("new_track_thresh must be 0.25")
        values["with_reid"] = False
        values["device"] = device
        self.config_path = config_path
        self._tracker = _ConnectionBOTSORT(IterableSimpleNamespace(**values))
        self._masks: dict[int, _MaskState] = {}

    @property
    def frame_id(self) -> int:
        return self._tracker.frame_id

    def update(self, boxes: Any, image: np.ndarray) -> np.ndarray:
        return self._tracker.update(boxes, image)

    def stabilize(
        self,
        objects: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        current: list[dict[str, Any]] = []
        current_ids: set[int] = set()
        current_boxes: list[np.ndarray] = []
        low_confidence_continuations = 0

        for item in objects:
            tracked = dict(item)
            track_id = int(tracked["track_id"])
            confidence = float(tracked["confidence"])
            source = (
                "continued_low"
                if confidence < ACTIVATION_CONFIDENCE
                else "detected"
            )
            if source == "continued_low":
                low_confidence_continuations += 1
            tracked.update({"source": source, "held": False, "hold_frames": 0})
            current.append(tracked)
            current_ids.add(track_id)
            bbox = np.asarray(tracked["bbox"], dtype=np.float32)
            current_boxes.append(bbox)
            polygon = np.asarray(tracked["mask_polygon"], dtype=np.float32).reshape(
                (-1, 2)
            )
            if len(polygon) >= 3:
                self._masks[track_id] = _MaskState(
                    bbox=bbox.copy(),
                    polygon=polygon.copy(),
                    confidence=confidence,
                    class_id=int(tracked["class_id"]),
                    class_name=str(tracked["class_name"]),
                    last_detection_frame=self.frame_id,
                )

        held_tracks = 0
        for lost in self._tracker.lost_stracks:
            track_id = int(lost.track_id)
            state = self._masks.get(track_id)
            if state is None or track_id in current_ids:
                continue
            gap = self.frame_id - state.last_detection_frame
            if gap != MAX_MASK_HOLD_FRAMES:
                continue
            target = np.asarray(lost.xyxy, dtype=np.float32).reshape(4)
            target[[0, 2]] = np.clip(target[[0, 2]], 0, width - 1)
            target[[1, 3]] = np.clip(target[[1, 3]], 0, height - 1)
            if target[2] <= target[0] or target[3] <= target[1]:
                continue
            if _max_iou(target, current_boxes) >= 0.5:
                continue
            polygon = _warp_polygon(state.polygon, state.bbox, target, width, height)
            if len(polygon) < 3:
                continue
            current.append(
                {
                    "track_id": track_id,
                    "class_id": state.class_id,
                    "class_name": state.class_name,
                    "confidence": round(
                        state.confidence * HOLD_CONFIDENCE_DECAY,
                        4,
                    ),
                    "bbox": [round(float(value), 1) for value in target],
                    "mask_polygon": [
                        [round(float(x), 1), round(float(y), 1)] for x, y in polygon
                    ],
                    "mask_area_px": round(_polygon_area(polygon), 1),
                    "source": "held",
                    "held": True,
                    "hold_frames": 1,
                    "hold_limit": 1,
                }
            )
            held_tracks += 1

        for track_id, state in list(self._masks.items()):
            if self.frame_id - state.last_detection_frame > 30:
                del self._masks[track_id]

        return current, {
            "detector_backed_tracks": len(objects),
            "low_confidence_continuations": low_confidence_continuations,
            "held_tracks": held_tracks,
        }

    def reset(self) -> None:
        self._tracker.reset()
        self._masks.clear()


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def _max_iou(box: np.ndarray, boxes: list[np.ndarray]) -> float:
    if not boxes:
        return 0.0
    candidates = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
    x1 = np.maximum(float(box[0]), candidates[:, 0])
    y1 = np.maximum(float(box[1]), candidates[:, 1])
    x2 = np.minimum(float(box[2]), candidates[:, 2])
    y2 = np.minimum(float(box[3]), candidates[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, float(box[2] - box[0])) * max(
        0.0,
        float(box[3] - box[1]),
    )
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0,
        candidates[:, 3] - candidates[:, 1],
    )
    union = area + candidate_area - intersection
    ious = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    return float(ious.max(initial=0.0))


def _warp_polygon(
    polygon: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    source_width = max(float(source[2] - source[0]), 1.0)
    source_height = max(float(source[3] - source[1]), 1.0)
    target_width = max(float(target[2] - target[0]), 1.0)
    target_height = max(float(target[3] - target[1]), 1.0)
    warped = np.asarray(polygon, dtype=np.float32).copy()
    warped[:, 0] = target[0] + (warped[:, 0] - source[0]) * target_width / source_width
    warped[:, 1] = target[1] + (warped[:, 1] - source[1]) * target_height / source_height
    warped[:, 0] = np.clip(warped[:, 0], 0, width - 1)
    warped[:, 1] = np.clip(warped[:, 1], 0, height - 1)
    return warped
