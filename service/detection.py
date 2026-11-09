"""Detector-independent boxes, polygons, and public class contract."""

from __future__ import annotations

from typing import Any

import numpy as np


class Boxes:
    """CPU XYXY/confidence/class rows consumed by tracking and renderers."""

    def __init__(self, data, orig_shape=None):
        self.data = np.asarray(data, dtype=np.float32).reshape(-1, 6)
        self.orig_shape = orig_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return Boxes(self.data[index], self.orig_shape)

    def cpu(self):
        return self

    def numpy(self):
        return self

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def xywh(self):
        result = self.xyxy.copy()
        result[:, :2] = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        result[:, 2:] = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return result

    @property
    def conf(self):
        return self.data[:, 4]

    @property
    def cls(self):
        return self.data[:, 5]


class Segmentation:
    def __init__(self, boxes, polygons):
        self.boxes = boxes
        self.masks = Polygons(polygons)


class Polygons:
    def __init__(self, polygons):
        self.xy = polygons


FACE_CLASS_ID = 0
FACE_CLASS_NAME = "face"
NUMBER_PLATE_CLASS_ID = 1
NUMBER_PLATE_CLASS_NAME = "number_plate"
EXPECTED_CLASS_NAMES = {
    FACE_CLASS_ID: FACE_CLASS_NAME,
    NUMBER_PLATE_CLASS_ID: NUMBER_PLATE_CLASS_NAME,
}


def is_face_object(item: dict[str, Any]) -> bool:
    """Return whether an object is eligible for AdaFace recognition."""

    class_id = _class_id(item)
    class_name = item.get("class_name")
    return class_id == FACE_CLASS_ID and class_name == FACE_CLASS_NAME


def is_number_plate_object(item: dict[str, Any]) -> bool:
    """Return whether an object must remain protected regardless of whitelist state."""

    class_id = _class_id(item)
    class_name = item.get("class_name")
    return class_id == NUMBER_PLATE_CLASS_ID or class_name == NUMBER_PLATE_CLASS_NAME


def _class_id(item: dict[str, Any]) -> int | None:
    value = item.get("class_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
