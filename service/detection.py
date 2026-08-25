"""Serving contract for the YOLO segmentation classes."""

from __future__ import annotations

from typing import Any

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
