"""Detection overlays for the browser image-inference demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

CLASS_COLORS = {
    "face": (48, 205, 96),
    "number_plate": (46, 126, 255),
}
DEFAULT_COLOR = (255, 196, 64)


def annotate_detections(image: np.ndarray, objects: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Return a copy with translucent mask regions and clear box/mask outlines."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR array")

    output = image.copy()
    overlay = output.copy()
    line_width = max(2, round(min(output.shape[:2]) / 360))
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        color = _color_for(item)
        polygon = _polygon(item.get("mask_polygon"))
        if polygon is not None:
            cv2.fillPoly(overlay, [polygon], color)
    output = cv2.addWeighted(overlay, 0.22, output, 0.78, 0)

    for item in objects:
        if not isinstance(item, Mapping):
            continue
        color = _color_for(item)
        polygon = _polygon(item.get("mask_polygon"))
        if polygon is not None:
            cv2.polylines(output, [polygon], True, color, line_width, cv2.LINE_AA)
        bbox = _bbox(item.get("bbox"))
        if bbox is not None:
            cv2.rectangle(output, bbox[:2], bbox[2:], color, line_width, cv2.LINE_AA)
        _draw_label(output, _label(item), bbox, color, line_width)
    return output


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode a BGR image as one complete JPEG."""

    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in 1..100")
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise ValueError("could not encode visualization as JPEG")
    return encoded.tobytes()


def _color_for(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return CLASS_COLORS.get(str(item.get("class_name", "")), DEFAULT_COLOR)


def _polygon(value: Any) -> np.ndarray | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    points: list[tuple[int, int]] = []
    for point in value:
        if not isinstance(point, Sequence) or len(point) < 2:
            return None
        try:
            points.append((round(float(point[0])), round(float(point[1]))))
        except (TypeError, ValueError, OverflowError):
            return None
    return np.asarray(points, dtype=np.int32)


def _bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (round(float(value[index])) for index in range(4))
    except (TypeError, ValueError, OverflowError):
        return None
    return x1, y1, x2, y2


def _label(item: Mapping[str, Any]) -> str:
    name = str(item.get("class_name") or "object")
    try:
        confidence = f" {float(item.get('confidence', 0.0)):.2f}"
    except (TypeError, ValueError):
        confidence = ""
    track_id = item.get("track_id")
    suffix = f"  #{track_id}" if track_id is not None else ""
    return f"{name}{confidence}{suffix}"


def _draw_label(
    image: np.ndarray,
    label: str,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    if bbox is None:
        return
    font_scale = max(0.45, min(0.85, min(image.shape[:2]) / 900))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, line_width)
    x = max(0, bbox[0])
    y = max(text_height + baseline + 4, bbox[1])
    top = y - text_height - baseline - 4
    right = min(image.shape[1], x + text_width + 8)
    cv2.rectangle(image, (x, top), (right, y), color, cv2.FILLED)
    cv2.putText(
        image,
        label,
        (x + 4, y - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        line_width,
        cv2.LINE_AA,
    )
