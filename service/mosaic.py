"""Fail-closed server-side face mosaic composition."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from service.protocol import MAX_JPEG_BYTES

JPEG_QUALITY = 90
BLUR_SIGMA = 16.0
BLUR_DOWNSAMPLE = 2
MASK_FEATHER_RADIUS = 8
MAX_MASK_POINTS = 64


def mosaic_jpeg(
    image: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    max_bytes: int = MAX_JPEG_BYTES,
) -> bytes:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("mosaic input must be a uint8 BGR image")
    if not 1 <= max_bytes <= MAX_JPEG_BYTES:
        raise ValueError(f"mosaic byte limit must be in 1..{MAX_JPEG_BYTES}")

    mask = _protected_mask(image.shape[:2], objects)
    blend_mask = _feathered_mask(mask)
    output = image
    mask_rows, mask_columns = np.nonzero(blend_mask)
    if mask_rows.size:
        padding = math.ceil(BLUR_SIGMA * 3)
        top = max(0, int(mask_rows.min()) - padding)
        bottom = min(image.shape[0], int(mask_rows.max()) + padding + 1)
        left = max(0, int(mask_columns.min()) - padding)
        right = min(image.shape[1], int(mask_columns.max()) + padding + 1)
        region = image[top:bottom, left:right]
        reduced_size = (
            max(1, math.ceil(region.shape[1] / BLUR_DOWNSAMPLE)),
            max(1, math.ceil(region.shape[0] / BLUR_DOWNSAMPLE)),
        )
        reduced = cv2.resize(region, reduced_size, interpolation=cv2.INTER_AREA)
        blurred_reduced = cv2.GaussianBlur(
            reduced,
            (0, 0),
            BLUR_SIGMA / BLUR_DOWNSAMPLE,
            BLUR_SIGMA / BLUR_DOWNSAMPLE,
        )
        blurred = cv2.resize(
            blurred_reduced,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        output = image.copy()
        alpha = blend_mask[top:bottom, left:right, None].astype(np.uint32)
        inverse_alpha = 255 - alpha
        output[top:bottom, left:right] = (
            region.astype(np.uint32) * inverse_alpha + blurred.astype(np.uint32) * alpha + 127
        ) // 255

    try:
        encoded, payload = cv2.imencode(
            ".jpg",
            output,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
    except cv2.error as error:
        raise ValueError("mosaic JPEG encoding failed") from error
    if not encoded:
        raise ValueError("mosaic JPEG encoding failed")
    jpeg = payload.tobytes()
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("mosaic encoder returned an invalid JPEG")
    if len(jpeg) > max_bytes:
        raise ValueError(f"mosaic JPEG exceeds the {max_bytes} byte limit")
    return jpeg


def _protected_mask(
    dimensions: tuple[int, int],
    objects: list[dict[str, Any]],
) -> np.ndarray:
    height, width = dimensions
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in objects:
        if item.get("whitelisted") is True:
            continue
        polygon = _polygon(item.get("mask_polygon"), width, height)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def _feathered_mask(mask: np.ndarray) -> np.ndarray:
    """Extend the protected area and taper only its outer boundary.

    The source face pixels always remain at full opacity.  The short outer
    taper blends the already-strongly blurred image into the scene instead of
    leaving a conspicuous hard edge around the segmentation polygon.
    """
    if not np.any(mask):
        return mask

    kernel_size = MASK_FEATHER_RADIUS * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask, kernel)
    distance = cv2.distanceTransform(expanded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.minimum(distance / MASK_FEATHER_RADIUS, 1.0) * 255.0
    feathered = np.rint(alpha).astype(np.uint8)
    feathered[mask != 0] = 255
    return feathered


def _polygon(value: Any, width: int, height: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or not 3 <= len(value) <= MAX_MASK_POINTS
        or any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in value)
    ):
        raise ValueError("protected face has an invalid mask polygon")
    try:
        polygon = np.asarray(value, dtype=np.float32).reshape((-1, 2))
    except (TypeError, ValueError) as error:
        raise ValueError("protected face has an invalid mask polygon") from error
    if not np.isfinite(polygon).all():
        raise ValueError("protected face mask contains a non-finite point")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    rounded = np.rint(polygon).astype(np.int32)
    area = float(cv2.contourArea(rounded))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("protected face mask has no area")
    return rounded
