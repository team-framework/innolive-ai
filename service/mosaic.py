"""Fail-closed server-side object mosaic composition."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from service.detection import is_number_plate_object
from service.protocol import MAX_JPEG_BYTES

JPEG_QUALITY = 90
DEFAULT_BLUR_RADIUS = 24.0
DEFAULT_PIXEL_SIZE = 2
MAX_BLUR_RADIUS = 64.0
MAX_PIXEL_SIZE = 8
MASK_FEATHER_RADIUS = 8
MAX_MASK_POINTS = 64


def validate_mosaic_params(
    blur_radius: Any,
    pixel_size: Any,
) -> tuple[float, int]:
    """Validate client-supplied mosaic strength and return (radius, size)."""
    if isinstance(blur_radius, bool) or not isinstance(blur_radius, (int, float)):
        raise ValueError("blur_radius must be a number of pixels")
    radius = float(blur_radius)
    if not math.isfinite(radius) or not 0 < radius <= MAX_BLUR_RADIUS:
        raise ValueError(f"blur_radius must be in (0, {MAX_BLUR_RADIUS}]")
    if (
        isinstance(pixel_size, bool)
        or not isinstance(pixel_size, int)
        or not 1 <= pixel_size <= MAX_PIXEL_SIZE
    ):
        raise ValueError(f"pixel_size must be an integer in 1..{MAX_PIXEL_SIZE}")
    return radius, pixel_size


def mosaic_jpeg(
    image: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
    max_bytes: int = MAX_JPEG_BYTES,
) -> bytes:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("mosaic input must be a uint8 BGR image")
    if not 1 <= max_bytes <= MAX_JPEG_BYTES:
        raise ValueError(f"mosaic byte limit must be in 1..{MAX_JPEG_BYTES}")
    blur_radius, pixel_size = validate_mosaic_params(blur_radius, pixel_size)

    mask = _protected_mask(image.shape[:2], objects)
    output = _mosaic_masked_region(image, _feathered_mask(mask), blur_radius, pixel_size)

    return _encode_jpeg(output, max_bytes)


def _mosaic_masked_region(
    image: np.ndarray,
    blend_mask: np.ndarray,
    blur_radius: float,
    pixel_size: int,
) -> np.ndarray:
    """Apply the protected-region transform to the smallest blur-safe crop."""

    mask_rows, mask_columns = np.nonzero(blend_mask)
    if not mask_rows.size:
        return image

    padding = math.ceil(blur_radius * 3)
    top = max(0, int(mask_rows.min()) - padding)
    bottom = min(image.shape[0], int(mask_rows.max()) + padding + 1)
    left = max(0, int(mask_columns.min()) - padding)
    right = min(image.shape[1], int(mask_columns.max()) + padding + 1)
    region = image[top:bottom, left:right]
    reduced_size = (
        max(1, math.ceil(region.shape[1] / pixel_size)),
        max(1, math.ceil(region.shape[0] / pixel_size)),
    )
    reduced = cv2.resize(region, reduced_size, interpolation=cv2.INTER_AREA)
    reduced_sigma = blur_radius / pixel_size
    blurred = cv2.resize(
        cv2.GaussianBlur(reduced, (0, 0), reduced_sigma, reduced_sigma),
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    output = image.copy()
    alpha = blend_mask[top:bottom, left:right, None].astype(np.uint32)
    inverse_alpha = 255 - alpha
    output[top:bottom, left:right] = (
        region.astype(np.uint32) * inverse_alpha + blurred.astype(np.uint32) * alpha + 127
    ) // 255
    return output


def _encode_jpeg(image: np.ndarray, max_bytes: int) -> bytes:
    """Encode the final BGR frame while retaining the serving byte contract."""

    try:
        encoded, payload = cv2.imencode(
            ".jpg",
            image,
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
        if item.get("whitelisted") is True and not is_number_plate_object(item):
            continue
        polygon = _polygon(item.get("mask_polygon"), width, height)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def _feathered_mask(mask: np.ndarray) -> np.ndarray:
    """Extend the protected area and taper only its outer boundary.

    The source protected pixels always remain at full opacity.  The short outer
    taper blends the already-strongly blurred image into the scene instead of
    leaving a conspicuous hard edge around the segmentation polygon.  Set
    ``MASK_FEATHER_RADIUS`` to 0 to blur exactly the segmentation polygon.
    """
    if not np.any(mask):
        return mask
    if MASK_FEATHER_RADIUS <= 0:
        return (mask != 0).astype(np.uint8) * 255

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
        raise ValueError("protected object has an invalid mask polygon")
    try:
        polygon = np.asarray(value, dtype=np.float32).reshape((-1, 2))
    except (TypeError, ValueError) as error:
        raise ValueError("protected object has an invalid mask polygon") from error
    if not np.isfinite(polygon).all():
        raise ValueError("protected object mask contains a non-finite point")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    rounded = np.rint(polygon).astype(np.int32)
    area = float(cv2.contourArea(rounded))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("protected object mask has no area")
    return rounded
