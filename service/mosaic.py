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
DEFAULT_MASK_MARGIN = 8
MASK_FEATHER_RADIUS = DEFAULT_MASK_MARGIN  # Compatibility for existing compositor imports.
MAX_MASK_MARGIN = 64
MAX_MASK_POINTS = 64
MAX_MASK_PAD = 16


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


def validate_mask_margin(value: Any) -> int:
    """Reject invalid margins without shrinking the detected protected region."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_MASK_MARGIN:
        raise ValueError(f"mask_margin must be an integer in 0..{MAX_MASK_MARGIN}")
    return value


def mosaic_jpeg(
    image: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
    mask_margin: int = DEFAULT_MASK_MARGIN,
    mask_pad: int = 0,
    max_bytes: int = MAX_JPEG_BYTES,
) -> bytes:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("mosaic input must be a uint8 BGR image")
    if not 1 <= max_bytes <= MAX_JPEG_BYTES:
        raise ValueError(f"mosaic byte limit must be in 1..{MAX_JPEG_BYTES}")
    blur_radius, pixel_size = validate_mosaic_params(blur_radius, pixel_size)

    mask = _protected_mask(image.shape[:2], objects, mask_pad)
    output = _mosaic_masked_region(
        image, _feathered_mask(mask, mask_margin), blur_radius, pixel_size
    )

    return _encode_jpeg(output, max_bytes)


def mosaic_yuv420p(
    image: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
    mask_margin: int = DEFAULT_MASK_MARGIN,
    mask_pad: int = 0,
) -> bytes:
    """Compose the mosaic and return raw yuv420p bytes for the raw wire format."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("mosaic input must be a uint8 BGR image")
    height, width = image.shape[:2]
    if width % 2 or height % 2:
        raise ValueError("raw mosaic requires even dimensions")
    blur_radius, pixel_size = validate_mosaic_params(blur_radius, pixel_size)

    mask = _protected_mask(image.shape[:2], objects, mask_pad)
    output = _mosaic_masked_region(
        image, _feathered_mask(mask, mask_margin), blur_radius, pixel_size
    )

    return _encode_yuv420p(output)


def _mosaic_masked_region(
    image: np.ndarray,
    blend_mask: np.ndarray,
    blur_radius: float,
    pixel_size: int,
) -> np.ndarray:
    """Apply the protected-region transform to the smallest blur-safe crop."""

    mask_left, mask_top, mask_width, mask_height = cv2.boundingRect(blend_mask)
    if not mask_width:
        return image

    padding = math.ceil(blur_radius * 3)
    top = max(0, mask_top - padding)
    bottom = min(image.shape[0], mask_top + mask_height + padding)
    left = max(0, mask_left - padding)
    right = min(image.shape[1], mask_left + mask_width + padding)
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
    # Complementary weights bound the rounded numerator to 255 * 255 + 127.
    # uint16 preserves the exact result while halving blend-buffer bandwidth.
    alpha = blend_mask[top:bottom, left:right, None].astype(np.uint16)
    inverse_alpha = 255 - alpha
    output[top:bottom, left:right] = (
        region.astype(np.uint16) * inverse_alpha + blurred.astype(np.uint16) * alpha + 127
    ) // 255
    return output


def _encode_yuv420p(image: np.ndarray) -> bytes:
    """Convert the final BGR frame to planar yuv420p bytes for the raw wire format."""

    try:
        planar = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    except cv2.error as error:
        raise ValueError("mosaic yuv420p conversion failed") from error
    return planar.tobytes()


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
    pad: int = 0,
) -> np.ndarray:
    """Fill protected polygons, then widen them by ``pad`` fully opaque pixels."""
    if isinstance(pad, bool) or not isinstance(pad, int) or not 0 <= pad <= MAX_MASK_PAD:
        raise ValueError(f"mask pad must be an integer in 0..{MAX_MASK_PAD}")
    height, width = dimensions
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in objects:
        if item.get("whitelisted") is True and not is_number_plate_object(item):
            continue
        polygon = _polygon(item.get("mask_polygon"), width, height)
        cv2.fillPoly(mask, [polygon], 255)
    if not pad:
        return mask
    left, top, box_width, box_height = cv2.boundingRect(mask)
    if not box_width:
        return mask
    # 보호 영역 경계 주변만 팽창한다. 범위 밖 픽셀은 반경 pad 안에 보호 픽셀이 없다.
    bottom = min(height, top + box_height + pad)
    right = min(width, left + box_width + pad)
    top, left = max(0, top - pad), max(0, left - pad)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    mask[top:bottom, left:right] = cv2.dilate(mask[top:bottom, left:right], kernel)
    return mask


def _feathered_mask(mask: np.ndarray, mask_margin: int = DEFAULT_MASK_MARGIN) -> np.ndarray:
    """Extend the protected area and taper only its outer boundary.

    The source protected pixels always remain at full opacity. The outer taper
    stays at most 8px wide, so larger margins extend the fully blurred area into
    the scene instead of just stretching a translucent band. It avoids
    leaving a conspicuous hard edge around the segmentation polygon. A zero
    margin protects exactly that polygon; negative margins are never accepted.
    """
    mask_margin = validate_mask_margin(mask_margin)
    left, top, width, height = cv2.boundingRect(mask)
    if not width:
        return mask
    if mask_margin == 0:
        return (mask != 0).astype(np.uint8) * 255

    if mask_margin != DEFAULT_MASK_MARGIN:
        # One distance transform keeps custom margins independent of kernel area.
        # Restrict it to the protected bounds plus the requested outer band.
        right = min(mask.shape[1], left + width + mask_margin)
        bottom = min(mask.shape[0], top + height + mask_margin)
        left, top = max(0, left - mask_margin), max(0, top - mask_margin)
        distance = cv2.distanceTransform(
            255 - mask[top:bottom, left:right], cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        outside = distance > mask_margin
        feather_width = min(mask_margin, DEFAULT_MASK_MARGIN)
        # Reuse the distance buffer rather than allocate full-frame float arrays.
        np.subtract(mask_margin + 1, distance, out=distance)
        distance *= 255.0 / feather_width
        np.clip(distance, 0.0, 255.0, out=distance)
        np.rint(distance, out=distance)
        alpha = distance.astype(np.uint8)
        alpha[outside] = 0
        feathered = np.zeros_like(mask)
        feathered[top:bottom, left:right] = alpha
        return feathered

    # Preserve the old 8px morphology and rounding exactly for existing clients.
    # 팽창 반경 바깥에 빈 픽셀 한 줄이 남는 범위만 계산한다. 범위 밖의 빈 픽셀까지의 거리는
    # 범위 가장자리 빈 픽셀까지의 거리보다 짧을 수 없어 화면 전체로 계산한 값과 같다.
    margin = DEFAULT_MASK_MARGIN + 1
    bottom = min(mask.shape[0], top + height + margin)
    right = min(mask.shape[1], left + width + margin)
    top, left = max(0, top - margin), max(0, left - margin)
    crop = mask[top:bottom, left:right]
    kernel_size = DEFAULT_MASK_MARGIN * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(crop, kernel)
    distance = cv2.distanceTransform(expanded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.minimum(distance / DEFAULT_MASK_MARGIN, 1.0) * 255.0
    feathered = np.zeros_like(mask)
    region = np.rint(alpha).astype(np.uint8)
    region[crop != 0] = 255
    feathered[top:bottom, left:right] = region
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
