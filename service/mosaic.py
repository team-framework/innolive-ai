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
# 축소 이미지 Gaussian sigma 상한. 기본값(24 / 2)과 같아 sigma가 이 값 이하인 설정은 결과가 그대로다.
MAX_REDUCED_SIGMA = 12.0
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

    output = _compose_protected(image, objects, blur_radius, pixel_size, mask_margin, mask_pad)

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

    output = _compose_protected(image, objects, blur_radius, pixel_size, mask_margin, mask_pad)

    return _encode_yuv420p(output)


def _compose_protected(
    image: np.ndarray,
    objects: list[dict[str, Any]],
    blur_radius: float,
    pixel_size: int,
    mask_margin: int,
    mask_pad: int,
) -> np.ndarray:
    """Blur every protected object, computing each group of nearby objects in its own crop.

    The result equals one whole-frame mask and blur pass. Objects farther apart than their mask
    reach cannot change each other's blend weights, and blurred pixels come from the
    frame-anchored blur, so far-apart faces no longer pull the empty space between them into
    the work area.
    """
    _validate_mask_pad(mask_pad)
    height, width = image.shape[:2]
    polygons = _protected_polygons(objects, width, height)
    mask_margin = validate_mask_margin(mask_margin)
    # 보호 픽셀에서 pad만큼 팽창하고 margin만큼 페더가 번지며, 페더 계산은 한 픽셀을 더 본다.
    reach = mask_pad + mask_margin + 1
    layers = []
    for left, top, right, bottom, members in _group_polygons(polygons, reach, width, height):
        offset = np.array((left, top), dtype=np.int32)
        mask = _fill_polygons(
            (bottom - top, right - left), [polygon - offset for polygon in members], mask_pad
        )
        alpha = _feathered_mask(mask, mask_margin)
        x, y, alpha_width, alpha_height = cv2.boundingRect(alpha)
        if alpha_width:
            layers.append((left + x, top + y, alpha[y : y + alpha_height, x : x + alpha_width]))
    return _blend_layers(image, layers, blur_radius, pixel_size)


def _mosaic_masked_region(
    image: np.ndarray,
    blend_mask: np.ndarray,
    blur_radius: float,
    pixel_size: int,
) -> np.ndarray:
    """Blend the frame-anchored blur into the protected bounds of one whole-frame mask."""

    mask_left, mask_top, mask_width, mask_height = cv2.boundingRect(blend_mask)
    if not mask_width:
        return image
    alpha = blend_mask[mask_top : mask_top + mask_height, mask_left : mask_left + mask_width]
    return _blend_layers(image, [(mask_left, mask_top, alpha)], blur_radius, pixel_size)


def _blend_layers(
    image: np.ndarray,
    layers: list[tuple[int, int, np.ndarray]],
    blur_radius: float,
    pixel_size: int,
) -> np.ndarray:
    """Blend blurred pixels into each (left, top, alpha) layer; layers must not overlap."""

    if not layers:
        return image
    height, width = image.shape[:2]
    cell = _reduction_cell(blur_radius, pixel_size)
    support = _blur_support(blur_radius, cell)
    boxes = [
        (left, top, left + alpha.shape[1], top + alpha.shape[0]) for left, top, alpha in layers
    ]
    union = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )

    def blur_area(box: tuple[int, int, int, int]) -> int:
        left, top, right, bottom = box
        return (min(width, right + support) - max(0, left - support)) * (
            min(height, bottom + support) - max(0, top - support)
        )

    # 레이어별 블러 영역의 합이 전체를 한 번 블러하는 영역보다 크면(밀집한 얼굴) 한 번에 계산한다.
    # 블러 값은 계산 영역과 무관하므로 어느 쪽이든 결과가 같다.
    shared = None
    if sum(blur_area(box) for box in boxes) >= blur_area(union):
        shared = _frame_blur(image, union, blur_radius, cell)

    output = image.copy()
    for (_, _, alpha), box in zip(layers, boxes, strict=True):
        left, top, right, bottom = box
        if shared is None:
            blurred = _frame_blur(image, box, blur_radius, cell)
        else:
            blurred = shared[top - union[1] : bottom - union[1], left - union[0] : right - union[0]]
        # Complementary weights bound the rounded numerator to 255 * 255 + 127.
        # uint16 preserves the exact result while halving blend-buffer bandwidth.
        weight = alpha[:, :, None].astype(np.uint16)
        output[top:bottom, left:right] = (
            image[top:bottom, left:right].astype(np.uint16) * (255 - weight)
            + blurred.astype(np.uint16) * weight
            + 127
        ) // 255
    return output


def _reduction_cell(blur_radius: float, pixel_size: int) -> int:
    """Return the reduction cell size, raised so the reduced sigma stays within the cap.

    Gaussian cost grows with the reduced sigma, while a coarser cell under the same radius changes
    the blurred pixels by only a level or two, so large radii on fine cells use a coarser cell.
    """

    return max(pixel_size, math.ceil(blur_radius / MAX_REDUCED_SIGMA))


def _blur_support(blur_radius: float, pixel_size: int) -> int:
    """Frame pixels on each side that can change one blurred pixel."""

    # OpenCV 8비트 Gaussian 커널 반경은 ceil(3 * sigma) 이하다. 축소 셀과 선형 보간 이웃, 여유 한 셀을 더한다.
    return pixel_size * (math.ceil(3 * blur_radius / pixel_size) + 3)


def _frame_blur(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    blur_radius: float,
    pixel_size: int,
) -> np.ndarray:
    """Return blurred pixels for ``box`` exactly as one whole-frame blur would produce them.

    The reduction grid is anchored to frame coordinates and the crop keeps every source pixel
    that the reduction, Gaussian, and interpolation can reach, so the values never depend on
    where the crop starts. The frame's right and bottom edges are padded by replication to whole
    reduction cells.
    """

    left, top, right, bottom = box
    height, width = image.shape[:2]
    support = _blur_support(blur_radius, pixel_size)
    crop_top = max(0, top - support) // pixel_size * pixel_size
    crop_left = max(0, left - support) // pixel_size * pixel_size
    crop_bottom = min(height, -(-(bottom + support) // pixel_size) * pixel_size)
    crop_right = min(width, -(-(right + support) // pixel_size) * pixel_size)
    region = image[crop_top:crop_bottom, crop_left:crop_right]
    pad_bottom, pad_right = -region.shape[0] % pixel_size, -region.shape[1] % pixel_size
    if pad_bottom or pad_right:
        region = cv2.copyMakeBorder(region, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
    reduced = cv2.resize(
        region,
        (region.shape[1] // pixel_size, region.shape[0] // pixel_size),
        interpolation=cv2.INTER_AREA,
    )
    reduced_sigma = blur_radius / pixel_size
    blurred = cv2.resize(
        cv2.GaussianBlur(reduced, (0, 0), reduced_sigma, reduced_sigma),
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return blurred[top - crop_top : bottom - crop_top, left - crop_left : right - crop_left]


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
    _validate_mask_pad(pad)
    height, width = dimensions
    return _fill_polygons(dimensions, _protected_polygons(objects, width, height), pad)


def _validate_mask_pad(pad: Any) -> None:
    if isinstance(pad, bool) or not isinstance(pad, int) or not 0 <= pad <= MAX_MASK_PAD:
        raise ValueError(f"mask pad must be an integer in 0..{MAX_MASK_PAD}")


def _protected_polygons(objects: list[dict[str, Any]], width: int, height: int) -> list[np.ndarray]:
    """Validate the polygons that must be blurred; whitelisted faces are skipped, plates never."""
    return [
        _polygon(item.get("mask_polygon"), width, height)
        for item in objects
        if item.get("whitelisted") is not True or is_number_plate_object(item)
    ]


def _group_polygons(
    polygons: list[np.ndarray], reach: int, width: int, height: int
) -> list[tuple[int, int, int, int, list[np.ndarray]]]:
    """Group polygons whose bounds widened by ``reach`` overlap; return each group's crop.

    Crops of different groups never overlap, so no protected pixel of one group lies within
    ``reach`` of another group.
    """
    groups: list[tuple[int, int, int, int, list[np.ndarray]]] = []
    for polygon in polygons:
        x, y, box_width, box_height = cv2.boundingRect(polygon)
        left, top = max(0, x - reach), max(0, y - reach)
        right, bottom = min(width, x + box_width + reach), min(height, y + box_height + reach)
        members = [polygon]
        # 합쳐서 커진 영역이 다른 그룹과 다시 겹칠 수 있으므로 겹치는 그룹이 없을 때까지 흡수한다.
        merged = True
        while merged:
            merged = False
            for index, (g_left, g_top, g_right, g_bottom, g_members) in enumerate(groups):
                if left < g_right and g_left < right and top < g_bottom and g_top < bottom:
                    del groups[index]
                    left, top = min(left, g_left), min(top, g_top)
                    right, bottom = max(right, g_right), max(bottom, g_bottom)
                    members = g_members + members
                    merged = True
                    break
        groups.append((left, top, right, bottom, members))
    return groups


def _fill_polygons(
    dimensions: tuple[int, int],
    polygons: list[np.ndarray],
    pad: int,
) -> np.ndarray:
    """Fill validated polygons, then widen them by ``pad`` fully opaque pixels."""
    height, width = dimensions
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
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
