"""Whole-frame mask helpers retained for regression comparisons with grouped composition."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from service.mosaic import _blend_layers, _fill_polygons, _protected_polygons, _validate_mask_pad


def mosaic_masked_region(
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


def protected_mask(
    dimensions: tuple[int, int],
    objects: list[dict[str, Any]],
    pad: int = 0,
) -> np.ndarray:
    """Fill protected polygons, then widen them by ``pad`` fully opaque pixels."""
    _validate_mask_pad(pad)
    height, width = dimensions
    return _fill_polygons(dimensions, _protected_polygons(objects, width, height), pad)
