from __future__ import annotations

import numpy as np

from experiments.trt_swap_client.app import _blur_objects, _iou
from experiments.trt_swap_client.video_io import VideoSpec
from service.mosaic import (
    DEFAULT_BLUR_RADIUS,
    DEFAULT_PIXEL_SIZE,
    _feathered_mask,
    _mosaic_masked_region,
    _protected_mask,
)


def test_fallback_composition_matches_existing_mosaic_mechanism() -> None:
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
    objects = [
        {
            "class_id": 0,
            "class_name": "face",
            "whitelisted": False,
            "mask_polygon": [[8, 8], [42, 8], [42, 42], [8, 42]],
        },
        {
            "class_id": 1,
            "class_name": "number_plate",
            "mask_polygon": [[45, 45], [59, 45], [59, 54], [45, 54]],
        },
    ]
    expected = _mosaic_masked_region(
        image,
        _feathered_mask(_protected_mask(image.shape[:2], objects)),
        DEFAULT_BLUR_RADIUS,
        DEFAULT_PIXEL_SIZE,
    )
    assert np.array_equal(_blur_objects(image, objects), expected)


def test_iou_and_video_spec() -> None:
    assert _iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    assert VideoSpec(1920, 1080, 30.0) == VideoSpec(1920, 1080, 30.0)
