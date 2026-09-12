from __future__ import annotations

import numpy as np

from experiments.trt_swap_client.app import (
    InSwapper,
    _blur_objects,
    _iou,
    _objects,
    _swap_providers,
)
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


def test_swap_provider_memory_cap() -> None:
    cuda, cpu = _swap_providers("0", 2.0)
    assert cuda[0] == "CUDAExecutionProvider"
    assert cuda[1]["gpu_mem_limit"] == 2 * 1024**3
    assert cuda[1]["arena_extend_strategy"] == "kSameAsRequested"
    assert cpu == "CPUExecutionProvider"


def test_objects_accepts_non_contiguous_segmentation_polygon() -> None:
    polygon = np.asarray(
        [[0, 0], [99, 99], [6, 0], [99, 99], [6, 6], [99, 99], [0, 6]],
        dtype=np.float32,
    )[::2]

    class Masks:
        def __init__(self) -> None:
            self.xy = [polygon]

    class Prediction:
        def __init__(self) -> None:
            self.masks = Masks()
            self.boxes = [object()]

    objects = _objects(
        Prediction(),
        np.asarray([[0, 0, 6, 6, 1, 0.9, 0, 0]], dtype=np.float32),
        {0: "face", 1: "number_plate"},
        8,
        8,
    )
    assert objects[0]["mask_area_px"] == 36.0


def test_swapper_analyzes_one_frame_once_for_multiple_yolo_faces() -> None:
    class Face:
        def __init__(self, bbox: list[float]) -> None:
            self.bbox = np.asarray(bbox, dtype=np.float32)

    class Analysis:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, image: np.ndarray) -> list[Face]:
            self.calls += 1
            return [Face([0, 0, 20, 20]), Face([40, 40, 60, 60])]

    class Model:
        def get(
            self, image: np.ndarray, target: Face, source: object, *, paste_back: bool
        ) -> np.ndarray:
            assert paste_back is True
            return image + 1

    swapper = InSwapper.__new__(InSwapper)
    swapper.analysis = Analysis()
    swapper.model = Model()
    swapper.source_face = object()
    output, succeeded = swapper.apply_many(
        np.zeros((4, 4, 3), dtype=np.uint8), [[0, 0, 20, 20], [40, 40, 60, 60]]
    )
    assert swapper.analysis.calls == 1
    assert succeeded == {0, 1}
    assert np.all(output == 2)
