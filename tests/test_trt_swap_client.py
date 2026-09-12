from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.trt_swap_client.app import (
    InSwapper,
    Settings,
    _blur_objects,
    _iou,
    _mapped_latent,
    _objects,
    _paste_inswapper,
    _prediction_to_bgr,
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


def test_cuda_swapper_settings_keep_ort_memory_cap(tmp_path: Path) -> None:
    settings = Settings(
        detector=tmp_path / "detector.engine",
        swapper=tmp_path / "swapper.onnx",
        swapper_engine=tmp_path / "swapper.engine",
        source=tmp_path / "source.png",
        device="0",
        max_batch=4,
        batch_wait_ms=3.0,
        max_queue=16,
        swap_ort_mem_gib=2.0,
        swap_min_mask_area_px=16_384,
        swapper_backend="cuda",
        swapper_trt_cache=tmp_path / "cache",
        swapper_trt_workspace_gib=1.0,
        target_aligner="yunet_roi",
        target_yunet=tmp_path / "yunet.onnx",
        input_video=None,
        hls_dir=tmp_path / "hls",
    )
    assert settings.swapper_backend == "cuda"
    assert settings.swap_ort_mem_gib == 2.0


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
    swapper.generator = swapper.model
    swapper.source_face = object()
    swapper.yunet = None
    output, succeeded = swapper.apply_many(
        np.zeros((4, 4, 3), dtype=np.uint8), [[0, 0, 20, 20], [40, 40, 60, 60]]
    )
    assert swapper.analysis.calls == 1
    assert succeeded == {0, 1}
    assert np.all(output == 2)


def test_prediction_decoder_uses_official_zero_to_one_output_range() -> None:
    prediction = np.full((1, 3, 128, 128), 0.5, dtype=np.float32)
    decoded = _prediction_to_bgr(prediction)
    assert decoded.shape == (128, 128, 3)
    assert np.all(decoded == 127)


def test_mapped_latent_is_float32_unit_vector() -> None:
    class Source:
        normed_embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)

    latent = _mapped_latent(Source(), np.eye(512, dtype=np.float32))
    assert latent.shape == (1, 512)
    assert latent.dtype == np.float32
    assert np.isclose(np.linalg.norm(latent), 1.0)


def test_paste_back_exposes_mask_and_warp_artifacts() -> None:
    target = np.full((64, 64, 3), 100, dtype=np.uint8)
    aligned = np.full((16, 16, 3), 200, dtype=np.uint8)
    artifacts: dict[str, np.ndarray] = {}
    result = _paste_inswapper(
        target,
        aligned,
        aligned,
        np.asarray(((1, 0, -24), (0, 1, -24)), dtype=np.float32),
        artifacts=artifacts,
    )
    assert result.shape == target.shape
    assert artifacts["swap_mask"].shape == target.shape[:2]
    assert artifacts["inverse_warp_swap"].shape == target.shape
