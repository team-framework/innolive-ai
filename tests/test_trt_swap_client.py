from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.trt_swap_client.app import (
    FrameJob,
    InSwapper,
    PreparedFace,
    Settings,
    SwapLab,
    _aligned_seg_mask,
    _blur_objects,
    _find_pending_job_index,
    _group_job_indices,
    _iou,
    _mapped_latent,
    _objects,
    _ort_raw_prediction,
    _paste_inswapper,
    _prediction_to_bgr,
    _require_current_swapper_engine,
    _resolve_swapper_engine,
    _swap_providers,
    _transform_polygon_to_aligned,
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


def test_swapper_engine_manifest_rejects_legacy_or_wrong_model(tmp_path: Path) -> None:
    model, engine = tmp_path / "swapper.onnx", tmp_path / "swapper.engine"
    model.write_bytes(b"model")
    engine.write_bytes(b"engine")
    engine.with_suffix(".engine.json").write_text(
        json.dumps(
            {
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "preserve_onnx_fp32_io": True,
                "precision": "fp32",
            }
        )
    )
    _require_current_swapper_engine(engine, model)


def test_swapper_engine_manifest_accepts_mixed_with_provenance(tmp_path: Path) -> None:
    official, engine = tmp_path / "swapper.onnx", tmp_path / "swapper.engine"
    official.write_bytes(b"model")
    engine.write_bytes(b"engine")
    engine.with_suffix(".engine.json").write_text(
        json.dumps(
            {
                "model_sha256": "mixed-onnx-hash",
                "base_model_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
                "mixed_recipe": {"converted_convs": ["Conv_42"]},
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "preserve_onnx_fp32_io": True,
                "precision": "fp32",
            }
        )
    )
    _require_current_swapper_engine(engine, official)


def test_swapper_engine_manifest_rejects_mixed_without_provenance(tmp_path: Path) -> None:
    official, engine = tmp_path / "swapper.onnx", tmp_path / "swapper.engine"
    official.write_bytes(b"model")
    engine.write_bytes(b"engine")
    engine.with_suffix(".engine.json").write_text(
        json.dumps(
            {
                "model_sha256": "other-hash",
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "preserve_onnx_fp32_io": True,
                "precision": "fp32",
            }
        )
    )
    with pytest.raises(RuntimeError, match="different ONNX model"):
        _require_current_swapper_engine(engine, official)


def test_ort_reference_does_not_apply_a_second_input_normalization() -> None:
    class Session:
        def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert names == ["output"]
            assert inputs["target"][0, 0, 0, 0] == 0.5
            assert inputs["source"][0, 0] == 1.0
            return [inputs["target"]]

    metadata = type(
        "Metadata",
        (),
        {"session": Session(), "output_names": ["output"], "input_names": ["target", "source"]},
    )()
    blob = np.full((1, 3, 128, 128), 0.5, dtype=np.float32)
    latent = np.ones((1, 512), dtype=np.float32)
    assert np.array_equal(_ort_raw_prediction(metadata, blob, latent), blob)


def test_transform_polygon_to_aligned_uses_affine_matrix() -> None:
    polygon = np.asarray([[10.0, 20.0], [30.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(_transform_polygon_to_aligned(polygon, identity), polygon)
    shifted = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    converted = _transform_polygon_to_aligned(polygon, shifted)
    assert np.allclose(converted, polygon + np.asarray([5.0, -3.0], dtype=np.float32))


def test_aligned_seg_mask_keeps_interior_and_feathers_boundary() -> None:
    polygon = np.asarray([[10.0, 10.0], [118.0, 10.0], [118.0, 118.0], [10.0, 118.0]])
    mask = _aligned_seg_mask(polygon)
    assert mask is not None
    assert mask.shape == (128, 128)
    assert mask.dtype == np.float32
    assert float(mask[64, 64]) == 1.0
    assert float(mask[0, 0]) == 0.0
    edge = mask[10, 64]
    assert 0.0 <= float(edge) <= 1.0
    assert _aligned_seg_mask(np.asarray([[0.0, 0.0], [1.0, 1.0]])) is None


def test_paste_inswapper_with_yolo_mask_preserves_background_outside() -> None:
    target = np.full((64, 64, 3), 100, dtype=np.uint8)
    aligned = np.full((16, 16, 3), 200, dtype=np.uint8)
    fake = np.full((16, 16, 3), 250, dtype=np.uint8)
    matrix = np.asarray(((1, 0, -24), (0, 1, -24)), dtype=np.float32)
    plain = _paste_inswapper(target, aligned, fake, matrix)
    assert plain[30, 30].mean() > 150
    seg = np.zeros((16, 16), dtype=np.float32)
    seg[:, :8] = 1.0
    constrained = _paste_inswapper(target, aligned, fake, matrix, seg_mask_aligned=seg)
    # Left half of the warped ROI follows the swap, right half stays original.
    assert constrained[30, 27].mean() > 150
    assert np.array_equal(constrained[30, 37], np.asarray([100, 100, 100], dtype=np.uint8))


def _batched_swapper() -> tuple[InSwapper, Any]:
    from types import SimpleNamespace

    class Face:
        def __init__(self, bbox: list[float]) -> None:
            self.bbox = np.asarray(bbox, dtype=np.float32)
            self.kps = np.asarray([[5, 5], [15, 5], [10, 10], [5, 15], [15, 15]], dtype=np.float32)

    class BatchGenerator:
        def __init__(self) -> None:
            self.seen_blobs: list[np.ndarray] = []
            self.seen_latents: list[np.ndarray] = []

        def forward_batch(self, blobs: np.ndarray, latents: np.ndarray) -> np.ndarray:
            assert blobs.shape[0] == latents.shape[0]
            self.seen_blobs.append(blobs.copy())
            self.seen_latents.append(latents.copy())
            count = blobs.shape[0]
            predictions = np.zeros((count, 3, 128, 128), dtype=np.float32)
            # Position-independent: a face prediction depends only on its own
            # inputs, like the production generator.
            for row in range(count):
                predictions[row] = 0.1 + 0.01 * float(latents[row, 0])
            return predictions

    swapper = InSwapper.__new__(InSwapper)
    swapper.analysis = SimpleNamespace(get=lambda image: [])
    swapper.model = SimpleNamespace(
        emap=np.eye(512, dtype=np.float32),
        input_size=(128, 128),
        input_std=255.0,
        input_mean=0.0,
    )
    generator = BatchGenerator()
    swapper.generator = generator
    swapper.source_face = SimpleNamespace(
        normed_embedding=np.ones(512, dtype=np.float32) / np.sqrt(512.0)
    )
    swapper.yunet = None
    swapper._cached_source_latent = None

    def _fake_align(frame: np.ndarray, kps: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((128, 128, 3), 100, dtype=np.uint8),
            np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        )

    swapper._align_target = _fake_align  # type: ignore[method-assign]
    faces = [Face([0, 0, 20, 20]), Face([40, 40, 60, 60])]

    def _fake_targets(frame: np.ndarray, boxes: list[list[float]]) -> dict[int, Face]:
        return {index: faces[index] for index in range(len(boxes))}

    swapper._target_faces = _fake_targets  # type: ignore[method-assign]
    return swapper, generator


def test_apply_many_batches_faces_with_order_and_timing() -> None:
    swapper, generator = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    output, succeeded = swapper.apply_many(
        frame,
        [[0, 0, 20, 20], [40, 40, 60, 60]],
        [[[0, 0], [63, 0], [63, 63], [0, 63]]] * 2,
    )
    assert succeeded == {0, 1}
    assert output.shape == frame.shape
    # The multi-face pipeline forwards once per face; both rows share the
    # single source latent (call order across threads is not asserted).
    assert len(generator.seen_blobs) == 2
    assert all(blob.shape == (1, 3, 128, 128) for blob in generator.seen_blobs)
    assert len(generator.seen_latents) == 2
    assert np.array_equal(generator.seen_latents[0], generator.seen_latents[1])
    assert swapper.last_batch_size == 2
    assert swapper.last_face_count == 2
    assert set(swapper.last_generator_timing) == {"prepare", "forward", "paste"}
    swapper.close()


def test_apply_many_per_face_latent_mapping_preserves_order() -> None:
    swapper, generator = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    latent_a = np.full((1, 512), 0.25, dtype=np.float32)
    latent_b = np.full((1, 512), 0.75, dtype=np.float32)
    output, succeeded = swapper.apply_many(
        frame,
        [[0, 0, 20, 20], [40, 40, 60, 60]],
        None,
        {0: latent_a, 1: latent_b},
    )
    assert succeeded == {0, 1}
    assert output.shape == frame.shape
    seen = sorted(
        (np.asarray(call).tobytes() for call in generator.seen_latents),
    )
    assert seen == sorted((latent_a.tobytes(), latent_b.tobytes()))
    swapper.close()


def test_find_pending_job_index_prefers_newest_same_stream() -> None:
    from types import SimpleNamespace

    stream_a, stream_b = object(), object()
    pending = [
        SimpleNamespace(stream=stream_a),
        SimpleNamespace(stream=stream_b),
        SimpleNamespace(stream=stream_a),
    ]
    assert _find_pending_job_index(pending, stream_a) == 2
    assert _find_pending_job_index(pending, stream_b) == 1
    assert _find_pending_job_index(pending, object()) is None


def test_group_job_indices_keeps_per_stream_order() -> None:
    from types import SimpleNamespace

    stream_a, stream_b = object(), object()
    jobs = [
        SimpleNamespace(stream=stream_a),
        SimpleNamespace(stream=stream_b),
        SimpleNamespace(stream=stream_a),
        SimpleNamespace(stream=stream_b),
    ]
    assert _group_job_indices(jobs) == [[0, 2], [1, 3]]
    assert _group_job_indices(jobs[:1]) == [[0]]
    assert _group_job_indices([]) == []


def test_concurrent_apply_many_keeps_per_thread_results() -> None:
    import threading

    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    results: dict[int, tuple[np.ndarray, set[int]]] = {}
    errors: list[BaseException] = []

    def run(slot: int, boxes: list[list[float]]) -> None:
        try:
            results[slot] = swapper.apply_many(frame, boxes)
        except BaseException as error:
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(0, [[0, 0, 20, 20]])),
        threading.Thread(target=run, args=(1, [[0, 0, 20, 20], [40, 40, 60, 60]])),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    assert not errors
    assert results[0][1] == {0}
    assert results[1][1] == {0, 1}
    assert results[0][0].shape == frame.shape
    assert results[1][0].shape == frame.shape
    swapper.close()


def test_resolve_swapper_engine_prefers_mixed_when_built(tmp_path: Path) -> None:
    import experiments.trt_swap_client.app as app_module

    stock = tmp_path / "stock.engine"
    stock.write_bytes(b"engine")
    mixed = tmp_path / "mixed.engine"
    mixed.write_bytes(b"engine")
    other = tmp_path / "other.engine"
    other.write_bytes(b"engine")
    real_default, real_mixed = app_module.DEFAULT_SWAPPER_ENGINE, app_module.MIXED_SWAPPER_ENGINE
    app_module.DEFAULT_SWAPPER_ENGINE = stock
    try:
        app_module.MIXED_SWAPPER_ENGINE = tmp_path / "missing.engine"
        assert _resolve_swapper_engine(stock) == stock.resolve()
        app_module.MIXED_SWAPPER_ENGINE = mixed
        assert _resolve_swapper_engine(stock) == mixed.resolve()
        assert _resolve_swapper_engine(other) == other.resolve()
    finally:
        app_module.DEFAULT_SWAPPER_ENGINE = real_default
        app_module.MIXED_SWAPPER_ENGINE = real_mixed


def test_parallel_yunet_uses_clones_and_keeps_index_order() -> None:
    from types import SimpleNamespace

    import experiments.trt_swap_client.app as app_module

    swapper, _ = _batched_swapper()
    swapper.yunet = object()
    clones = [object(), object()]
    swapper._yunet_clones = clones
    # _batched_swapper stubs _target_faces; restore the real method.
    del swapper._target_faces
    used: list[int] = []
    real = app_module._target_from_yunet_with

    def recording(ynet: object, frame: np.ndarray, box: list[float]) -> object:
        used.append(id(ynet))
        return SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32))

    app_module._target_from_yunet_with = recording  # type: ignore[method-assign]
    try:
        targets = swapper._target_faces(
            np.zeros((64, 64, 3), dtype=np.uint8), [[0, 0, 20, 20], [40, 40, 60, 60]]
        )
    finally:
        app_module._target_from_yunet_with = real  # type: ignore[method-assign]
    assert sorted(targets) == [0, 1]
    assert sorted(used) == sorted(id(clone) for clone in clones)
    swapper.close()


def test_pipelined_multi_face_matches_sequential_helpers() -> None:
    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [[0, 0, 20, 20], [40, 40, 60, 60]]
    pipelined, succeeded = swapper.apply_many(frame, boxes)
    assert succeeded == {0, 1}
    targets = swapper._target_faces(frame, boxes)
    prepared, _ = swapper._prepare_faces(frame, targets, None, None)
    fakes, _ = swapper._forward_prepared(prepared)
    expected, _, _ = swapper._paste_prepared(frame, prepared, fakes)
    assert np.array_equal(pipelined, expected)
    swapper.close()


def test_pipelined_prepare_failure_fails_whole_frame() -> None:
    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [[0, 0, 20, 20], [40, 40, 60, 60]]
    bad = np.full((1, 512), np.nan, dtype=np.float32)
    output, succeeded = swapper.apply_many(
        frame, boxes, None, {0: bad, 1: np.ones((1, 512), dtype=np.float32)}
    )
    assert succeeded == set()
    assert np.array_equal(output, frame)
    swapper.close()


def test_pipelined_render_failure_skips_only_that_face() -> None:
    import experiments.trt_swap_client.app as app_module

    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = [[0, 0, 20, 20], [40, 40, 60, 60]]
    real_render = app_module._render_face_layer
    calls = {"count": 0}

    def flaky_render(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("warp failed")
        return real_render(*args, **kwargs)

    app_module._render_face_layer = flaky_render  # type: ignore[method-assign]
    try:
        output, succeeded = swapper.apply_many(frame, boxes)
    finally:
        app_module._render_face_layer = real_render  # type: ignore[method-assign]
    assert len(succeeded) == 1
    assert output.shape == frame.shape
    swapper.close()


def _stub_lab(max_queue: int = 2) -> SwapLab:
    from types import SimpleNamespace

    lab = SwapLab.__new__(SwapLab)
    lab.settings = SimpleNamespace(max_batch=4, batch_wait_ms=3.0, max_queue=max_queue)
    lab.queue = asyncio.Queue(maxsize=max_queue)
    lab.dropped_frames = 0
    return lab


def test_submit_coalesces_pending_same_stream_frame_when_full() -> None:
    async def scenario() -> None:
        lab = _stub_lab(max_queue=2)
        loop = asyncio.get_running_loop()
        stream_a, stream_b = object(), object()
        first_a: asyncio.Future = loop.create_future()
        first_b: asyncio.Future = loop.create_future()
        old_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        lab.queue.put_nowait(FrameJob(old_frame, stream_a, first_a))
        lab.queue.put_nowait(FrameJob(old_frame, stream_b, first_b))
        new_frame = np.full((4, 4, 3), 7, dtype=np.uint8)
        pending = asyncio.ensure_future(lab.submit(new_frame, stream_a))
        await asyncio.sleep(0)
        # The queued job now carries the newest frame and both callers share it.
        assert lab.queue.qsize() == 2
        assert next(iter(list(lab.queue._queue))).frame is new_frame  # type: ignore[attr-defined]
        first_a.set_result((new_frame, {"ok": True}))
        output, _ = await pending
        assert np.array_equal(output, new_frame)
        assert lab.dropped_frames == 0

    asyncio.run(scenario())


def test_submit_evicts_oldest_when_full_without_same_stream() -> None:
    async def scenario() -> None:
        lab = _stub_lab(max_queue=2)
        loop = asyncio.get_running_loop()
        stream_a, stream_b, stream_c = object(), object(), object()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        future_b: asyncio.Future = loop.create_future()
        future_c: asyncio.Future = loop.create_future()
        lab.queue.put_nowait(FrameJob(frame, stream_b, future_b))
        lab.queue.put_nowait(FrameJob(frame, stream_c, future_c))
        pending = asyncio.ensure_future(lab.submit(frame, stream_a))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="superseded"):
            await future_b
        assert lab.dropped_frames == 1
        assert lab.queue.qsize() == 2
        for job in list(lab.queue._queue):  # type: ignore[attr-defined]
            if job.stream is stream_a:
                job.future.set_result((frame, {"ok": True}))
        output, _ = await pending
        assert np.array_equal(output, frame)

    asyncio.run(scenario())


def test_forward_batch_keeps_distinct_rows_with_reused_buffer() -> None:
    """Per-row forwards may reuse one buffer; stacked rows must not alias it."""

    from types import SimpleNamespace

    class AliasingGenerator:
        def __init__(self) -> None:
            self.buffer = np.zeros((1, 3, 128, 128), dtype=np.float32)

        def forward(self, blob: np.ndarray, latent: np.ndarray) -> np.ndarray:
            self.buffer[0] = float(latent[0, 0])
            return self.buffer

    swapper, _ = _batched_swapper()
    swapper.generator = AliasingGenerator()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    latent_a = np.full((1, 512), 0.25, dtype=np.float32)
    latent_b = np.full((1, 512), 0.75, dtype=np.float32)
    prepared, _ = swapper._prepare_faces(
        frame,
        {0: SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32))},
        None,
        None,
    )
    assert len(prepared) == 1
    template = prepared[0]
    # Sanity: raw per-row views into one buffer really do alias.
    swapper.generator.buffer[0] = 0.0
    first = np.array(swapper.generator.forward(template.blob, latent_a), copy=False)
    second = np.array(swapper.generator.forward(template.blob, latent_b), copy=False)
    assert np.all(first == second)
    fakes, _ = swapper._forward_prepared(
        [
            PreparedFace(
                index=0,
                target=template.target,
                aimg=template.aimg,
                matrix=template.matrix,
                blob=template.blob,
                latent=latent_a,
            ),
            PreparedFace(
                index=1,
                target=template.target,
                aimg=template.aimg,
                matrix=template.matrix,
                blob=template.blob,
                latent=latent_b,
            ),
        ]
    )
    assert not np.array_equal(fakes[0], fakes[1])


def test_paste_prepared_blends_all_faces_into_one_canvas() -> None:
    from types import SimpleNamespace

    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    prepared, _ = swapper._prepare_faces(
        frame,
        {
            0: SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32)),
            1: SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32)),
        },
        None,
        None,
    )
    fakes = [np.full((128, 128, 3), 200, dtype=np.uint8)] * 2
    output, succeeded, _ = swapper._paste_prepared(frame, prepared, fakes)
    assert succeeded == {0, 1}
    assert output.shape == frame.shape
    # The caller's frame is never mutated in place.
    assert np.array_equal(frame, np.zeros((64, 64, 3), dtype=np.uint8))


def test_paste_prepared_matches_sequential_for_disjoint_rois() -> None:
    from types import SimpleNamespace

    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 512, 3), dtype=np.uint8)
    left = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    right = np.asarray([[1, 0, -384], [0, 1, 0]], dtype=np.float32)
    aimg = np.full((128, 128, 3), 100, dtype=np.uint8)
    prepared = [
        PreparedFace(
            index=0,
            target=SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32)),
            aimg=aimg,
            matrix=left,
            blob=np.zeros((1, 3, 128, 128), dtype=np.float32),
            latent=np.zeros((1, 512), dtype=np.float32),
        ),
        PreparedFace(
            index=1,
            target=SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32)),
            aimg=aimg,
            matrix=right,
            blob=np.zeros((1, 3, 128, 128), dtype=np.float32),
            latent=np.zeros((1, 512), dtype=np.float32),
        ),
    ]
    fakes = [
        np.full((128, 128, 3), 200, dtype=np.uint8),
        np.full((128, 128, 3), 50, dtype=np.uint8),
    ]
    batched, succeeded_batched, _ = swapper._paste_prepared(frame, prepared, fakes)
    assert succeeded_batched == {0, 1}
    expected = frame.copy()
    for item, fake in zip(prepared, fakes, strict=True):
        expected = _paste_inswapper(
            expected, item.aimg, fake, item.matrix, seg_mask_aligned=item.seg_mask_aligned
        )
    assert np.array_equal(batched, expected)
    # Each half actually received its own face.
    assert batched[32, 64].mean() > 100
    assert batched[32, 448].mean() < 100


def test_paste_prepared_matches_sequential_for_overlapping_rois() -> None:
    from types import SimpleNamespace

    swapper, _ = _batched_swapper()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    matrix = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    aimg = np.full((128, 128, 3), 100, dtype=np.uint8)
    prepared = [
        PreparedFace(
            index=index,
            target=SimpleNamespace(kps=np.zeros((5, 2), dtype=np.float32)),
            aimg=aimg,
            matrix=matrix,
            blob=np.zeros((1, 3, 128, 128), dtype=np.float32),
            latent=np.zeros((1, 512), dtype=np.float32),
        )
        for index in range(2)
    ]
    fakes = [
        np.full((128, 128, 3), 200, dtype=np.uint8),
        np.full((128, 128, 3), 50, dtype=np.uint8),
    ]
    batched, succeeded_batched, _ = swapper._paste_prepared(frame, prepared, fakes)
    assert succeeded_batched == {0, 1}
    expected = frame.copy()
    for item, fake in zip(prepared, fakes, strict=True):
        expected = _paste_inswapper(
            expected, item.aimg, fake, item.matrix, seg_mask_aligned=item.seg_mask_aligned
        )
    assert np.array_equal(batched, expected)


def test_yunet_landmarks_use_columns_4_to_14() -> None:
    """YuNet rows are [box(4), landmarks(10), score]; [5:15] shifts into the score."""

    import sys
    import types

    class FakeYuNet:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = None  # type: ignore[assignment]

        def setInputSize(self, size: tuple[int, int]) -> None:
            self.sizes = size

        def detect(self, roi: np.ndarray) -> tuple[None, np.ndarray]:
            row = np.asarray(
                [[10, 10, 100, 100, 20, 30, 40, 30, 30, 50, 25, 70, 35, 70, 0.9]],
                dtype=np.float32,
            )
            return None, row

    class FakeFace:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    package = types.ModuleType("insightface")
    app_module = types.ModuleType("insightface.app")
    common_module = types.ModuleType("insightface.app.common")
    common_module.Face = FakeFace  # type: ignore[attr-defined]
    saved = {
        name: sys.modules[name] for name in list(sys.modules) if name.startswith("insightface")
    }
    sys.modules["insightface"] = package
    sys.modules["insightface.app"] = app_module
    sys.modules["insightface.app.common"] = common_module
    try:
        swapper = InSwapper.__new__(InSwapper)
        swapper.yunet = FakeYuNet()
        face = swapper._target_from_yunet(
            np.zeros((200, 200, 3), dtype=np.uint8), [50, 50, 150, 150]
        )
    finally:
        for name in [n for n in sys.modules if n.startswith("insightface")]:
            del sys.modules[name]
        sys.modules.update(saved)
    assert face is not None
    assert swapper.yunet.sizes == (170, 170)
    assert np.array_equal(
        face.kps,
        np.asarray([[35, 45], [55, 45], [45, 65], [40, 85], [50, 85]], dtype=np.float32),
    )
