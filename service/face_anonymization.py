"""Shared lazy experimental runtime, with identity state owned by each video RPC."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from service.detection import is_face_object
from service.face_metadata import DEFAULT_METADATA_MODEL, AttributePrediction, MetadataExtractor
from service.face_presets import (
    DEFAULT_PRESET_MANIFEST,
    METADATA_REFRESH_FRAMES,
    PresetCatalog,
    StreamFaceIdentities,
)
from service.inswapper import DEFAULT_ARCFACE, DEFAULT_SWAPPER, DEFAULT_YUNET, InSwapperRenderer
from service.mosaic import (
    _encode_jpeg,
    _encode_yuv420p,
    _feathered_mask,
    _mosaic_masked_region,
    _protected_mask,
)

LOGGER = logging.getLogger("innolive.face_swap")
MODES = {"blur": 0, "face_swap": 1, "face_metadata": 2}


@dataclass(frozen=True, slots=True)
class FaceAnonymizationConfig:
    metadata_model: Path = DEFAULT_METADATA_MODEL
    preset_manifest: Path = DEFAULT_PRESET_MANIFEST
    swapper_model: Path = DEFAULT_SWAPPER
    arcface_model: Path = DEFAULT_ARCFACE
    yunet_model: Path = DEFAULT_YUNET
    metadata_device: str = "cpu"
    swap_provider: str = "cpu"
    retention_frames: int = 30

    def __post_init__(self) -> None:
        if self.swap_provider not in {"cpu", "cuda"}:
            raise ValueError("swap_provider must be cpu or cuda")
        if self.retention_frames < 1:
            raise ValueError("identity retention must be positive")


class FaceAnonymizationRuntime:
    """Used only from the server's bounded, serialized composition executor."""

    def __init__(self, config: FaceAnonymizationConfig):
        self.config = config
        self.extractor: Any | None = None
        self.catalog: PresetCatalog | None = None
        self.renderer: Any | None = None
        self.load_error: str | None = None
        self.renderer_error: str | None = None

    def _metadata(self) -> None:
        if self.load_error:
            raise RuntimeError(self.load_error)
        if self.extractor is not None:
            return
        try:
            self.catalog = PresetCatalog(self.config.preset_manifest)
            self.extractor = MetadataExtractor(
                self.config.metadata_model, self.config.metadata_device
            )
        except Exception:
            self.load_error = "metadata_unavailable"
            LOGGER.exception(
                "experimental metadata runtime unavailable; restart after fixing artifacts"
            )
            raise

    def _renderer(self) -> None:
        if self.renderer_error:
            raise RuntimeError(self.renderer_error)
        if self.renderer is not None:
            return
        try:
            self.renderer = InSwapperRenderer(
                self.config.swapper_model,
                self.config.arcface_model,
                self.config.yunet_model,
                self.config.swap_provider,
            )
        except Exception:
            self.renderer_error = "swapper_unavailable"
            LOGGER.exception("experimental InSwapper unavailable; restart after fixing artifacts")
            raise

    def process(
        self,
        image: np.ndarray,
        objects: list[dict[str, Any]],
        identities: StreamFaceIdentities,
        frame: int,
        mode: int,
        *,
        render: bool,
        pix_fmt: str,
        blur_radius: float,
        pixel_size: int,
        max_bytes: int,
    ) -> bytes:
        identities.expire(
            {int(item["track_id"]) for item in objects if item.get("track_id") is not None}, frame
        )
        output = image.copy()
        remaining = list(objects)
        eligible = []
        crops = []
        refresh_indices = []
        predictions: list[AttributePrediction | None] = []
        for item in objects:
            if not is_face_object(item) or item.get("whitelisted") is True:
                continue
            item["anonymization"] = {
                "status": "blur_fallback",
                "reason": "held_face" if item.get("held") else "metadata_unavailable",
            }
            track_id = item.get("track_id")
            state = identities.tracks.get(track_id)
            if state is not None:
                item["anonymization"]["identity_key"] = state.key
            if item.get("held"):
                continue
            bbox = np.asarray(item["bbox"], dtype=float)
            if bbox.shape != (4,) or not np.isfinite(bbox).all():
                raise ValueError("invalid metadata bbox")
            left, top = np.maximum(np.floor(bbox[:2]), 0).astype(int)
            right, bottom = np.minimum(np.ceil(bbox[2:]), [image.shape[1], image.shape[0]]).astype(
                int
            )
            crop = image[top:bottom, left:right]
            if min(crop.shape[:2]) < 16:
                item["anonymization"]["reason"] = "face_too_small"
                continue
            eligible.append(item)
            cached = identities.metadata.get(track_id)
            if cached is not None and frame - cached.refreshed_at < METADATA_REFRESH_FRAMES:
                predictions.append(cached.prediction)
                continue
            predictions.append(None)
            refresh_indices.append(len(eligible) - 1)
            crops.append(crop)
            if track_id is not None:
                identities.cache_metadata(int(track_id), None, frame)
        if crops and self.load_error is None:
            try:
                self._metadata()
                refreshed = self.extractor.predict(crops)
                if len(refreshed) != len(refresh_indices):
                    raise ValueError("metadata batch size mismatch")
            except Exception:
                LOGGER.warning("metadata extraction failed; applying blur", exc_info=True)
                refreshed = []
            for index, prediction in zip(refresh_indices, refreshed, strict=False):
                predictions[index] = prediction
                track_id = eligible[index].get("track_id")
                if track_id is not None:
                    identities.cache_metadata(int(track_id), prediction, frame)
        for item, prediction in zip(eligible, predictions, strict=True):
            if prediction is None:
                continue
            info = item["anonymization"]
            info.update(attributes=asdict(prediction.attributes), confidence=prediction.confidence)
            track_id = item.get("track_id")
            if track_id is None:
                info["reason"] = "untracked_face"
                continue
            state, variant = identities.observe(int(track_id), prediction.attributes, frame)
            info["identity_key"] = state.key
            preset = self.catalog.match(variant, state.identity)
            if preset is None or not preset.image.is_file():
                info["reason"] = "preset_missing"
                continue
            info["preset_key"] = preset.key
            if mode == MODES["face_metadata"] or not render:
                info.update(status="metadata_only", reason="")
                continue
            if self.renderer_error:
                info["reason"] = "swapper_unavailable"
                continue
            try:
                self._renderer()
                candidate = self.renderer.swap(output.copy(), item, preset)
                if candidate.shape != image.shape or candidate.dtype != np.uint8:
                    raise ValueError("invalid rendered frame")
                output = candidate
                remaining = [entry for entry in remaining if entry is not item]
                info.update(status="swapped", reason="")
            except Exception:
                info["reason"] = "swapper_unavailable" if self.renderer_error else "swap_failed"
                LOGGER.warning("track %s swap failed; applying blur", track_id, exc_info=True)
        if not render:
            return b""
        # Apply failed/held face and number-plate blur last so it takes precedence
        # over nearby successful swaps. Whitelist exclusions use the usual mask.
        mask = _protected_mask(image.shape[:2], remaining)
        output = _mosaic_masked_region(output, _feathered_mask(mask), blur_radius, pixel_size)
        return _encode_yuv420p(output) if pix_fmt else _encode_jpeg(output, max_bytes)
