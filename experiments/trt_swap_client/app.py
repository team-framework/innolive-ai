#!/usr/bin/env python3
"""WebSocket test client for batched YOLO face swap with fail-closed blur."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from experiments.trt_swap_client.video_io import (
    NvdecReader,
    NvencHlsWriter,
    probe_video,
    require_nvcodec_ffmpeg,
)
from service.adaface_model import DEFAULT_FACE_DETECTOR, AdaFaceConfig, AdaFaceRuntime
from service.detection import is_face_object, is_number_plate_object
from service.mosaic import DEFAULT_BLUR_RADIUS, DEFAULT_PIXEL_SIZE, MASK_FEATHER_RADIUS
from service.recognition import RecognitionConfig, SessionRegistry, StreamRecognition
from service.runtime import IMAGE_SIZE, MAX_DETECTIONS, MAX_POLYGON_POINTS
from service.tracking import DETECTOR_CONFIDENCE, StreamTracker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = ROOT / "models" / "best_swap_b4.engine"
DEFAULT_DETECTOR_CHECKPOINT = ROOT / "models" / "best.pt"
DEFAULT_SOURCE = Path.home() / "Documents" / "input.png"
DEFAULT_SWAPPER = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
DEFAULT_SWAPPER_ENGINE = ROOT / "models" / "face_swap" / "inswapper_128_trt11_fp32.engine"
DEFAULT_HLS_DIR = ROOT / "face_swap_lab_output" / "trt_hls"
DEFAULT_SWAPPER_TRT_CACHE = ROOT / "face_swap_lab_output" / "trt_swapper_cache"


@dataclass(frozen=True, slots=True)
class Settings:
    detector: Path
    swapper: Path
    swapper_engine: Path
    source: Path
    device: str
    max_batch: int
    batch_wait_ms: float
    max_queue: int
    swap_ort_mem_gib: float
    swap_min_mask_area_px: float
    swapper_backend: str
    swapper_trt_cache: Path
    swapper_trt_workspace_gib: float
    target_aligner: str
    target_yunet: Path
    input_video: Path | None
    hls_dir: Path
    swap_debug_dir: Path | None = None
    swap_debug_frames: int = 1
    stream_jpeg_quality: int = 100


@dataclass(slots=True)
class StreamState:
    session_id: str
    tracker: StreamTracker
    recognition: StreamRecognition
    sequence: int = 0


@dataclass(slots=True)
class FrameJob:
    frame: np.ndarray
    stream: StreamState
    future: asyncio.Future[tuple[np.ndarray, dict[str, Any]]]


class LazyAdaFaceRuntime:
    """Do not reserve AdaFace VRAM until a session actually enrolls a face."""

    def __init__(self, config: AdaFaceConfig, *, fallback_device: str):
        self._config = config
        self._fallback_device = fallback_device
        self._runtime: AdaFaceRuntime | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._runtime is not None and self._runtime.ready

    @property
    def load_error(self) -> str | None:
        return None if self._runtime is None else self._runtime.load_error

    def ensure(self) -> None:
        with self._lock:
            if self._runtime is None:
                self._runtime = AdaFaceRuntime(self._config, fallback_device=self._fallback_device)
            if not self._runtime.ready:
                raise RuntimeError(f"AdaFace unavailable: {self._runtime.load_error}")

    def submit(self, image: np.ndarray, *, owner: str) -> asyncio.Future[np.ndarray] | None:
        return None if self._runtime is None else self._runtime.submit(image, owner=owner)

    def submit_enrollment(
        self, image: np.ndarray, *, owner: str
    ) -> asyncio.Future[np.ndarray] | None:
        return (
            None if self._runtime is None else self._runtime.submit_enrollment(image, owner=owner)
        )

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()


def _swap_providers(device: str, memory_gib: float) -> list[Any]:
    """Bound ONNX Runtime's per-session CUDA arena without changing model outputs."""

    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": int(device),
                "gpu_mem_limit": int(memory_gib * 1024**3),
                "arena_extend_strategy": "kSameAsRequested",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]


def _require_current_swapper_engine(engine_path: Path, model_path: Path) -> None:
    """Reject legacy full-FP16 ONNX conversion engines before they can blur output."""

    manifest_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not manifest_path.is_file():
        raise RuntimeError(
            f"TensorRT swap engine manifest is missing: {manifest_path}. "
            "Rebuild the engine with python -m experiments.trt_swap_client.export_swapper --force"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid TensorRT swap engine manifest: {manifest_path}") from error
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if manifest.get("model_sha256") != model_hash:
        raise RuntimeError("TensorRT swap engine was built from a different ONNX model; rebuild it")
    if manifest.get("preserve_onnx_fp32_io") is not True:
        raise RuntimeError(
            "legacy TensorRT swap engine converted ONNX I/O to FP16; rebuild it with the current exporter"
        )
    if manifest.get("precision") != "fp32":
        raise RuntimeError(
            "InSwapper FP16 TensorRT output has unacceptable raw error; rebuild with --precision fp32"
        )
    engine_hash = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if manifest.get("engine_sha256") != engine_hash:
        raise RuntimeError("TensorRT swap engine does not match its manifest; rebuild it")


class TensorRtInSwapperGenerator:
    """TensorRT 11 runner that preserves InsightFace's crop and paste-back math."""

    def __init__(self, metadata: Any, engine_path: Path, device: str):
        if not engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT swap engine is missing: {engine_path}. Build it with "
                "python -m experiments.trt_swap_client.export_swapper"
            )
        import tensorrt as trt

        self.metadata = metadata
        self.device = int(device)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT swap engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.inputs = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        ]
        image_inputs = [name for name in self.inputs if len(self.engine.get_tensor_shape(name)) == 4]
        latent_inputs = [name for name in self.inputs if len(self.engine.get_tensor_shape(name)) == 2]
        if len(image_inputs) != 1 or len(latent_inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError(f"unexpected InSwapper TensorRT bindings: {self.inputs} -> {self.outputs}")
        self.image_input, self.latent_input, self.output = (
            image_inputs[0],
            latent_inputs[0],
            self.outputs[0],
        )
        self.image_dtype = self.engine.get_tensor_dtype(self.image_input)
        self.latent_dtype = self.engine.get_tensor_dtype(self.latent_input)
        self.output_dtype = self.engine.get_tensor_dtype(self.output)
        self._validate_static_bindings()
        self.debug_dumper: SwapDebugDumper | None = None

    def _validate_static_bindings(self) -> None:
        """Fail early for a stale/wrong engine instead of producing plausible garbage."""

        image_shape = tuple(self.engine.get_tensor_shape(self.image_input))
        latent_shape = tuple(self.engine.get_tensor_shape(self.latent_input))
        output_shape = tuple(self.engine.get_tensor_shape(self.output))
        expected_image = (1, 3, self.metadata.input_size[1], self.metadata.input_size[0])
        if all(dimension >= 0 for dimension in image_shape) and image_shape != expected_image:
            raise RuntimeError(f"unexpected TensorRT image shape: {image_shape}, expected {expected_image}")
        if all(dimension >= 0 for dimension in latent_shape) and latent_shape != (1, 512):
            raise RuntimeError(f"unexpected TensorRT latent shape: {latent_shape}, expected (1, 512)")
        if all(dimension >= 0 for dimension in output_shape) and output_shape != expected_image:
            raise RuntimeError(f"unexpected TensorRT output shape: {output_shape}, expected {expected_image}")

    def provider_summary(self) -> list[str]:
        return [
            "TensorRTDirect",
            f"input={self.image_dtype}",
            f"latent={self.latent_dtype}",
            f"output={self.output_dtype}",
            f"image_name={self.image_input}",
            f"latent_name={self.latent_input}",
            f"output_name={self.output}",
        ]

    @staticmethod
    def _torch_dtype(trt_dtype: Any, torch: Any) -> Any:
        import tensorrt as trt

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }
        try:
            return mapping[trt_dtype]
        except KeyError as error:
            raise RuntimeError(f"unsupported TensorRT InSwapper tensor dtype: {trt_dtype}") from error

    def _forward(self, image: np.ndarray, latent: np.ndarray) -> np.ndarray:
        import torch

        image = np.ascontiguousarray(image, dtype=np.float32)
        latent = np.ascontiguousarray(latent, dtype=np.float32)
        if image.shape != (1, 3, self.metadata.input_size[1], self.metadata.input_size[0]):
            raise ValueError(f"unexpected InSwapper image tensor shape: {image.shape}")
        if latent.shape != (1, 512):
            raise ValueError(f"unexpected InSwapper latent tensor shape: {latent.shape}")
        if not np.isfinite(image).all() or not np.isfinite(latent).all():
            raise ValueError("InSwapper input contains NaN or infinity")

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
            device=f"cuda:{self.device}", dtype=self._torch_dtype(self.image_dtype, torch)
        )
        latent_tensor = torch.from_numpy(np.ascontiguousarray(latent)).to(
            device=f"cuda:{self.device}", dtype=self._torch_dtype(self.latent_dtype, torch)
        )
        self.context.set_input_shape(self.image_input, tuple(image_tensor.shape))
        self.context.set_input_shape(self.latent_input, tuple(latent_tensor.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output))
        expected_output = (1, 3, self.metadata.input_size[1], self.metadata.input_size[0])
        if output_shape != expected_output:
            raise RuntimeError(f"unresolved or unexpected TensorRT output shape: {output_shape}")
        output_tensor = torch.empty(
            output_shape,
            device=image_tensor.device,
            dtype=self._torch_dtype(self.output_dtype, torch),
        )
        self.context.set_tensor_address(self.image_input, image_tensor.data_ptr())
        self.context.set_tensor_address(self.latent_input, latent_tensor.data_ptr())
        self.context.set_tensor_address(self.output, output_tensor.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT InSwapper execution failed")
        output = output_tensor.float().cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("TensorRT InSwapper output contains NaN or infinity")
        return output

    def get(self, img: np.ndarray, target_face: Any, source_face: Any, *, paste_back: bool) -> np.ndarray:
        from insightface.utils import face_align

        aimg, matrix = face_align.norm_crop2(img, target_face.kps, self.metadata.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.metadata.input_std,
            self.metadata.input_size,
            (self.metadata.input_mean,) * 3,
            swapRB=True,
        )
        latent = _mapped_latent(source_face, self.metadata.emap)
        prediction = self._forward(blob, latent)
        bgr_fake = _prediction_to_bgr(prediction)
        if not paste_back:
            return bgr_fake
        debugger = self.debug_dumper if self.debug_dumper and self.debug_dumper.consume() else None
        artifacts: dict[str, np.ndarray] | None = {} if debugger is not None else None
        result = _paste_inswapper(img, aimg, bgr_fake, matrix, artifacts=artifacts)
        if debugger is not None:
            # The metadata model has an explicit CPU ORT session solely for this
            # opt-in comparison.  It never participates in the live TRT result.
            try:
                ort_prediction = _ort_raw_prediction(self.metadata, blob, latent)
                debugger.dump(
                    original=img,
                    landmarks=np.asarray(target_face.kps),
                    aligned=aimg,
                    blob=blob,
                    latent=latent,
                    ort_prediction=ort_prediction,
                    trt_prediction=prediction,
                    final=result,
                    artifacts=artifacts or {},
                )
            except Exception as error:
                # Diagnostics must not turn an otherwise-valid swap into a blur fallback.
                print(f"InSwapper debug dump failed: {error}")
        return result


def _mapped_latent(source_face: Any, emap: np.ndarray) -> np.ndarray:
    """Apply the InSwapper mapping matrix with explicit shape/dtype checks."""

    embedding = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)
    if embedding.shape != (1, 512):
        raise ValueError(f"unexpected source embedding shape: {embedding.shape}")
    latent = np.asarray(np.dot(embedding, emap), dtype=np.float32)
    norm = float(np.linalg.norm(latent))
    if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise ValueError(f"invalid mapped source latent norm: {norm}")
    return np.ascontiguousarray(latent / norm, dtype=np.float32)


def _prediction_to_bgr(prediction: np.ndarray) -> np.ndarray:
    """Decode this repository's official InSwapper output, whose range is [0, 1]."""

    output = np.asarray(prediction, dtype=np.float32)
    if output.shape != (1, 3, 128, 128):
        raise ValueError(f"unexpected InSwapper output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("InSwapper output contains NaN or infinity")
    return np.clip(255.0 * output.transpose((0, 2, 3, 1))[0], 0, 255).astype(np.uint8)[:, :, ::-1]


def _ort_raw_prediction(metadata: Any, blob: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Match InsightFace INSwapper.get() without applying forward() normalization again."""

    return metadata.session.run(
        metadata.output_names,
        {
            metadata.input_names[0]: blob,
            metadata.input_names[1]: latent,
        },
    )[0]


def _paste_inswapper(
    target_img: np.ndarray,
    aligned: np.ndarray,
    fake: np.ndarray,
    matrix: np.ndarray,
    *,
    artifacts: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Exact paste-back behavior from InsightFace INSwapper.get()."""

    inverse = cv2.invertAffineTransform(matrix)
    fake = cv2.warpAffine(fake, inverse, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    white = cv2.warpAffine(
        np.full(aligned.shape[:2], 255, dtype=np.float32),
        inverse,
        (target_img.shape[1], target_img.shape[0]),
        borderValue=0.0,
    )
    white[white > 20] = 255
    mask_h, mask_w = np.where(white == 255)
    if not len(mask_h) or not len(mask_w):
        raise RuntimeError("empty InSwapper paste mask")
    mask_size = int(np.sqrt((np.max(mask_h) - np.min(mask_h)) * (np.max(mask_w) - np.min(mask_w))))
    mask = cv2.erode(white, np.ones((max(mask_size // 10, 10),) * 2, np.uint8), iterations=1)
    blur = tuple(2 * value + 1 for value in (max(mask_size // 20, 5),) * 2)
    mask = cv2.GaussianBlur(mask, blur, 0).reshape((*target_img.shape[:2], 1)) / 255
    if artifacts is not None:
        artifacts["inverse_warp_swap"] = fake
        artifacts["swap_mask"] = np.rint(mask[:, :, 0] * 255).astype(np.uint8)
        artifacts["paste_before_blend"] = target_img
    return (mask * fake + (1 - mask) * target_img.astype(np.float32)).astype(np.uint8)


class SwapDebugDumper:
    """Write a stable, inspectable latest-frame bundle without affecting inference."""

    def __init__(self, directory: Path, *, max_dumps: int):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._remaining = max_dumps
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Reserve one opt-in diagnostic snapshot without taxing every frame."""

        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    @staticmethod
    def _write(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"could not write swap debug image: {path}")

    @staticmethod
    def _stats(values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=np.float32)
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    def dump(
        self,
        *,
        original: np.ndarray,
        landmarks: np.ndarray,
        aligned: np.ndarray,
        blob: np.ndarray,
        latent: np.ndarray,
        ort_prediction: np.ndarray,
        trt_prediction: np.ndarray,
        final: np.ndarray,
        artifacts: dict[str, np.ndarray],
    ) -> None:
        landmark_image = original.copy()
        for x, y in landmarks:
            cv2.circle(landmark_image, (round(float(x)), round(float(y))), 3, (0, 0, 255), -1)
        reconstructed = np.clip(blob[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        self._write(self.directory / "00_original.png", original)
        self._write(self.directory / "02_landmarks.png", landmark_image)
        self._write(self.directory / "03_aligned_target.png", aligned)
        self._write(self.directory / "04_model_input_reconstructed.png", reconstructed)
        self._write(self.directory / "05_onnx_raw_swap.png", _prediction_to_bgr(ort_prediction))
        self._write(self.directory / "06_trt_raw_swap.png", _prediction_to_bgr(trt_prediction))
        prefixes = {"swap_mask": "07", "inverse_warp_swap": "08", "paste_before_blend": "09"}
        for name, prefix in prefixes.items():
            image = artifacts.get(name)
            if image is not None:
                self._write(self.directory / f"{prefix}_{name}.png", image)
        self._write(self.directory / "10_after_blend.png", final)
        self._write(self.directory / "11_final_frame.png", final)
        diff = np.asarray(trt_prediction, dtype=np.float32) - np.asarray(ort_prediction, dtype=np.float32)
        report = {
            "input": self._stats(blob),
            "mapped_latent": {**self._stats(latent), "norm": float(np.linalg.norm(latent))},
            "onnx": self._stats(ort_prediction),
            "tensorrt": self._stats(trt_prediction),
            "ort_vs_trt": {
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "max_error": float(np.max(np.abs(diff))),
            },
            "mask": self._stats(artifacts["swap_mask"] / 255.0) if "swap_mask" in artifacts else None,
        }
        (self.directory / "debug.json").write_text(json.dumps(report, indent=2) + "\n")


class InSwapper:
    """Keep the established generator/face-analysis behavior, isolated from server code."""

    def __init__(
        self,
        source_path: Path,
        model_path: Path,
        analysis_providers: list[Any],
        *,
        backend: str,
        engine_path: Path,
        device: str,
        target_aligner: str,
        target_yunet: Path,
        debug_dir: Path | None,
        debug_frames: int,
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"InSwapper model is missing: {model_path}")
        from insightface import model_zoo
        from insightface.app import FaceAnalysis

        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"could not read swap source: {source_path}")
        self.analysis = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=analysis_providers,
        )
        self.analysis.prepare(ctx_id=0, det_size=(640, 640))
        # TensorRT 11 cannot be loaded by ONNX Runtime's TensorRT 10 EP.  Keep
        # ONNX Runtime only for the explicit CUDA comparison path; the default
        # uses a direct TensorRT engine for the generator.
        if backend == "tensorrt":
            _require_current_swapper_engine(engine_path, model_path)
            self.model = model_zoo.get_model(str(model_path), providers=["CPUExecutionProvider"])
            self.generator: Any = TensorRtInSwapperGenerator(self.model, engine_path, device)
            if debug_dir is not None:
                self.generator.debug_dumper = SwapDebugDumper(debug_dir, max_dumps=debug_frames)
        else:
            self.model = model_zoo.get_model(
                str(model_path), providers=_swap_providers(device, 2.0)
            )
            self.generator = self.model
        self.target_aligner = target_aligner
        self.yunet = None
        if target_aligner == "yunet_roi":
            if not target_yunet.is_file():
                raise FileNotFoundError(f"YuNet target aligner is missing: {target_yunet}")
            self.yunet = cv2.FaceDetectorYN.create(
                str(target_yunet), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=32
            )
        self.last_alignment_ms = 0.0
        self.last_generator_ms = 0.0
        faces = self.analysis.get(source)
        if not faces:
            raise ValueError("no source face found")
        self.source_face = max(
            faces, key=lambda face: float(np.prod(face.bbox[2:] - face.bbox[:2]))
        )
        # Initialize CUDA kernels, TensorRT tactics, and allocator state before
        # the first camera frame.  This removes one-time latency from the live
        # swap generator metric.
        if backend == "tensorrt":
            self._warmup(source)

    def _warmup(self, source: np.ndarray) -> None:
        started = time.perf_counter()
        debug_dumper = self.generator.debug_dumper
        self.generator.debug_dumper = None
        try:
            self.generator.get(source, self.source_face, self.source_face, paste_back=False)
        finally:
            self.generator.debug_dumper = debug_dumper
        print(f"InSwapper TensorRT warm-up completed in {(time.perf_counter() - started) * 1_000:.1f}ms")

    def provider_summary(self) -> dict[str, list[str]]:
        analysis: set[str] = set()
        for model in self.analysis.models.values():
            session = getattr(model, "session", None)
            if session is not None:
                analysis.update(session.get_providers())
        generator = self.generator.provider_summary()
        return {
            "analysis": sorted(analysis),
            "generator": list(generator),
            "target_aligner": [self.target_aligner],
        }

    def apply(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        output, succeeded = self.apply_many(frame, [bbox])
        if not succeeded:
            raise RuntimeError("swapper could not align detected face")
        return output

    def apply_many(
        self, frame: np.ndarray, boxes: list[list[float]]
    ) -> tuple[np.ndarray, set[int]]:
        """Run FaceAnalysis once per frame, then map only YOLO class-0 boxes to it."""

        alignment_started = time.perf_counter()
        targets = self._target_faces(frame, boxes)
        self.last_alignment_ms = (time.perf_counter() - alignment_started) * 1_000
        output = frame
        succeeded: set[int] = set()
        generator_started = time.perf_counter()
        for index, target in targets.items():
            if target is None:
                continue
            try:
                output = self.generator.get(output, target, self.source_face, paste_back=True)
            except Exception:
                continue
            succeeded.add(index)
        self.last_generator_ms = (time.perf_counter() - generator_started) * 1_000
        return output, succeeded

    def _target_faces(self, frame: np.ndarray, boxes: list[list[float]]) -> dict[int, Any]:
        targets: dict[int, Any] = {}
        if self.yunet is not None:
            for index, box in enumerate(boxes):
                target = self._target_from_yunet(frame, box)
                if target is not None:
                    targets[index] = target
        missing = [index for index in range(len(boxes)) if index not in targets]
        if not missing:
            return targets
        faces = self.analysis.get(frame)
        available = set(range(len(faces)))
        for index in missing:
            if not available:
                break
            target_index = max(
                available, key=lambda candidate: _iou(faces[candidate].bbox, boxes[index])
            )
            target = faces[target_index]
            if _iou(target.bbox, boxes[index]) < 0.2:
                continue
            available.remove(target_index)
            targets[index] = target
        return targets

    def _target_from_yunet(self, frame: np.ndarray, box: list[float]) -> Any | None:
        if self.yunet is None:
            return None
        x1, y1, x2, y2 = (float(value) for value in box)
        padding_x = (x2 - x1) * 0.35
        padding_y = (y2 - y1) * 0.35
        left = max(0, int(np.floor(x1 - padding_x)))
        top = max(0, int(np.floor(y1 - padding_y)))
        right = min(frame.shape[1], int(np.ceil(x2 + padding_x)))
        bottom = min(frame.shape[0], int(np.ceil(y2 + padding_y)))
        if right - left < 32 or bottom - top < 32:
            return None
        roi = frame[top:bottom, left:right]
        try:
            self.yunet.setInputSize((roi.shape[1], roi.shape[0]))
            _, detections = self.yunet.detect(roi)
        except cv2.error:
            return None
        if detections is None or not len(detections):
            return None
        candidate = max(
            detections,
            key=lambda row: _iou(
                [row[0] + left, row[1] + top, row[0] + row[2] + left, row[1] + row[3] + top], box
            ),
        )
        candidate_box = [
            float(candidate[0] + left),
            float(candidate[1] + top),
            float(candidate[0] + candidate[2] + left),
            float(candidate[1] + candidate[3] + top),
        ]
        if _iou(candidate_box, box) < 0.2:
            return None
        from insightface.app.common import Face

        landmarks = np.asarray(candidate[5:15], dtype=np.float32).reshape((5, 2))
        landmarks[:, 0] += left
        landmarks[:, 1] += top
        return Face(bbox=np.asarray(candidate_box, dtype=np.float32), kps=landmarks)


class SwapLab:
    def __init__(self, settings: Settings):
        from ultralytics import YOLO

        self.settings = settings
        self.model, self.detector_backend = self._load_detector(YOLO, settings.detector)
        self.names = {int(key): str(value) for key, value in self.model.names.items()}
        if self.names != {0: "face", 1: "number_plate"}:
            raise RuntimeError(f"expected class 0=face and 1=number_plate, got {self.names}")
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is required for this GPU test client")
        self.swapper = InSwapper(
            settings.source,
            settings.swapper,
            _swap_providers(settings.device, settings.swap_ort_mem_gib),
            backend=settings.swapper_backend,
            engine_path=settings.swapper_engine,
            device=settings.device,
            target_aligner=settings.target_aligner,
            target_yunet=settings.target_yunet,
            debug_dir=settings.swap_debug_dir,
            debug_frames=settings.swap_debug_frames,
        )
        self.sessions = SessionRegistry()
        self.adaface = LazyAdaFaceRuntime(
            AdaFaceConfig(device=f"cuda:{settings.device}", queue_capacity=32),
            fallback_device=settings.device,
        )
        self.queue: asyncio.Queue[FrameJob] = asyncio.Queue(maxsize=settings.max_queue)
        self.worker: asyncio.Task[None] | None = None
        self.frames = 0
        self.latencies: deque[float] = deque(maxlen=300)

    @staticmethod
    def _load_detector(yolo: Any, detector_path: Path) -> tuple[Any, str]:
        """Keep the swap test runnable when a serialized detector is from another TRT runtime."""

        try:
            model = yolo(str(detector_path), task="segment")
            # Ultralytics may defer TensorRT deserialization until this property access.
            _ = model.names
            return model, "TensorRT" if detector_path.suffix == ".engine" else "PyTorch"
        except Exception as error:
            if detector_path.suffix != ".engine" or not DEFAULT_DETECTOR_CHECKPOINT.is_file():
                raise
            print(
                f"Detector engine could not load ({error}); using {DEFAULT_DETECTOR_CHECKPOINT.name} "
                "until it is rebuilt for this TensorRT runtime."
            )
            return yolo(str(DEFAULT_DETECTOR_CHECKPOINT), task="segment"), "PyTorchFallback"

    async def start(self) -> None:
        self.worker = asyncio.create_task(self._batch_loop(), name="yolo-swap-batcher")

    async def close(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        await asyncio.to_thread(self.adaface.close)

    def create_stream(self, session_id: str) -> StreamState:
        self.sessions.get_or_create(session_id)
        return StreamState(
            session_id=session_id,
            # The prior one-frame mask hold visibly lags fast movement.  This
            # live renderer always uses the current detector polygon; tracking
            # remains enabled for stable identity association only.
            tracker=StreamTracker(device=self.settings.device, mask_hold_frames=0),
            recognition=StreamRecognition(self.adaface, RecognitionConfig(), owner=session_id),
        )

    async def submit(
        self, frame: np.ndarray, stream: StreamState
    ) -> tuple[np.ndarray, dict[str, Any]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[np.ndarray, dict[str, Any]]] = loop.create_future()
        try:
            self.queue.put_nowait(FrameJob(frame, stream, future))
        except asyncio.QueueFull as error:
            raise RuntimeError("detector queue is saturated; drop this frame") from error
        return await future

    async def enroll(self, session_id: str, image: np.ndarray) -> dict[str, Any]:
        await asyncio.to_thread(self.adaface.ensure)
        future = self.adaface.submit_enrollment(image, owner=f"enroll:{session_id}")
        if future is None:
            raise HTTPException(503, "AdaFace queue is full")
        embedding = await future
        entry, count, version = self.sessions.append(session_id, embedding)
        return {"entry_id": entry.entry_id, "entry_count": count, "whitelist_version": version}

    async def _batch_loop(self) -> None:
        while True:
            first = await self.queue.get()
            jobs = [first]
            deadline = asyncio.get_running_loop().time() + self.settings.batch_wait_ms / 1000
            while len(jobs) < self.settings.max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    jobs.append(await asyncio.wait_for(self.queue.get(), remaining))
                except TimeoutError:
                    break
            try:
                inference_started = time.perf_counter()
                results = await asyncio.to_thread(self._predict, [job.frame for job in jobs])
                detector_batch_ms = (time.perf_counter() - inference_started) * 1_000
                for job, prediction in zip(jobs, results, strict=True):
                    output, meta = await self._compose(job.frame, prediction, job.stream, len(jobs))
                    meta["detector_batch_ms"] = round(detector_batch_ms, 2)
                    job.future.set_result((output, meta))
            except Exception as error:
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(error)
            finally:
                for _ in jobs:
                    self.queue.task_done()

    def _predict(self, frames: list[np.ndarray]) -> list[Any]:
        return list(
            self.model.predict(
                source=frames,
                batch=len(frames),
                imgsz=IMAGE_SIZE,
                conf=DETECTOR_CONFIDENCE,
                iou=0.70,
                classes=[0, 1],
                max_det=MAX_DETECTIONS,
                retina_masks=True,
                device=self.settings.device,
                quantize=16,
                verbose=False,
            )
        )

    async def _compose(
        self, frame: np.ndarray, prediction: Any, stream: StreamState, batch_size: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        boxes = prediction.boxes.cpu().numpy()
        tracks = stream.tracker.update(boxes, frame)
        objects = _objects(prediction, tracks, self.names, frame.shape[1], frame.shape[0])
        objects, temporal = stream.tracker.stabilize(objects, frame.shape[1], frame.shape[0])
        stream.sequence += 1
        recognition = stream.recognition.process(
            frame, objects, self.sessions.snapshot(stream.session_id), stream.sequence
        )
        output = frame.copy()
        swapped = 0
        fallback = 0
        small_face_fallbacks = 0
        fallback_objects: list[dict[str, Any]] = []
        swap_candidates: list[dict[str, Any]] = []
        for item in objects:
            if is_number_plate_object(item):
                fallback_objects.append(item)
                fallback += 1
                continue
            if not is_face_object(item) or item.get("whitelisted") is True:
                continue
            if (
                item.get("held")
                or float(item.get("mask_area_px", 0.0)) < self.settings.swap_min_mask_area_px
            ):
                fallback_objects.append(item)
                fallback += 1
                small_face_fallbacks += 1
                continue
            swap_candidates.append(item)
        swap_started = time.perf_counter()
        if swap_candidates:
            try:
                output, succeeded = self.swapper.apply_many(
                    output, [item["bbox"] for item in swap_candidates]
                )
            except Exception:
                succeeded = set()
            for index, item in enumerate(swap_candidates):
                if index in succeeded:
                    swapped += 1
                else:
                    fallback_objects.append(item)
                    fallback += 1
        swap_ms = (time.perf_counter() - swap_started) * 1_000
        if fallback_objects:
            output = _blur_objects(output, fallback_objects)
        elapsed = (time.perf_counter() - started) * 1_000
        self.frames += 1
        self.latencies.append(elapsed)
        return output, {
            "detections": len(objects),
            "swap_faces": swapped,
            "fallback_blurs": fallback,
            "small_face_fallbacks": small_face_fallbacks,
            "swap_ms": round(swap_ms, 2),
            "swap_alignment_ms": round(self.swapper.last_alignment_ms, 2),
            "swap_generator_ms": round(self.swapper.last_generator_ms, 2),
            "yolo_batch": batch_size,
            "adaface": recognition,
            "tracking": temporal,
            "total_ms": round(elapsed, 2),
        }


def _objects(prediction: Any, tracks: np.ndarray, names: dict[int, str], width: int, height: int):
    polygons = prediction.masks.xy if prediction.masks is not None else []
    objects: list[dict[str, Any]] = []
    for row in tracks:
        index = int(row[-1])
        if not 0 <= index < len(prediction.boxes):
            continue
        try:
            polygon = (
                np.asarray(polygons[index], dtype=np.float32).reshape((-1, 2))
                if index < len(polygons)
                else np.empty((0, 2), dtype=np.float32)
            )
        except (TypeError, ValueError):
            polygon = np.empty((0, 2), dtype=np.float32)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            x1, y1, x2, y2 = row[:4]
            polygon = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
        stride = max(1, int(np.ceil(len(polygon) / MAX_POLYGON_POINTS)))
        polygon = np.ascontiguousarray(polygon[::stride][:MAX_POLYGON_POINTS], dtype=np.float32)
        class_id = int(row[6])
        objects.append(
            {
                "track_id": int(row[4]),
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": float(row[5]),
                "bbox": [float(value) for value in row[:4]],
                "mask_polygon": polygon.tolist(),
                "mask_area_px": _polygon_area(polygon),
            }
        )
    return objects


def _polygon_area(polygon: np.ndarray) -> float:
    """Compute a finite contour area without requiring an OpenCV contour layout."""

    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if len(points) < 3 or not np.isfinite(points).all():
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def _blur_objects(image: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    """Apply the production fallback algorithm once, after all swap attempts."""

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for item in items:
        polygon = np.rint(np.asarray(item["mask_polygon"], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    if not np.any(mask):
        return image
    if MASK_FEATHER_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_FEATHER_RADIUS * 2 + 1, MASK_FEATHER_RADIUS * 2 + 1)
        )
        expanded = cv2.dilate(mask, kernel)
        distance = cv2.distanceTransform(expanded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha_mask = np.minimum(distance / MASK_FEATHER_RADIUS, 1.0) * 255.0
        alpha_mask = np.rint(alpha_mask).astype(np.uint8)
        alpha_mask[mask != 0] = 255
    else:
        alpha_mask = mask
    rows, columns = np.nonzero(alpha_mask)
    padding = int(np.ceil(DEFAULT_BLUR_RADIUS * 3))
    top = max(0, int(rows.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(image.shape[1], int(columns.max()) + padding + 1)
    region = image[top:bottom, left:right]
    reduced_size = (
        max(1, int(np.ceil(region.shape[1] / DEFAULT_PIXEL_SIZE))),
        max(1, int(np.ceil(region.shape[0] / DEFAULT_PIXEL_SIZE))),
    )
    reduced = cv2.resize(region, reduced_size, interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(
        cv2.GaussianBlur(reduced, (0, 0), DEFAULT_BLUR_RADIUS / DEFAULT_PIXEL_SIZE),
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    output = image.copy()
    alpha = alpha_mask[top:bottom, left:right, None].astype(np.uint32)
    output[top:bottom, left:right] = (
        region.astype(np.uint32) * (255 - alpha) + blurred.astype(np.uint32) * alpha + 127
    ) // 255
    return output


def _iou(left: Any, right: list[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in left[:4])
    a1, b1, a2, b2 = right
    intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union > 0 else 0.0


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="InnoLive TensorRT Face Swap Lab")
    lab = SwapLab(settings)
    file_task: asyncio.Task[None] | None = None
    settings.hls_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/hls", StaticFiles(directory=str(settings.hls_dir)), name="hls")

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal file_task
        await lab.start()
        if settings.input_video is not None:
            file_task = asyncio.create_task(
                _run_nvcodec_file(lab, settings.input_video, settings.hls_dir),
                name="nvdec-nvenc-file-swap",
            )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if file_task is not None:
            file_task.cancel()
            await asyncio.gather(file_task, return_exceptions=True)
        await lab.close()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _HTML.replace("__JPEG_QUALITY__", str(settings.stream_jpeg_quality / 100))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ready": True,
            "frames": lab.frames,
            "queue": lab.queue.qsize(),
            "p50_ms": float(np.percentile(lab.latencies, 50)) if lab.latencies else None,
            "swapper_providers": lab.swapper.provider_summary(),
            "detector_backend": lab.detector_backend,
            "last_swap_alignment_ms": round(lab.swapper.last_alignment_ms, 2),
            "last_swap_generator_ms": round(lab.swapper.last_generator_ms, 2),
        }

    @app.post("/api/enroll/{session_id}")
    async def enroll(session_id: str, payload: dict[str, str]) -> dict[str, Any]:
        raw = base64.b64decode(payload.get("jpeg", ""), validate=True)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "invalid jpeg")
        return await lab.enroll(session_id, image)

    @app.websocket("/ws/{session_id}")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        state = lab.create_stream(session_id)
        try:
            while True:
                payload = await websocket.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                try:
                    output, metadata = await lab.submit(frame, state)
                except RuntimeError as error:
                    await websocket.send_json({"error": str(error), "dropped": True})
                    continue
                ok, encoded = cv2.imencode(
                    ".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, settings.stream_jpeg_quality]
                )
                if ok:
                    await websocket.send_json(metadata)
                    await websocket.send_bytes(encoded.tobytes())
        except WebSocketDisconnect:
            state.recognition.close()
            state.tracker.reset()

    return app


async def _run_nvcodec_file(lab: SwapLab, input_video: Path, output_dir: Path) -> None:
    """Run an optional hardware decode/encode stream through the same batcher."""

    ffmpeg = await asyncio.to_thread(require_nvcodec_ffmpeg)
    spec = await asyncio.to_thread(probe_video, input_video)
    reader = NvdecReader(input_video, spec, ffmpeg=ffmpeg)
    writer = NvencHlsWriter(output_dir, spec, ffmpeg=ffmpeg)
    state = lab.create_stream("nvcodec-file")
    try:
        while frame := await asyncio.to_thread(reader.read):
            output, _ = await lab.submit(frame, state)
            await asyncio.to_thread(writer.write, output)
    finally:
        state.recognition.close()
        state.tracker.reset()
        await asyncio.to_thread(reader.close)
        await asyncio.to_thread(writer.close)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--detector", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--swapper", type=Path, default=DEFAULT_SWAPPER)
    parser.add_argument("--swapper-engine", type=Path, default=DEFAULT_SWAPPER_ENGINE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=3.0)
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument(
        "--swap-ort-mem-gib",
        type=float,
        default=2.0,
        help="per-session ONNX Runtime CUDA arena limit for InSwapper (default: 2 GiB)",
    )
    parser.add_argument(
        "--swap-min-mask-area-px",
        type=float,
        default=16_384,
        help="face segmentation mask area below which the client uses protected blur",
    )
    parser.add_argument("--swapper-backend", choices=("tensorrt", "cuda"), default="tensorrt")
    parser.add_argument("--swapper-trt-cache", type=Path, default=DEFAULT_SWAPPER_TRT_CACHE)
    parser.add_argument(
        "--swapper-trt-workspace-gib",
        type=float,
        default=1.0,
        help="TensorRT EP workspace cap for the 128 swap generator",
    )
    parser.add_argument(
        "--target-aligner",
        choices=("yunet_roi", "insightface"),
        default="yunet_roi",
        help="target five-point landmark path; insightface is the exact legacy comparison path",
    )
    parser.add_argument("--target-yunet", type=Path, default=DEFAULT_FACE_DETECTOR)
    parser.add_argument(
        "--input-video",
        type=Path,
        help="optional compressed input decoded by NVDEC and written as /hls/live.m3u8 via NVENC",
    )
    parser.add_argument("--hls-dir", type=Path, default=DEFAULT_HLS_DIR)
    parser.add_argument(
        "--swap-debug-dir",
        type=Path,
        help="save a limited ORT/TRT raw and paste-back diagnostic bundle here (TensorRT only)",
    )
    parser.add_argument(
        "--swap-debug-frames",
        type=int,
        default=1,
        help="number of diagnostic swaps to save; debug is disabled when --swap-debug-dir is absent",
    )
    parser.add_argument(
        "--stream-jpeg-quality",
        type=int,
        default=100,
        help="browser input and WebSocket output JPEG quality (1-100; default: 100 for diagnosis)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.max_batch < 1
        or args.max_queue < args.max_batch
        or args.batch_wait_ms < 0
        or not 0 < args.swap_ort_mem_gib <= 8
        or args.swap_min_mask_area_px < 1
        or not 0 < args.swapper_trt_workspace_gib <= 4
        or args.swap_debug_frames < 1
        or not 1 <= args.stream_jpeg_quality <= 100
    ):
        raise SystemExit(
            "max-batch >= 1, max-queue >= max-batch, batch-wait-ms >= 0, "
            "swap-ort-mem-gib in (0, 8], swap-min-mask-area-px >= 1, "
            "swapper-trt-workspace-gib in (0, 4], swap-debug-frames >= 1, and "
            "stream-jpeg-quality in [1, 100] are required"
        )
    input_video = args.input_video.expanduser().resolve() if args.input_video else None
    if input_video is not None and not input_video.is_file():
        raise SystemExit(f"input video not found: {input_video}")
    settings = Settings(
        args.detector.expanduser().resolve(),
        args.swapper.expanduser().resolve(),
        args.swapper_engine.expanduser().resolve(),
        args.source.expanduser().resolve(),
        args.device,
        args.max_batch,
        args.batch_wait_ms,
        args.max_queue,
        args.swap_ort_mem_gib,
        args.swap_min_mask_area_px,
        args.swapper_backend,
        args.swapper_trt_cache.expanduser().resolve(),
        args.swapper_trt_workspace_gib,
        args.target_aligner,
        args.target_yunet.expanduser().resolve(),
        input_video,
        args.hls_dir.expanduser().resolve(),
        args.swap_debug_dir.expanduser().resolve() if args.swap_debug_dir else None,
        args.swap_debug_frames,
        args.stream_jpeg_quality,
    )
    import uvicorn

    uvicorn.run(create_app(settings), host=args.host, port=args.port)


_HTML = """<!doctype html><meta charset=utf-8><title>TensorRT Swap Lab</title><style>body{font:16px system-ui;background:#111;color:#eee;margin:2rem}video,img{width:min(48%,720px);background:#222}pre{background:#222;padding:1rem}</style><h1>TensorRT Face Swap Lab</h1><p>Browser webcam → batched YOLO → class-0 swap / protected fallback</p><video id=v autoplay muted playsinline></video><img id=o><pre id=m>starting…</pre><script>const id=crypto.randomUUID(),ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/${id}`),v=document.querySelector('#v'),o=document.querySelector('#o'),m=document.querySelector('#m'),c=document.createElement('canvas');let busy=false;navigator.mediaDevices.getUserMedia({video:{width:1920,height:1080},audio:false}).then(s=>v.srcObject=s);ws.onmessage=e=>{if(typeof e.data==='string'){m.textContent=e.data;return}o.src=URL.createObjectURL(e.data);busy=false};setInterval(()=>{if(busy||!v.videoWidth||ws.readyState!==1)return;busy=true;c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(b=>{if(b)ws.send(b);else busy=false},'image/jpeg',__JPEG_QUALITY__)},33)</script>"""


if __name__ == "__main__":
    main()
