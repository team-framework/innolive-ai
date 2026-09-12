#!/usr/bin/env python3
"""WebSocket test client for batched YOLO face swap with fail-closed blur."""

from __future__ import annotations

import argparse
import asyncio
import base64
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
DEFAULT_SOURCE = Path.home() / "Documents" / "input.png"
DEFAULT_SWAPPER = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
DEFAULT_SWAPPER_ENGINE = ROOT / "models" / "face_swap" / "inswapper_128_trt11.engine"
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

    def provider_summary(self) -> list[str]:
        return [
            "TensorRTDirect",
            f"input={self.image_dtype}",
            f"latent={self.latent_dtype}",
            f"output={self.output_dtype}",
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

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(
            device=f"cuda:{self.device}", dtype=self._torch_dtype(self.image_dtype, torch)
        )
        latent_tensor = torch.from_numpy(np.ascontiguousarray(latent)).to(
            device=f"cuda:{self.device}", dtype=self._torch_dtype(self.latent_dtype, torch)
        )
        self.context.set_input_shape(self.image_input, tuple(image_tensor.shape))
        self.context.set_input_shape(self.latent_input, tuple(latent_tensor.shape))
        output_shape = tuple(self.context.get_tensor_shape(self.output))
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
        return output_tensor.float().cpu().numpy()

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
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.metadata.emap)
        latent /= np.linalg.norm(latent)
        prediction = self._forward(blob, latent.astype(np.float32, copy=False))
        bgr_fake = np.clip(255 * prediction.transpose((0, 2, 3, 1))[0], 0, 255).astype(np.uint8)[
            :, :, ::-1
        ]
        if not paste_back:
            return bgr_fake
        return _paste_inswapper(img, aimg, bgr_fake, matrix)


def _paste_inswapper(target_img: np.ndarray, aligned: np.ndarray, fake: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Exact paste-back behavior from InsightFace INSwapper.get()."""

    fake_diff = np.abs(fake.astype(np.float32) - aligned.astype(np.float32)).mean(axis=2)
    fake_diff[:2, :], fake_diff[-2:, :], fake_diff[:, :2], fake_diff[:, -2:] = 0, 0, 0, 0
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
    return (mask * fake + (1 - mask) * target_img.astype(np.float32)).astype(np.uint8)


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
            self.model = model_zoo.get_model(str(model_path), providers=["CPUExecutionProvider"])
            self.generator: Any = TensorRtInSwapperGenerator(self.model, engine_path, device)
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
        self.model = YOLO(str(settings.detector), task="segment")
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
            tracker=StreamTracker(device=self.settings.device),
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
        return _HTML

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ready": True,
            "frames": lab.frames,
            "queue": lab.queue.qsize(),
            "p50_ms": float(np.percentile(lab.latencies, 50)) if lab.latencies else None,
            "swapper_providers": lab.swapper.provider_summary(),
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
                ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 90])
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
    ):
        raise SystemExit(
            "max-batch >= 1, max-queue >= max-batch, batch-wait-ms >= 0, "
            "swap-ort-mem-gib in (0, 8], swap-min-mask-area-px >= 1, and "
            "swapper-trt-workspace-gib in (0, 4] are required"
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
    )
    import uvicorn

    uvicorn.run(create_app(settings), host=args.host, port=args.port)


_HTML = """<!doctype html><meta charset=utf-8><title>TensorRT Swap Lab</title><style>body{font:16px system-ui;background:#111;color:#eee;margin:2rem}video,img{width:min(48%,720px);background:#222}pre{background:#222;padding:1rem}</style><h1>TensorRT Face Swap Lab</h1><p>Browser webcam → batched YOLO → class-0 swap / protected fallback</p><video id=v autoplay muted playsinline></video><img id=o><pre id=m>starting…</pre><script>const id=crypto.randomUUID(),ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/${id}`),v=document.querySelector('#v'),o=document.querySelector('#o'),m=document.querySelector('#m'),c=document.createElement('canvas');let busy=false;navigator.mediaDevices.getUserMedia({video:{width:1920,height:1080},audio:false}).then(s=>v.srcObject=s);ws.onmessage=e=>{if(typeof e.data==='string'){m.textContent=e.data;return}o.src=URL.createObjectURL(e.data);busy=false};setInterval(()=>{if(busy||!v.videoWidth||ws.readyState!==1)return;busy=true;c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(b=>{if(b)ws.send(b);else busy=false},'image/jpeg',.9)},33)</script>"""


if __name__ == "__main__":
    main()
