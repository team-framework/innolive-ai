#!/usr/bin/env python3
"""TensorRT YuNet runner with OpenCV-identical postprocessing (issue #21).

Runs the YuNet graph on GPU and decodes raw stride outputs exactly like
OpenCV's FaceDetectorYN (score = sqrt(clamp(cls) * clamp(obj)), no sigmoid;
center+wh bbox decode; landmarks relative to cell; cv2 NMS).  Validated
against cv2.FaceDetectorYN on production ROIs: box p50 0.38px, kps p50
0.16px, ~1ms vs 3-10ms CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STRIDES = (8, 16, 32)
MAX_SIDE = 1024


def extend_window(
    frame: np.ndarray, left: int, top: int, right: int, bottom: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Grow an ROI to 32-multiples with real frame pixels (clamped).

    The YuNet graph needs height/width divisible by 32.  Extending with real
    pixels keeps coordinates exact; only a clamped remainder (frame edge) is
    replicate-padded.  Returns (window, (origin_x, origin_y)).
    """

    height, width = frame.shape[:2]
    padded_w = int(math.ceil((right - left) / 32) * 32)
    padded_h = int(math.ceil((bottom - top) / 32) * 32)
    origin_x, origin_y = left, top
    far_x = min(width, origin_x + padded_w)
    origin_x = max(0, far_x - padded_w)
    far_y = min(height, origin_y + padded_h)
    origin_y = max(0, far_y - padded_h)
    window = frame[origin_y:far_y, origin_x:far_x]
    missing_h = int(math.ceil(window.shape[0] / 32) * 32) - window.shape[0]
    missing_w = int(math.ceil(window.shape[1] / 32) * 32) - window.shape[1]
    if missing_h or missing_w:
        window = cv2.copyMakeBorder(window, 0, missing_h, 0, missing_w, cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(window), (origin_x, origin_y)


def decode_detections(
    raw: dict[str, np.ndarray],
    width: int,
    height: int,
    *,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 32,
) -> np.ndarray:
    """Decode YuNet stride outputs to Nx15 rows like cv2.FaceDetectorYN."""

    kept: list[list[float]] = []
    for stride in STRIDES:
        cls = np.asarray(raw[f"cls_{stride}"]).reshape(-1)
        obj = np.asarray(raw[f"obj_{stride}"]).reshape(-1)
        bbox = np.asarray(raw[f"bbox_{stride}"]).reshape(-1, 4)
        kps = np.asarray(raw[f"kps_{stride}"]).reshape(-1, 10)
        cols, rows = width // stride, height // stride
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                score = math.sqrt(
                    min(max(float(cls[index]), 0.0), 1.0) * min(max(float(obj[index]), 0.0), 1.0)
                )
                if score < score_threshold:
                    continue
                center_x = (col + float(bbox[index, 0])) * stride
                center_y = (row + float(bbox[index, 1])) * stride
                w = math.exp(float(bbox[index, 2])) * stride
                h = math.exp(float(bbox[index, 3])) * stride
                face = [center_x - w / 2, center_y - h / 2, w, h]
                for point in range(5):
                    face.append((float(kps[index, 2 * point]) + col) * stride)
                    face.append((float(kps[index, 2 * point + 1]) + row) * stride)
                face.append(score)
                kept.append(face)
    if not kept:
        return np.empty((0, 15), dtype=np.float32)
    faces = np.asarray(kept, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        [list(map(int, row)) for row in faces[:, :4]],
        faces[:, 14].tolist(),
        score_threshold,
        nms_threshold,
        top_k=top_k,
    )
    return np.ascontiguousarray(faces[np.asarray(indices).reshape(-1)])


class YuNetTRT:
    """GPU YuNet with one shared context; thread-safe via an internal lock."""

    def __init__(
        self,
        engine_path: Path,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 32,
    ):
        import tensorrt as trt

        manifest_path = engine_path.with_suffix(engine_path.suffix + ".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"YuNet engine manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid YuNet engine manifest: {manifest_path}") from error
        if not engine_path.is_file():
            raise FileNotFoundError(f"YuNet TensorRT engine is missing: {engine_path}")
        engine_hash = hashlib.sha256(engine_path.read_bytes()).hexdigest()
        if manifest.get("engine_sha256") != engine_hash:
            raise RuntimeError("YuNet engine does not match its manifest; rebuild it")
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize YuNet engine: {engine_path}")
        tensor_names = [
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        ]
        inputs = [name for name in tensor_names if len(self.engine.get_tensor_shape(name)) == 4]
        if len(inputs) != 1:
            raise RuntimeError(f"unexpected YuNet bindings: {tensor_names}")
        self.input_name = inputs[0]
        self.output_names = [name for name in tensor_names if name != self.input_name]
        self.context = self.engine.create_execution_context()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.max_side = int(manifest.get("max_size", MAX_SIDE))
        self._lock = threading.Lock()
        self._buffers: dict[str, Any] = {}

    def infer(self, window: np.ndarray) -> dict[str, np.ndarray]:
        """Run one window (HxW, any size within profile); returns raw stride outputs."""

        import torch

        height, width = window.shape[:2]
        blob = cv2.dnn.blobFromImage(window, 1.0, (width, height), (0, 0, 0), swapRB=False)
        image = torch.from_numpy(np.ascontiguousarray(blob, dtype=np.float32)).cuda()
        with self._lock:
            self.context.set_input_shape(self.input_name, tuple(image.shape))
            self.context.set_tensor_address(self.input_name, image.data_ptr())
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                cached = self._buffers.get(name)
                if cached is None or tuple(cached.shape) != shape:
                    cached = torch.empty(shape, device=image.device)
                    self._buffers[name] = cached
                self.context.set_tensor_address(name, cached.data_ptr())
            stream = torch.cuda.current_stream()
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("YuNet TensorRT execution failed")
            stream.synchronize()
            return {name: self._buffers[name].float().cpu().numpy() for name in self.output_names}

    def detect(self, window: np.ndarray) -> np.ndarray:
        """Decode one window to Nx15 rows in window coordinates."""

        raw = self.infer(window)
        return decode_detections(
            raw,
            window.shape[1],
            window.shape[0],
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k,
        )
