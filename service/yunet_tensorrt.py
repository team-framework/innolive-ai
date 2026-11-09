"""Serialized FP32 YuNet with bounded buffers and vectorized OpenCV postprocess.

Score, stride decode, integer-box NMS and zero padding follow OpenCV's
modules/objdetect/src/face_detect.cpp. Source conditioning retains CPU YuNet.
"""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np


def decode(raw, width, height):
    candidates = []
    for stride in (8, 16, 32):
        scores = np.sqrt(
            np.clip(raw[f"cls_{stride}"].reshape(-1), 0, 1)
            * np.clip(raw[f"obj_{stride}"].reshape(-1), 0, 1)
        )
        if len(scores) != (width // stride) * (height // stride):
            raise ValueError("YuNet output size mismatch")
        indices = np.flatnonzero(scores >= 0.6)
        if not len(indices):
            continue
        boxes = raw[f"bbox_{stride}"].reshape(-1, 4)[indices]
        landmarks = raw[f"kps_{stride}"].reshape(-1, 5, 2)[indices]
        grid = np.column_stack((indices % (width // stride), indices // (width // stride)))
        center = (boxes[:, :2] + grid) * stride
        size = np.exp(boxes[:, 2:]) * stride
        points = (landmarks + grid[:, None, :]) * stride
        faces = np.column_stack(
            (center - size / 2, size, points.reshape(-1, 10), scores[indices])
        ).astype(np.float32)
        if not np.isfinite(faces).all():
            raise ValueError("non-finite YuNet detections")
        candidates.append(faces)
    if not candidates:
        return np.empty((0, 15), np.float32)
    faces = np.concatenate(candidates)
    if len(faces) == 1:
        return faces
    kept = cv2.dnn.NMSBoxes(
        faces[:, :4].astype(np.int32).tolist(), faces[:, 14].tolist(), 0.6, 0.3, top_k=5000
    )
    return faces[np.asarray(kept, dtype=int).reshape(-1)]


class YuNetTensorRT:
    """Used only under MobileFSGANRenderer.lock; at most four H/W buffer sets."""

    def __init__(self, path):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise ValueError("invalid YuNet TensorRT engine")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        expected_outputs = {
            f"{kind}_{stride}" for kind in ("cls", "obj", "bbox", "kps") for stride in (8, 16, 32)
        }
        if len(inputs) != 1 or set(names) - set(inputs) != expected_outputs:
            raise ValueError("YuNet TensorRT IO mismatch")
        if any(self.engine.get_tensor_dtype(n) != trt.float32 for n in names):
            raise ValueError("YuNet requires FP32 IO")
        self.input_name, self.names = inputs[0], names
        self.outputs = sorted(expected_outputs)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("YuNet context creation failed")
        self.stream = torch.cuda.Stream()
        self.buffers = OrderedDict()

    def detect(self, image, input_size=640):
        height, width = image.shape[:2]
        if min(height, width) < 1:
            raise ValueError("empty YuNet image")
        factor = min(1.0, input_size / max(height, width))
        small = cv2.resize(image, (max(32, int(width * factor)), max(32, int(height * factor))))
        h, w = small.shape[:2]
        padded_h, padded_w = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        if max(padded_h, padded_w) > 640:
            raise ValueError("YuNet input exceeds 640px profile")
        shape = (1, 3, padded_h, padded_w)
        torch = self.torch
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            if not self.context.set_input_shape(self.input_name, shape):
                raise ValueError("YuNet engine rejected image size")
            if shape not in self.buffers:
                buffers = {}
                for name in self.names:
                    resolved = tuple(self.context.get_tensor_shape(name))
                    if min(resolved) <= 0:
                        raise ValueError("unresolved YuNet shape")
                    buffers[name] = (
                        torch.empty(resolved, dtype=torch.float32, pin_memory=True),
                        torch.empty(resolved, dtype=torch.float32, device="cuda"),
                    )
                self.buffers[shape] = buffers
                if len(self.buffers) > 4:
                    self.buffers.popitem(last=False)
            self.buffers.move_to_end(shape)
            buffers = self.buffers[shape]
            host, device = buffers[self.input_name]
            values = host.numpy()
            values.fill(0)
            values[0, :, :h, :w] = small.transpose(2, 0, 1)
            device.copy_(host, non_blocking=True)
            for name, (_, value) in buffers.items():
                if not self.context.set_tensor_address(name, value.data_ptr()):
                    raise RuntimeError("YuNet tensor binding failed")
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("YuNet execution failed")
            for name in self.outputs:
                host, device = buffers[name]
                host.copy_(device, non_blocking=True)
        self.stream.synchronize()
        faces = decode({n: buffers[n][0].numpy() for n in self.outputs}, padded_w, padded_h)
        faces[:, :14] /= factor
        boxes = np.column_stack((faces[:, :2], faces[:, :2] + faces[:, 2:4], faces[:, -1]))
        return boxes, faces[:, 4:14].reshape(-1, 5, 2)
