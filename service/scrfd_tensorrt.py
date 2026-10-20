"""Optional, serialized-use SCRFD TensorRT runner with bounded dynamic batches.

Build on the serving GPU with scripts/build_scrfd_engine.py. The original ONNX
is retained; generated engines are deployment artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class ScrfdTensorRT:
    def __init__(self, path: Path, onnx_path: Path):
        import tensorrt as trt
        import torch

        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        expected = {
            "format": 1,
            "onnx_sha256": sha256(onnx_path),
            "engine_sha256": sha256(path),
            "tensorrt": trt.__version__,
            "gpu": torch.cuda.get_device_name(),
        }
        if metadata.get("precision") not in {"fp32", "fp16"}:
            raise ValueError("unsupported SCRFD engine precision")
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("SCRFD engine metadata mismatch; rebuild on this serving GPU")
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None or self.engine.num_io_tensors != 10:
            raise ValueError("invalid SCRFD TensorRT engine")
        names = [self.engine.get_tensor_name(i) for i in range(10)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        if len(inputs) != 1 or any(self.engine.get_tensor_dtype(n) != trt.float32 for n in names):
            raise ValueError("expected one FP32 SCRFD input and nine FP32 outputs")
        self.input_name = inputs[0]
        self.outputs = metadata["outputs"]
        if set(self.outputs) != set(names) - set(inputs) or len(self.outputs) != 9:
            raise ValueError("SCRFD output names do not match the source model")
        self.max_batch = int(self.engine.get_tensor_profile_shape(self.input_name, 0)[2][0])
        if not 1 <= self.max_batch <= 16:
            raise ValueError("SCRFD engine batch limit must be 1..16")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        self.pixel_buffers = {}
        logging.getLogger("innolive.mobilefaceswap").info(
            "SCRFD TensorRT ready: precision=%s max_batch=%s GPU pixel preprocessing",
            metadata["precision"],
            self.max_batch,
        )

    def _run(self, batch, *, blob=None, pixels=None):
        torch = self.torch
        if not 1 <= batch <= self.max_batch:
            raise ValueError("invalid SCRFD batch shape")
        if not self.context.set_input_shape(self.input_name, (batch, 3, 640, 640)):
            raise ValueError("SCRFD engine rejected input shape")
        if batch not in self.buffers:
            buffers = {}
            for name in [self.input_name, *self.outputs]:
                shape = tuple(self.context.get_tensor_shape(name))
                if min(shape) <= 0:
                    raise ValueError("unresolved SCRFD tensor shape")
                buffers[name] = (
                    torch.empty(shape, dtype=torch.float32, pin_memory=True),
                    torch.empty(shape, dtype=torch.float32, device="cuda"),
                )
            self.buffers[batch] = buffers
        buffers = self.buffers[batch]
        for name, (_, device) in buffers.items():
            self.context.set_tensor_address(name, device.data_ptr())
        host, device = buffers[self.input_name]
        with torch.cuda.stream(self.stream):
            if pixels is None:
                np.copyto(host.numpy(), blob)
                device.copy_(host, non_blocking=True)
            else:
                key = tuple(pixels.shape)
                if key not in self.pixel_buffers:
                    self.pixel_buffers[key] = (
                        torch.empty(key, dtype=torch.uint8, pin_memory=True),
                        torch.empty(key, dtype=torch.uint8, device="cuda"),
                    )
                pixel_host, pixel_device = self.pixel_buffers[key]
                np.copyto(pixel_host.numpy(), pixels)
                pixel_device.copy_(pixel_host, non_blocking=True)
                size = pixels.shape[1]
                device.zero_()
                for channel in range(3):
                    device[:, channel, :size, :size].copy_(pixel_device[:, :, :, 2 - channel])
                device.sub_(127.5).mul_(1 / 128)

            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("SCRFD TensorRT execution failed")
            for name in self.outputs:
                host, device = buffers[name]
                host.copy_(device, non_blocking=True)
        self.stream.synchronize()
        return [buffers[name][0].numpy() for name in self.outputs]

    def run_batch(self, blob: np.ndarray) -> list[np.ndarray]:
        if blob.ndim != 4 or blob.shape[1:] != (3, 640, 640):
            raise ValueError("invalid SCRFD batch shape")
        return self._run(blob.shape[0], blob=blob)

    def run_images(self, pixels: np.ndarray) -> list[np.ndarray]:
        if (
            pixels.ndim != 4
            or pixels.shape[1:] not in ((320, 320, 3), (640, 640, 3))
            or pixels.dtype != np.uint8
        ):
            raise ValueError("invalid SCRFD image batch")
        return self._run(pixels.shape[0], pixels=pixels)

    def run(self, output_names, inputs):
        return self.run_batch(inputs[self.input_name])
