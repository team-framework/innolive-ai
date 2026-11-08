"""K22741 MobileFSGAN adapter for the serialized face composition lane.

The supplied BN-calibrated G and own MobileFaceNet C remain unchanged. Source
alignment, RGB [-1, 1] tensors and ellipse blending follow the inference bundle.
Only validated face polygons receive generated pixels; failures are blurred by
FaceAnonymizationRuntime. TensorRT contexts, graphs and buffers are never shared
between concurrent calls or mutated after capture.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from service.cuda_capture import CUDA_CAPTURE_LOCK
from service.inswapper import REFERENCE
from service.mobilefaceswap import MobileFaceSwapFrame

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models/face_swap/mobilefsgan"
LOGGER = logging.getLogger("innolive.mobilefsgan")
SIZE = 256
TEMPLATE = REFERENCE * (SIZE / 112)
_YY, _XX = np.mgrid[:SIZE, :SIZE].astype(np.float32)
_RADIUS = ((_XX - 128) / 105) ** 2 + ((_YY - 140) / 116) ** 2
ALPHA = np.clip((1 - _RADIUS) / 0.22, 0, 1)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def alignment(points):
    points = np.asarray(points, np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("expected five finite MobileFSGAN landmarks")
    if np.linalg.norm(points[1] - points[0]) < 5:
        raise ValueError("MobileFSGAN eyes too close for alignment")
    a, b = points - points.mean(0), TEMPLATE - TEMPLATE.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    sign = np.eye(2)
    sign[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    scale = (s * np.diag(sign)).sum() / (a * a).sum()
    matrix = np.concatenate(
        [(scale * rotation).T, (TEMPLATE.mean(0) - scale * points.mean(0) @ rotation)[:, None]],
        axis=1,
    ).astype(np.float32)
    fitted = points @ matrix[:, :2].T + matrix[:, 2]
    error = np.sqrt(((fitted - TEMPLATE) ** 2).sum(1).mean())
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0 or error > 25:
        raise ValueError("invalid MobileFSGAN alignment")
    return matrix


def crop_face(image, matrix):
    return cv2.warpAffine(
        image, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def image_tensor(crop):
    return np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1), np.float32) / 127.5 - 1


def paste_back(image, generated, matrix, polygon):
    """Validate first, then blend only the polygon ROI into the owned output."""
    if generated.shape != (3, SIZE, SIZE) or not np.isfinite(generated).all():
        raise ValueError("invalid MobileFSGAN output")
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3:
        raise ValueError("invalid MobileFSGAN face polygon")
    if not np.isfinite(points).all() or np.abs(points).max() > 1e7:
        raise ValueError("invalid MobileFSGAN face polygon coordinates")
    points = np.rint(points).astype(np.int32)
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x), max(0, y)
    right, bottom = min(image.shape[1], x + width), min(image.shape[0], y + height)
    if right <= left or bottom <= top:
        raise ValueError("MobileFSGAN polygon outside frame")
    matrix = np.asarray(matrix, np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid MobileFSGAN affine")
    if np.linalg.det(matrix[:, :2]) <= 1e-8:
        raise ValueError("degenerate MobileFSGAN affine")
    fake = np.rint((generated.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    fake = np.ascontiguousarray(fake[:, :, ::-1])
    inverse = cv2.invertAffineTransform(matrix)
    inverse[:, 2] -= [left, top]
    size = (right - left, bottom - top)
    placed = cv2.warpAffine(fake, inverse, size, flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(ALPHA, inverse, size, flags=cv2.INTER_LINEAR)
    segmentation = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(segmentation, [points - [left, top]], 1)
    alpha *= segmentation
    if not np.any(alpha > 0.5):
        raise ValueError("MobileFSGAN output does not overlap face polygon")
    alpha = alpha[:, :, None]
    roi = image[top:bottom, left:right]
    blended = np.rint(placed * alpha + roi * (1 - alpha)).clip(0, 255).astype(np.uint8)
    roi[:] = blended


class YuNetLandmarks:
    def __init__(self, path):
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (640, 640), 0.6, 0.3, 5000)

    def detect(self, image, input_size=640):
        height, width = image.shape[:2]
        if min(height, width) < 1:
            raise ValueError("empty MobileFSGAN landmark crop")
        scale = min(1.0, input_size / max(height, width))
        small = cv2.resize(image, (max(32, int(width * scale)), max(32, int(height * scale))))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
        faces = faces.copy()
        faces[:, :14] /= scale
        boxes = np.column_stack((faces[:, :2], faces[:, :2] + faces[:, 2:4], faces[:, -1]))
        return boxes, faces[:, 4:14].reshape(-1, 5, 2)


class MobileFSGANFrame(MobileFaceSwapFrame):
    """Share current-frame full-size YuNet detections across all tracked faces."""

    def prime(self, bboxes):
        boxes, points = self.detector.detect(self.image)
        self.full_frame[640] = (boxes, points)
        for bbox in bboxes:
            matched = self.match(boxes, points, np.asarray(bbox))
            if matched is not None:
                self.roi_points[tuple(bbox)] = matched


def validate_bundle(bundle, engine_dir, *, version, gpu, capability):
    manifest = json.loads((bundle / "model_manifest.json").read_text())
    engines = json.loads((engine_dir / "engine_manifest.json").read_text())
    if (
        engines.get("tensorrt") != version
        or engines.get("gpu") != gpu
        or engines.get("compute_capability") != list(capability)
        or engines.get("precision") not in {"fp32", "mixed_fp16"}
    ):
        raise ValueError("MobileFSGAN engine environment mismatch; rebuild on this GPU")
    for relative in ("onnx/C_fp32.onnx", "onnx/G_fp32.onnx", "detector/yunet.onnx"):
        if manifest["asset_sha256"].get(relative) != sha256(bundle / relative):
            raise ValueError(f"MobileFSGAN asset hash mismatch: {relative}")
    for name in ("C", "G"):
        if (
            engines["model_onnx_sha256"].get(name)
            != manifest["asset_sha256"][f"onnx/{name}_fp32.onnx"]
        ):
            raise ValueError(f"MobileFSGAN engine source mismatch: {name}")
        filename = engines.get("engine_files", {}).get(name, f"{name}_fp32.plan")
        if filename not in {f"{name}_fp32.plan", "G_mixed_fp16.plan"}:
            raise ValueError("unsupported MobileFSGAN engine filename")
        if engines["engines"].get(filename) != sha256(engine_dir / filename):
            raise ValueError(f"MobileFSGAN engine hash mismatch: {name}")
    compiled = engines.get("compiled_onnx_sha256", {})
    if engines["precision"] == "mixed_fp16" and "G_mixed_fp16.onnx" not in compiled:
        raise ValueError("mixed MobileFSGAN engine requires converted graph provenance")
    for filename, digest in compiled.items():
        if filename not in {"C_fp32.onnx", "G_fp32.onnx", "G_mixed_fp16.onnx"}:
            raise ValueError("unsupported MobileFSGAN graph filename")
        if sha256(engine_dir / filename) != digest:
            raise ValueError("MobileFSGAN compiled graph hash mismatch")
    if engines.get("landmarks_engine"):
        if engines["landmarks_engine"] != "yunet_fp32.plan":
            raise ValueError("unsupported MobileFSGAN landmarks engine")
        if engines.get("landmarks_onnx_sha256") != manifest["asset_sha256"]["detector/yunet.onnx"]:
            raise ValueError("MobileFSGAN landmarks source mismatch")
        if sha256(engine_dir / "yunet_fp32.plan") != engines["engines"].get("yunet_fp32.plan"):
            raise ValueError("MobileFSGAN landmarks engine hash mismatch")
    return engines


class TensorRTEngine:
    """One immutable context and address set per bounded batch, including graphs."""

    def __init__(self, path, inputs, output, max_batch, stream, graphs):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise ValueError("invalid MobileFSGAN TensorRT engine")
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        self.inputs, self.output = inputs, output
        expected = {**inputs, output: (512,) if output == "embedding" else (3, SIZE, SIZE)}
        if names != set(expected):
            raise ValueError("MobileFSGAN engine tensor names mismatch")
        for name, tail in expected.items():
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise ValueError("MobileFSGAN engine expects FP32 IO")
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input != (name in inputs):
                raise ValueError("MobileFSGAN engine tensor mode mismatch")
            if is_input:
                low, _, high = self.engine.get_tensor_profile_shape(name, 0)
                if (
                    tuple(low[1:]) != tail
                    or tuple(high[1:]) != tail
                    or low[0] > 1
                    or high[0] < max_batch
                ):
                    raise ValueError("MobileFSGAN engine batch profile mismatch")
        self.expected = expected
        self.max_batch, self.stream, self.graphs = max_batch, stream, graphs
        self.states = {}
        self.workspace = None

    def state(self, batch):
        if not 1 <= batch <= self.max_batch:
            raise ValueError("MobileFSGAN batch outside configured bounds")
        if batch in self.states:
            return self.states[batch]
        torch = self.torch
        with CUDA_CAPTURE_LOCK, torch.inference_mode(), torch.cuda.stream(self.stream):
            context = self.engine.create_execution_context(
                self.trt.ExecutionContextAllocationStrategy.USER_MANAGED
            )
            if context is None:
                raise RuntimeError("MobileFSGAN execution context allocation failed")
            if self.workspace is None:
                for name, tail in self.inputs.items():
                    if not context.set_input_shape(name, (self.max_batch, *tail)):
                        raise ValueError("MobileFSGAN engine rejected maximum batch")
                size = context.update_device_memory_size_for_shapes()
                if size <= 0:
                    raise RuntimeError("MobileFSGAN workspace shape inference failed")
                self.workspace = torch.empty(size, dtype=torch.uint8, device="cuda")
            for name, tail in self.inputs.items():
                if not context.set_input_shape(name, (batch, *tail)):
                    raise ValueError("MobileFSGAN engine rejected batch shape")
            # All contexts execute on one stream under the renderer lock, so their
            # activation workspace can be shared. Captured addresses stay fixed.
            # Allocate for the configured B8 limit, not the bundle's B32 profile.
            size = context.update_device_memory_size_for_shapes()
            if not 0 < size <= self.workspace.numel():
                raise RuntimeError("MobileFSGAN shape exceeds shared workspace")
            context.device_memory = self.workspace.data_ptr()
            buffers = {}
            for name, tail in self.expected.items():
                shape = tuple(context.get_tensor_shape(name))
                if shape != (batch, *tail):
                    raise ValueError("MobileFSGAN resolved shape mismatch")
                buffers[name] = torch.zeros(shape, device="cuda", dtype=torch.float32)
                if not context.set_tensor_address(name, buffers[name].data_ptr()):
                    raise RuntimeError("MobileFSGAN tensor binding failed")
            # Flush shape-dependent work before CUDA capture (TensorRT contract).
            for _ in range(3):
                if not context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("MobileFSGAN warmup execution failed")
            self.stream.synchronize()
            graph = None
            if self.graphs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
                    if not context.execute_async_v3(self.stream.cuda_stream):
                        raise RuntimeError("MobileFSGAN graph capture failed")
                self.stream.synchronize()
            host = {
                name: torch.empty_like(value, device="cpu", pin_memory=True)
                for name, value in buffers.items()
            }
            self.states[batch] = context, buffers, host, graph
        return self.states[batch]

    def run(self, values):
        torch = self.torch
        if set(values) != set(self.inputs):
            raise ValueError("MobileFSGAN engine inputs mismatch")
        batch = len(next(iter(values.values())))
        context, buffers, host, graph = self.state(batch)
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for name, value in values.items():
                if tuple(value.shape) != tuple(buffers[name].shape):
                    raise ValueError("MobileFSGAN input shape mismatch")
                if isinstance(value, np.ndarray):
                    np.copyto(host[name].numpy(), value)
                    buffers[name].copy_(host[name], non_blocking=True)
                else:
                    buffers[name].copy_(value, non_blocking=True)
            if graph is not None:
                graph.replay()
            elif not context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("MobileFSGAN TensorRT execution failed")
        return buffers[self.output]

    def numpy(self, values):
        result = self.run(values)
        host = self.states[len(result)][2][self.output]
        with self.torch.inference_mode(), self.torch.cuda.stream(self.stream):
            host.copy_(result, non_blocking=True)
        self.stream.synchronize()
        return host.numpy().copy()


class MobileFSGANRenderer:
    def __init__(self, bundle=DEFAULT_BUNDLE, *, engine_dir=None, max_batch=8, cuda_graphs=True):
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("MobileFSGAN requires CUDA")
        if not 1 <= max_batch <= 8:
            raise ValueError("MobileFSGAN max_batch must be 1..8")
        bundle = Path(bundle)
        engine_dir = Path(engine_dir) if engine_dir else bundle / "engines"
        self.manifest = validate_bundle(
            bundle,
            engine_dir,
            version=trt.__version__,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
        )
        self.torch, self.stream, self.lock = torch, torch.cuda.Stream(), RLock()
        self.max_batch = max_batch
        self.detector = YuNetLandmarks(bundle / "detector/yunet.onnx")
        self.source_detector = self.detector
        if self.manifest.get("landmarks_engine"):
            from service.yunet_tensorrt import YuNetTensorRT

            self.detector = YuNetTensorRT(engine_dir / self.manifest["landmarks_engine"])
        self.c = TensorRTEngine(
            engine_dir / self.manifest.get("engine_files", {}).get("C", "C_fp32.plan"),
            {"source_img": (3, SIZE, SIZE)},
            "embedding",
            1,
            self.stream,
            False,
        )
        self.g = TensorRTEngine(
            engine_dir / self.manifest.get("engine_files", {}).get("G", "G_fp32.plan"),
            {"target_img": (3, SIZE, SIZE), "source_z": (512,)},
            "image_256",
            max_batch,
            self.stream,
            cuda_graphs,
        )
        self.sources = OrderedDict()
        self.source_cache_size = 640

    def source_latent(self, preset):
        key = (preset.image, preset.sha256)
        if key in self.sources:
            self.sources.move_to_end(key)
            return self.sources[key]
        data = preset.image.read_bytes()
        if preset.sha256 is not None and hashlib.sha256(data).hexdigest() != preset.sha256:
            raise ValueError("preset image SHA-256 does not match catalog")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cannot decode MobileFSGAN source")
        boxes, points = self.source_detector.detect(image)
        order = np.argsort(-np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1))
        if not len(order):
            raise ValueError("no MobileFSGAN source face")
        areas = np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1)[order]
        if len(order) > 1 and areas[1] > areas[0] * 0.3:
            raise ValueError("multiple MobileFSGAN source faces")
        crop = crop_face(image, alignment(points[order[0]]))
        values = self.c.numpy({"source_img": image_tensor(crop)[None]})
        if not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values), 1, atol=1e-3):
            raise ValueError("invalid MobileFSGAN source embedding")
        self.sources[key] = values[0].copy()
        if len(self.sources) > self.source_cache_size:
            self.sources.popitem(last=False)
        return self.sources[key]

    def warmup(self, presets):
        with self.lock:
            self.c.state(1)
            for batch in range(1, self.max_batch + 1):
                self.g.state(batch)
            # Cache embeddings, not an entire generator per source identity.
            for preset in presets:
                try:
                    self.source_latent(preset)
                except Exception:
                    LOGGER.warning("MobileFSGAN preset unavailable: %s", preset.key, exc_info=True)
            if presets and not self.sources:
                raise ValueError("no usable MobileFSGAN source presets")
            LOGGER.info(
                "MobileFSGAN warmup complete: %d embeddings, max_batch=%d, precision=%s",
                len(self.sources),
                self.max_batch,
                self.manifest["precision"],
            )

    def prepare_frame(self, image):
        return MobileFSGANFrame(image, self.detector)

    def render_crops(self, crops, presets):
        if len(crops) != len(presets):
            raise ValueError("MobileFSGAN crop/source count mismatch")
        results = [None] * len(crops)
        with self.lock:
            for start in range(0, len(crops), self.max_batch):
                indices, latents, tensors = [], [], []
                for i in range(start, min(start + self.max_batch, len(crops))):
                    try:
                        latent = self.source_latent(presets[i])
                        tensor = image_tensor(crops[i])
                        if tensor.shape != (3, SIZE, SIZE) or not np.isfinite(tensor).all():
                            raise ValueError("invalid MobileFSGAN target crop")
                        indices.append(i)
                        latents.append(latent)
                        tensors.append(tensor)
                    except Exception as error:
                        results[i] = error
                if not indices:
                    continue
                try:
                    values = self.g.numpy(
                        {"target_img": np.stack(tensors), "source_z": np.stack(latents)}
                    )
                    for i, value in zip(indices, values, strict=True):
                        results[i] = (
                            value
                            if np.isfinite(value).all()
                            else ValueError("non-finite MobileFSGAN output")
                        )
                except Exception as error:
                    for i in indices:
                        results[i] = error
        return results

    def swap_many(self, output, jobs, *, target):
        with self.lock:
            errors, prepared, crops, presets = [None] * len(jobs), [], [], []
            try:
                target.prime([item["bbox"] for item, _ in jobs])
            except Exception:
                LOGGER.warning(
                    "MobileFSGAN full-frame landmarks failed; retrying ROIs", exc_info=True
                )
            for index, (item, preset) in enumerate(jobs):
                try:
                    matrix = alignment(target.landmarks(item["bbox"]))
                    crop = crop_face(target.image, matrix)
                    prepared.append((index, matrix))
                    crops.append(crop)
                    presets.append(preset)
                except Exception as error:
                    errors[index] = error
            values = self.render_crops(crops, presets)
            for (index, matrix), value in zip(prepared, values, strict=True):
                try:
                    if isinstance(value, Exception):
                        raise value
                    paste_back(output, value, matrix, jobs[index][0]["mask_polygon"])
                except Exception as error:
                    errors[index] = error
            return errors
