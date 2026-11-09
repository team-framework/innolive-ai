"""RF-DETR Seg Small inference without detector-specific result wrappers.

The checkpoint uses contiguous face/plate IDs and a final background logit.
RGB square resize and ImageNet normalization match RF-DETR 1.11.2. Both
backends share the same score decoding and contour conversion.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from service.cuda_capture import CUDA_CAPTURE_LOCK
from service.detection import EXPECTED_CLASS_NAMES, Boxes, Segmentation

RESOLUTION = 624
MODEL_NAME = "RFDETRSegSmall"
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def selected_pairs(logits, threshold, max_detections):
    probabilities = 1 / (1 + np.exp(-np.clip(logits[:, :2], -80, 80)))
    flat = probabilities.ravel()
    selected = np.argsort(-flat, kind="stable")[:max_detections]
    selected = selected[flat[selected] >= threshold]
    return selected // 2, selected % 2, flat[selected]


def checkpoint_metadata(path: Path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config", {})
    args = checkpoint.get("args", {})
    names = (
        args.get("class_names") if isinstance(args, dict) else getattr(args, "class_names", None)
    )
    if checkpoint.get("model_name") != MODEL_NAME or not config.get("segmentation_head"):
        raise ValueError("expected RFDETRSegSmall segmentation checkpoint")
    if names not in (["face", "plate"], ["face", "number_plate"]):
        raise ValueError(f"expected checkpoint classes face/plate, got {names!r}")
    if config.get("num_classes") != 2 or config.get("resolution") != RESOLUTION:
        raise ValueError("expected 2-class RF-DETR checkpoint at resolution 624")
    weights = checkpoint.get("model", {})
    if tuple(weights.get("class_embed.weight", np.empty(0)).shape) != (3, 256):
        raise ValueError("expected two foreground classes and final background slot")
    return {
        "model_name": MODEL_NAME,
        "resolution": RESOLUTION,
        "classes": names,
        "rfdetr_version": checkpoint.get("rfdetr_version"),
        "epoch": checkpoint.get("epoch"),
    }


def decode(dets, labels, masks, shape, threshold=0.01, max_detections=100):
    """Decode named raw outputs; never admit the background slot to tracking."""
    boxes = np.asarray(dets, dtype=np.float32).reshape(-1, 4)
    logits = np.asarray(labels, dtype=np.float32).reshape(-1, 3)
    masks = np.asarray(masks, dtype=np.float32)
    if masks.ndim == 4:
        masks = masks[0]
    if len(boxes) != len(logits) or masks.shape != (len(boxes), RESOLUTION // 4, RESOLUTION // 4):
        raise ValueError("RF-DETR output shapes do not match the trained contract")
    # Stable ordering agrees with the upstream export decoder on tied logits.
    queries, classes, scores = selected_pairs(logits, threshold, max_detections)
    selected_boxes = boxes[queries]
    xyxy = np.concatenate(
        (
            selected_boxes[:, :2] - selected_boxes[:, 2:] / 2,
            selected_boxes[:, :2] + selected_boxes[:, 2:] / 2,
        ),
        axis=1,
    )
    height, width = shape[:2]
    xyxy *= np.asarray((width, height, width, height), dtype=np.float32)
    xyxy = np.clip(xyxy, (0, 0, 0, 0), (width - 1, height - 1, width - 1, height - 1))
    valid = ((xyxy[:, 2:] - xyxy[:, :2]) > 0).all(axis=1)
    xyxy, queries, classes, scores = xyxy[valid], queries[valid], classes[valid], scores[valid]
    polygons = _Contours(masks, queries, width, height) if len(queries) else []
    return Segmentation(Boxes(np.column_stack((xyxy, scores, classes)), shape[:2]), polygons)


class _Contours:
    """Only tracked masks become polygons; own planes survive the next B1 inference."""

    def __init__(self, masks, queries, width, height):
        unique, inverse = np.unique(queries, return_inverse=True)
        self.planes = masks[unique]  # Advanced indexing owns the selected mask planes.
        self.indices = inverse
        self.scale = np.asarray((width / RESOLUTION, height / RESOLUTION), np.float32)
        self.cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        plane = int(self.indices[index])
        if plane not in self.cache:
            field = cv2.resize(
                self.planes[plane], (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR
            )
            contours, _ = cv2.findContours(
                (field > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # A single wire polygon must protect all disconnected components.
            if len(contours) > 1:
                contours = [cv2.convexHull(np.concatenate(contours))]
            polygon = (
                contours[0].reshape(-1, 2).astype(np.float32)
                if contours
                else np.empty((0, 2), np.float32)
            )
            self.cache[plane] = polygon * self.scale
        return self.cache[plane]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class RFDETRDetector:
    """Serialized B1 runner. Lab batches preserve ordering through this same path."""

    names = EXPECTED_CLASS_NAMES

    def __init__(self, artifact: Path, backend: str, device: str):
        import torch

        self.torch = torch
        self.backend = backend
        self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        self._pixels = OrderedDict()
        self.mean = torch.tensor(NORMALIZATION_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(NORMALIZATION_STD, device=self.device).view(1, 3, 1, 1)
        if backend == "tensorrt":
            self._init_trt(artifact)
        else:
            from rfdetr import RFDETR

            checkpoint_metadata(artifact)
            self.wrapper = RFDETR.from_checkpoint(str(artifact), device=str(self.device))
            self.model = self.wrapper.model.model.eval().to(self.device)
            self.model.export()

    def _init_trt(self, artifact):
        import tensorrt as trt

        torch = self.torch
        with torch.cuda.device(self.device):
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(artifact.read_bytes())
            if self.engine is None:
                raise RuntimeError("RF-DETR TensorRT engine deserialization failed")
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            if len(inputs) != 1 or set(names) - set(inputs) != {"dets", "labels", "masks"}:
                raise ValueError(f"RF-DETR engine IO names mismatch: {names}")
            self.input_name = inputs[0]
            if tuple(self.engine.get_tensor_shape(self.input_name)) != (
                1,
                3,
                RESOLUTION,
                RESOLUTION,
            ):
                raise ValueError("RF-DETR requires static B1 624 engine")
            if any(self.engine.get_tensor_dtype(n) != trt.float32 for n in names):
                raise ValueError("RF-DETR engine must expose FP32 IO")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("RF-DETR execution context creation failed")
            self.stream = torch.cuda.Stream(device=self.device)
            self.buffers = {
                n: torch.empty(
                    tuple(self.engine.get_tensor_shape(n)), dtype=torch.float32, device=self.device
                )
                for n in names
            }
            self.host = {
                n: torch.empty_like(v, device="cpu", pin_memory=True)
                for n, v in self.buffers.items()
                if n != self.input_name
            }
            for name, tensor in self.buffers.items():
                self.context.set_tensor_address(name, tensor.data_ptr())
            self.finite_host = torch.empty((), dtype=torch.bool, pin_memory=True)
            self.selected_host = torch.empty_like(self.host["masks"][0], pin_memory=True)
            self.selected_device = torch.empty_like(self.buffers["masks"][0])
            queries = self.selected_host.shape[0]
            self.indices_host = torch.empty(queries, dtype=torch.int64, pin_memory=True)
            self.indices_device = torch.empty(queries, dtype=torch.int64, device=self.device)

    def _preprocess(self, pixels):
        torch = self.torch
        rgb = pixels.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        # Resize before RGB reorder to avoid an original-resolution channel copy.
        rgb = torch.nn.functional.interpolate(
            rgb, (RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False
        )
        return (rgb.flip(1) / 255 - self.mean) / self.std

    def _trt(self, image, *, mask_threshold=None, max_detections=100):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RF-DETR requires uint8 HWC BGR input")
        torch = self.torch
        shape = tuple(image.shape)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.inference_mode():
            if shape not in self._pixels:
                if len(self._pixels) >= 4:
                    self.stream.synchronize()
                    self._pixels.popitem(last=False)
                host = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
                device = torch.zeros(shape, dtype=torch.uint8, device=self.device)

                def launch():
                    self.buffers[self.input_name].copy_(self._preprocess(device))
                    if not self.context.execute_async_v3(self.stream.cuda_stream):
                        raise RuntimeError("RF-DETR TensorRT execution failed")
                    # Inspect before host-side decoding so NaNs cannot erase protected objects.
                    return torch.stack(
                        [
                            v.isfinite().all()
                            for n, v in self.buffers.items()
                            if n != self.input_name
                        ]
                    ).all()

                with CUDA_CAPTURE_LOCK:
                    for _ in range(3):
                        launch()
                    self.stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        finite = launch()
                self._pixels[shape] = host, device, graph, finite
            self._pixels.move_to_end(shape)
            host, device, graph, finite = self._pixels[shape]
            np.copyto(host.numpy(), image)
            device.copy_(host, non_blocking=True)
            graph.replay()
            for name, dest in self.host.items():
                if name == "masks" and mask_threshold is not None:
                    continue
                dest.copy_(self.buffers[name], non_blocking=True)
            self.finite_host.copy_(finite, non_blocking=True)
            self.stream.synchronize()
            if not self.finite_host.item():
                raise RuntimeError("non-finite RF-DETR TensorRT output")
            if mask_threshold is not None:
                # Only requested query planes cross PCIe. Labels/boxes are tiny; a second
                # synchronization costs less than transferring 100 full float mask planes.
                queries, _, _ = selected_pairs(
                    self.host["labels"].numpy()[0], mask_threshold, max_detections
                )
                queries = np.unique(queries)
                count = len(queries)
                if count:
                    np.copyto(self.indices_host.numpy()[:count], queries)
                    self.indices_device[:count].copy_(self.indices_host[:count], non_blocking=True)
                    torch.index_select(
                        self.buffers["masks"][0],
                        0,
                        self.indices_device[:count],
                        out=self.selected_device[:count],
                    )
                    self.selected_host[:count].copy_(
                        self.selected_device[:count], non_blocking=True
                    )
                    self.stream.synchronize()
                    self.host["masks"].numpy()[0, queries] = self.selected_host.numpy()[:count]
        return {name: value.numpy() for name, value in self.host.items()}

    def raw(self, image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RF-DETR requires uint8 HWC BGR input")
        if self.backend == "tensorrt":
            return self._trt(image)
        torch = self.torch
        with torch.inference_mode():
            tensor = self._preprocess(
                torch.as_tensor(np.ascontiguousarray(image), device=self.device)
            )
            output = self.model(tensor)
            result = {
                n: v.float().cpu().numpy()
                for n, v in zip(("dets", "labels", "masks"), output, strict=True)
            }
        if not all(np.isfinite(v).all() for v in result.values()):
            raise RuntimeError("non-finite RF-DETR PyTorch output")
        return result

    def predict(self, image, *, threshold=0.01, max_detections=100):
        outputs = (
            self._trt(image, mask_threshold=threshold, max_detections=max_detections)
            if self.backend == "tensorrt"
            else self.raw(image)
        )
        return decode(
            outputs["dets"],
            outputs["labels"],
            outputs["masks"],
            image.shape,
            threshold=threshold,
            max_detections=max_detections,
        )
