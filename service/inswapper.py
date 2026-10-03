"""Optional ONNX InSwapper-128 adapter using YuNet landmarks and ArcFace identity."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from service.face_presets import FacePreset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWAPPER = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
DEFAULT_ARCFACE = ROOT / "models" / "face_swap" / "w600k_r50.onnx"
DEFAULT_YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def alignment(landmarks: np.ndarray, size: int) -> np.ndarray:
    reference = REFERENCE.copy()
    # InsightFace's 128px template adds an 8px horizontal margin to 112px.
    if size == 128:
        reference[:, 0] += 8
    elif size != 112:
        raise ValueError("ArcFace alignment supports only 112 or 128 pixels")
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, reference, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("face landmark alignment failed")
    return matrix


class InSwapperRenderer:
    """One serialized adapter with bounded cached source latents; no model downloads."""

    def __init__(self, swapper: Path, arcface: Path, yunet: Path, provider: str = "cpu"):
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        for path in (swapper, arcface, yunet):
            if not path.is_file():
                raise FileNotFoundError(f"face swap model not found: {path}")
        providers = ["CPUExecutionProvider"]
        if provider == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider unavailable; install onnxruntime-gpu")
            providers.insert(0, "CUDAExecutionProvider")
        elif provider != "cpu":
            raise ValueError("InSwapper provider must be cpu or cuda")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.swapper = ort.InferenceSession(str(swapper), sess_options=options, providers=providers)
        self.arcface = ort.InferenceSession(str(arcface), sess_options=options, providers=providers)
        self.inputs = self.swapper.get_inputs()
        if [entry.shape for entry in self.inputs] != [[1, 3, 128, 128], [1, 512]]:
            raise ValueError(
                "expected fixed InSwapper-128 image and 512-dimensional identity inputs"
            )
        graph = onnx.load(str(swapper), load_external_data=False)
        self.emap = numpy_helper.to_array(graph.graph.initializer[-1]).copy()
        if self.emap.shape != (512, 512) or not np.isfinite(self.emap).all():
            raise ValueError("invalid InSwapper identity projection")
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.65, 0.3, 5000)
        self.sources: OrderedDict[Path, np.ndarray] = OrderedDict()

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        return [] if faces is None else list(faces)

    def source_latent(self, preset: FacePreset) -> np.ndarray:
        if preset.image in self.sources:
            self.sources.move_to_end(preset.image)
            return self.sources[preset.image]
        image = cv2.imread(str(preset.image))
        if image is None:
            raise ValueError("preset image could not be decoded")
        faces = self.detect(image)
        if len(faces) != 1:
            raise ValueError("preset image must contain exactly one detectable face")
        matrix = alignment(faces[0][4:14].reshape(5, 2), 112)
        crop = cv2.warpAffine(image, matrix, (112, 112))
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        embedding = self.arcface.run(None, {self.arcface.get_inputs()[0].name: blob})[0]
        if embedding.shape != (1, 512) or not np.isfinite(embedding).all():
            raise ValueError("invalid ArcFace source embedding")
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        latent = embedding @ self.emap
        norm = float(np.linalg.norm(latent))
        if not np.isfinite(latent).all() or norm <= 1e-12:
            raise ValueError("invalid source identity latent")
        latent = (latent / norm).astype(np.float32)
        self.sources[preset.image] = latent
        if len(self.sources) > 64:
            self.sources.popitem(last=False)
        return latent

    def swap(self, image: np.ndarray, item: dict[str, Any], preset: FacePreset) -> np.ndarray:
        # Detect within a padded YOLO ROI to associate landmarks with this track.
        x1, y1, x2, y2 = np.asarray(item["bbox"], dtype=float)
        pad = max(x2 - x1, y2 - y1) * 0.3
        left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        right, bottom = min(image.shape[1], int(x2 + pad)), min(image.shape[0], int(y2 + pad))
        roi = image[top:bottom, left:right]
        candidates = self.detect(roi)
        best = None
        best_iou = 0.25
        for face in candidates:
            bx, by, bw, bh = face[:4]
            bx, by = bx + left, by + top
            intersection = max(0, min(x2, bx + bw) - max(x1, bx)) * max(
                0, min(y2, by + bh) - max(y1, by)
            )
            union = (x2 - x1) * (y2 - y1) + bw * bh - intersection
            iou = intersection / max(union, 1)
            if iou > best_iou:
                best, best_iou = face, iou
        if best is None:
            raise ValueError("no aligned face matching the tracked bbox")
        landmarks = best[4:14].reshape(5, 2).copy() + np.array([left, top])
        matrix = alignment(landmarks, 128)
        crop = cv2.warpAffine(image, matrix, (128, 128))
        blob = cv2.dnn.blobFromImage(crop, 1 / 255, (128, 128), swapRB=True)
        values = self.swapper.run(
            None,
            {
                self.inputs[0].name: blob,
                self.inputs[1].name: self.source_latent(preset),
            },
        )[0]
        if values.shape != (1, 3, 128, 128) or not np.isfinite(values).all():
            raise ValueError("invalid InSwapper result")
        fake = np.clip(values[0].transpose(1, 2, 0)[:, :, ::-1] * 255, 0, 255).astype(np.uint8)
        inverse = cv2.invertAffineTransform(matrix)
        size = (image.shape[1], image.shape[0])
        mapped = cv2.warpAffine(fake, inverse, size)
        mask = np.zeros((128, 128), np.uint8)
        mask[8:-8, 8:-8] = 255
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        alpha = cv2.warpAffine(mask, inverse, size).astype(np.float32) / 255
        polygon_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(polygon_mask, [np.rint(item["mask_polygon"]).astype(np.int32)], 1)
        alpha *= polygon_mask
        if not np.any(alpha > 0.5):
            raise ValueError("swapped face does not overlap its segmentation mask")
        alpha = alpha[:, :, None]
        return np.rint(mapped * alpha + image * (1 - alpha)).astype(np.uint8)
