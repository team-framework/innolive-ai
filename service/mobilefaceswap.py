"""MobileFaceSwap-224 Paddle inference using the original, unconverted weights.

Model/alignment reference: Seanseattle/MobileFaceSwap @ 29552474c2f621c27818ff7fc32447e1a60a96fb.
SCRFD output contract: deepinsight/insightface python-package/model_zoo/scrfd.py.
Paddle and ONNX Runtime are optional and imported only during renderer loading.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from service.face_presets import FacePreset

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "mobilefaceswap"
SIZE = 224
# Full affine least-squares alignment, as used by the original MobileFaceSwap demo.
REFERENCE = (
    np.array(
        [
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.3655014],
            [70.72990036, 92.20410156],
        ],
        dtype=np.float64,
    )
    * 2
)


def alignment(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("expected five finite MobileFaceSwap landmarks")
    transform, _, rank, _ = np.linalg.lstsq(
        np.column_stack((points, np.ones(5))), REFERENCE, rcond=None
    )
    matrix = transform.T.astype(np.float32)
    if rank != 3 or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
        raise ValueError("degenerate MobileFaceSwap alignment")
    return matrix


def image_tensor(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255


def source_tensor(aligned: np.ndarray) -> np.ndarray:
    # The source is first aligned at 224, then resized to 112 (not aligned at 112).
    values = image_tensor(cv2.resize(aligned, (112, 112)))
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    return (values - mean) / std


def paste_back(image, generated, mask, matrix, polygon):
    if generated.shape != (1, 3, SIZE, SIZE) or mask.shape != (1, 1, SIZE, SIZE):
        raise ValueError("invalid MobileFaceSwap output shapes")
    if not np.isfinite(generated).all() or not np.isfinite(mask).all():
        raise ValueError("non-finite MobileFaceSwap output")
    fake = generated[0].transpose(1, 2, 0)[:, :, ::-1] * 255
    # Match upstream dealign(), then additionally restrict to the YOLO face mask.
    alpha = mask[0, 0].copy()
    alpha[alpha > 0.001] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    alpha = cv2.erode(cv2.dilate(alpha, kernel), kernel, iterations=2)
    alpha = cv2.blur(alpha, (7, 7))
    edge = np.zeros((SIZE, SIZE), np.float32)
    edge[10:-10, 10:-10] = 1
    alpha *= cv2.blur(edge, (11, 11))
    # Upstream uses the pseudoinverse of the homogeneous affine transform.
    inverse = np.linalg.pinv(np.vstack((matrix, [0, 0, 1])))[:2]
    size = (image.shape[1], image.shape[0])
    mapped = cv2.warpAffine(fake, inverse, size)
    alpha = cv2.warpAffine(alpha, inverse, size)
    segmentation = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(segmentation, [np.rint(polygon).astype(np.int32)], 1)
    alpha = np.clip(alpha, 0, 1) * segmentation
    if not np.any(alpha > 0.5):
        raise ValueError("MobileFaceSwap output does not overlap the tracked face mask")
    alpha = alpha[:, :, None]
    return np.clip(mapped * alpha + image * (1 - alpha), 0, 255).astype(np.uint8)


class ScrfdLandmarks:
    """Local SCRFD-10G BNKPS, 640px input, threshold 0.6, inclusive-pixel NMS 0.4."""

    def __init__(self, path: Path, provider: str = "cpu"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.add_free_dimension_override_by_name("?", 640)
        names = {
            "cpu": "CPUExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }
        if provider not in names or names[provider] not in ort.get_available_providers():
            raise RuntimeError(f"SCRFD provider unavailable: {provider}")
        providers: list[Any] = ["CPUExecutionProvider"]
        if provider == "coreml":
            providers.insert(
                0,
                (
                    names[provider],
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "CPUAndGPU",
                        "RequireStaticInputShapes": "1",
                        "EnableOnSubgraphs": "0",
                    },
                ),
            )
        elif provider == "cuda":
            # Use the CUDA/cuDNN libraries supplied by the matching PyTorch wheel.
            # CUDA wheels are a separate install from the Mac/CPU ORT dependency.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers.insert(0, names[provider])
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if names[provider] not in self.session.get_providers():
            raise RuntimeError(f"SCRFD failed to initialize {provider}")
        self.session.disable_fallback()
        if len(self.session.get_inputs()) != 1 or len(self.session.get_outputs()) != 9:
            raise ValueError("expected SCRFD with three strides and five-point landmarks")
        self.input_name = self.session.get_inputs()[0].name
        self.centers = []
        for stride in (8, 16, 32):
            grid = np.stack(np.mgrid[: 640 // stride, : 640 // stride][::-1], axis=-1)
            self.centers.append(np.repeat(grid.reshape(-1, 2), 2, axis=0) * stride)

    def detect(self, image: np.ndarray, *, input_size: int = 640) -> tuple[np.ndarray, np.ndarray]:
        if input_size not in (320, 640):
            raise ValueError("SCRFD image size must be 320 or 640")
        height, width = image.shape[:2]
        if min(height, width) < 2:
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
        # Keep the model's static 640 canvas, but allow a smaller image inside it.
        # Enlarging a tight webcam face ROI to 640 can exceed SCRFD's face scale.
        factor = input_size / max(height, width)
        resized = cv2.resize(image, (max(1, int(width * factor)), max(1, int(height * factor))))
        # Upstream restores both coordinates with the actual resized height ratio.
        scale = resized.shape[0] / height
        canvas = np.zeros((640, 640, 3), np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (640, 640), (127.5,) * 3, swapRB=True)
        outputs = self.session.run(None, {self.input_name: blob})
        boxes, landmarks = [], []
        for index, stride in enumerate((8, 16, 32)):
            scores = outputs[index].reshape(-1)
            distances = outputs[index + 3].reshape(-1, 4) * stride
            offsets = outputs[index + 6].reshape(-1, 5, 2) * stride
            centers = self.centers[index]
            if not (len(scores) == len(distances) == len(offsets) == len(centers)):
                raise ValueError("invalid SCRFD output dimensions")
            selected = np.flatnonzero(scores >= 0.6)
            centers = centers[selected]
            distances = distances[selected]
            xyxy = np.column_stack((centers - distances[:, :2], centers + distances[:, 2:]))
            boxes.append(np.column_stack((xyxy / scale, scores[selected])))
            landmarks.append((centers[:, None] + offsets[selected]) / scale)
        boxes = np.concatenate(boxes).astype(np.float32)
        landmarks = np.concatenate(landmarks).astype(np.float32)
        if not np.isfinite(boxes).all() or not np.isfinite(landmarks).all():
            raise ValueError("non-finite SCRFD output")
        order = np.argsort(-boxes[:, 4], kind="stable")
        keep = []
        areas = np.prod(boxes[:, 2:4] - boxes[:, :2] + 1, axis=1)
        while order.size:
            current, rest = order[0], order[1:]
            keep.append(current)
            overlap = np.maximum(
                0,
                np.minimum(boxes[current, 2:4], boxes[rest, 2:4])
                - np.maximum(boxes[current, :2], boxes[rest, :2])
                + 1,
            ).prod(axis=1)
            iou = overlap / np.maximum(areas[current] + areas[rest] - overlap, 1e-12)
            order = rest[iou <= 0.4]
        return boxes[keep], landmarks[keep]


class MobileFaceSwapFrame:
    """Original pixels and fallback detections, scoped to exactly one frame."""

    def __init__(self, image: np.ndarray, detector: ScrfdLandmarks):
        self.image = image
        self.detector = detector
        self.full_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def match(boxes: np.ndarray, points: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        if not len(boxes):
            return None
        overlap = np.maximum(
            0, np.minimum(boxes[:, 2:4], bbox[2:]) - np.maximum(boxes[:, :2], bbox[:2])
        ).prod(axis=1)
        union = (
            np.prod(bbox[2:] - bbox[:2]) + np.prod(boxes[:, 2:4] - boxes[:, :2], axis=1) - overlap
        )
        iou = overlap / np.maximum(union, 1)
        best = int(np.argmax(iou))
        return points[best] if iou[best] > 0.25 else None

    def landmarks(self, bbox: Any) -> np.ndarray:
        bbox = np.asarray(bbox, dtype=float)
        x1, y1, x2, y2 = bbox
        pad = max(x2 - x1, y2 - y1) * 0.3
        left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        right, bottom = (
            min(self.image.shape[1], int(x2 + pad)),
            min(self.image.shape[0], int(y2 + pad)),
        )
        boxes, points = self.detector.detect(self.image[top:bottom, left:right], input_size=320)
        boxes = boxes.copy()
        boxes[:, :4] += [left, top, left, top]
        matched = self.match(boxes, points + np.array([left, top]), bbox)
        if matched is not None:
            return matched
        # Tight/edge crops can lose context. Retry original pixels at two scales,
        # sharing these detections with other tracks in this frame only.
        for size in (640, 320):
            if size not in self.full_frame:
                self.full_frame[size] = self.detector.detect(self.image, input_size=size)
            matched = self.match(*self.full_frame[size], bbox)
            if matched is not None:
                return matched
        raise ValueError("no MobileFaceSwap landmarks matching the tracked bbox at either scale")


def load_weights(model: Any, path: Path, paddle: Any) -> dict:
    weights = paddle.load(str(path))
    expected = model.state_dict()
    if set(expected) != set(weights) or any(
        tuple(weights[key].shape) != tuple(value.shape) for key, value in expected.items()
    ):
        raise ValueError(f"MobileFaceSwap checkpoint does not match the official model: {path}")
    missing, unexpected = model.set_state_dict(weights)
    if missing or unexpected:
        raise ValueError(f"incomplete MobileFaceSwap checkpoint: {path}")
    model.eval()
    return weights


class MobileFaceSwapRenderer:
    """CPU Paddle models with a bounded cache of source-conditioned 224px networks."""

    def __init__(self, model_dir: Path, provider: str = "cpu", *, landmarks_provider: str = "cpu"):
        if provider != "cpu":
            raise ValueError("MobileFaceSwap currently supports --face-swap-provider cpu only")
        swapper = model_dir / "MobileFaceSwap_224.pdparams"
        arcface = model_dir / "arcface.pdparams"
        detector = model_dir / "landmarks" / "scrfd_10g_bnkps.onnx"
        for path in (swapper, arcface, detector):
            if not path.is_file():
                raise FileNotFoundError(f"MobileFaceSwap model not found: {path}")
        import paddle

        from service._vendor.mobilefaceswap.arcface import IRBlock, ResNet
        from service._vendor.mobilefaceswap.model import BuildFaceSwap, FaceSwap

        self.paddle = paddle
        self.face_swap_class = FaceSwap
        paddle.set_device("cpu")
        self.arcface = ResNet(IRBlock, [3, 4, 23, 3])
        load_weights(self.arcface, arcface, paddle)
        self.weights = load_weights(BuildFaceSwap(), swapper, paddle)
        self.detector = ScrfdLandmarks(detector, landmarks_provider)
        self.sources: OrderedDict[tuple[Path, str | None], Any] = OrderedDict()

    def source_model(self, preset: FacePreset):
        key = (preset.image, preset.sha256)
        if key in self.sources:
            self.sources.move_to_end(key)
            return self.sources[key]
        data = preset.image.read_bytes()
        if preset.sha256 is not None and hashlib.sha256(data).hexdigest() != preset.sha256:
            raise ValueError("preset image SHA-256 does not match the catalog")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("preset image could not be decoded")
        boxes, points = self.detector.detect(image)
        if len(boxes) != 1:
            raise ValueError("preset image must contain exactly one detectable face")
        crop = cv2.warpAffine(image, alignment(points[0]), (SIZE, SIZE))
        with self.paddle.no_grad():
            embedding, features = self.arcface(self.paddle.to_tensor(source_tensor(crop)))
            norm = self.paddle.norm(embedding, p=2, axis=1, keepdim=True)
            if not np.isfinite(embedding.numpy()).all() or float(norm.item()) <= 1e-12:
                raise ValueError("invalid MobileFaceSwap source embedding")
            model = self.face_swap_class(use_gpu=False)
            model.set_model_param(embedding / norm, features, self.weights)
            model.eval()
        self.sources[key] = model
        if len(self.sources) > 16:
            self.sources.popitem(last=False)
        return model

    def prepare_frame(self, image: np.ndarray) -> MobileFaceSwapFrame:
        return MobileFaceSwapFrame(image, self.detector)

    def swap(
        self,
        image: np.ndarray,
        item: dict[str, Any],
        preset: FacePreset,
        *,
        target: MobileFaceSwapFrame | None = None,
    ) -> np.ndarray:
        target = target if target is not None else self.prepare_frame(image)
        matrix = alignment(target.landmarks(item["bbox"]))
        crop = cv2.warpAffine(target.image, matrix, (SIZE, SIZE))
        generated, mask = self.render_crop(crop, preset)
        return paste_back(image, generated, mask, matrix, item["mask_polygon"])

    def render_crop(self, crop: np.ndarray, preset: FacePreset) -> tuple[np.ndarray, np.ndarray]:
        model = self.source_model(preset)
        with self.paddle.no_grad():
            generated, mask = model(self.paddle.to_tensor(image_tensor(crop)))
        return generated.numpy(), mask.numpy()
