"""Lazy, checkpoint-verified face attribute inference for experimental anonymization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_MODEL = ROOT / "models" / "face_metadata_extracter.pt"
CLASSES = {
    "gender": ("female", "male"),
    "age": ("10s", "20s", "30s", "over40s"),
    "glasses": ("off", "on"),
    "exp": ("none", "anger", "disgust", "fear", "happy", "sad", "surprise"),
}


@dataclass(frozen=True, slots=True)
class FaceAttributes:
    gender: str
    age: str
    glasses: str
    exp: str

    def __post_init__(self) -> None:
        for name, classes in CLASSES.items():
            if getattr(self, name) not in classes:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}")


@dataclass(frozen=True, slots=True)
class AttributePrediction:
    attributes: FaceAttributes
    confidence: dict[str, float]


class MetadataExtractor:
    """Rebuild the supplied MobileNetV3 state dict; never unpickle arbitrary objects."""

    def __init__(self, path: Path = DEFAULT_METADATA_MODEL, device: str = "cpu"):
        import torch
        from torch import nn
        from torchvision.models import mobilenet_v3_large

        # The training checkpoint also contains NumPy RNG state. Allow only the
        # concrete NumPy primitives needed for that state, keeping weights_only.
        allowed = [np._core.multiarray._reconstruct, np.ndarray, np.dtype]
        allowed += [type(np.dtype(kind)) for kind in ("uint32", "int64", "float32", "float64")]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if tuple(config["targets"]) != tuple(CLASSES) or any(
            tuple(config["classes"][name]) != classes for name, classes in CLASSES.items()
        ):
            raise ValueError("metadata checkpoint class order does not match the serving contract")
        if not str(config["architecture"]).startswith("torchvision.mobilenet_v3_large.features"):
            raise ValueError("unsupported metadata backbone")
        self.input_size = int(config["input_size"])
        if not 32 <= self.input_size <= 512:
            raise ValueError("metadata input_size must be in 32..512")

        class AttributeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = mobilenet_v3_large(weights=None).features
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.heads = nn.ModuleDict(
                    {
                        name: nn.Sequential(
                            nn.Linear(960, 128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, len(classes)),
                        )
                        for name, classes in CLASSES.items()
                    }
                )

            def forward(self, inputs):
                features = self.pool(self.features(inputs)).flatten(1)
                return {name: head(features) for name, head in self.heads.items()}

        self.device = torch.device(device)
        self.model = AttributeModel().to(self.device).eval()
        self.model.load_state_dict(checkpoint["model"], strict=True)
        # Training config does not record these values. ImageNet normalization
        # and ReLU are the initial contract; verify against the training code.
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device)[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device)[:, None, None]

    def predict(self, crops: list[np.ndarray]) -> list[AttributePrediction]:
        import torch

        if not crops:
            return []
        inputs = []
        for crop in crops:
            if crop.size == 0 or crop.dtype != np.uint8 or crop.shape[-1] != 3:
                raise ValueError("metadata crop must be a nonempty uint8 BGR image")
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
            )
            tensor = torch.from_numpy(resized).permute(2, 0, 1).to(self.device).float() / 255
            inputs.append((tensor - self.mean) / self.std)
        with torch.inference_mode():
            probabilities = {
                name: values.softmax(1).cpu().numpy()
                for name, values in self.model(torch.stack(inputs)).items()
            }
        predictions = []
        for index in range(len(crops)):
            attributes = {}
            confidence = {}
            for name, classes in CLASSES.items():
                row = probabilities[name][index]
                if not np.isfinite(row).all():
                    raise ValueError("metadata model returned non-finite probabilities")
                choice = int(row.argmax())
                attributes[name] = classes[choice]
                confidence[name] = float(row[choice])
            predictions.append(AttributePrediction(FaceAttributes(**attributes), confidence))
        return predictions
