"""PyTorch inference port of the MobileFaceSwap identity-conditioned UNet.

Adapted from Seanseattle/MobileFaceSwap, commit 29552474c2f621c27818ff7fc32447e1a60a96fb.
Original Apache-2.0 license: service/_vendor/mobilefaceswap/LICENSE.
Changes: torch operators, strict Paddle state conversion, eval BN fusion and device cache.
Source identity/weight prediction runs once per preset using the original Paddle models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections import OrderedDict
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn

from service.cuda_capture import CUDA_CAPTURE_LOCK
from service.face_presets import FacePreset
from service.mobilefaceswap import MobileFaceSwapRenderer, image_tensor

LOGGER = logging.getLogger("innolive.mobilefaceswap")
CACHE_FORMAT = "mobilefaceswap-224-fused-fp32-v1"


def fuse_model(model):
    for layers in [model.final, *list(model.mask)[:5]]:
        for index in range(len(layers) - 1):
            if isinstance(layers[index], nn.Conv2d) and isinstance(
                layers[index + 1], nn.BatchNorm2d
            ):
                layers[index] = torch.nn.utils.fuse_conv_bn_eval(layers[index], layers[index + 1])
                layers[index + 1] = nn.Identity()
    return model


class ConditionedModelCache:
    """Optional CPU weight cache, bound to exact model and source-image hashes."""

    def __init__(self, directory: Path, model_dir: Path):
        digest = hashlib.sha256(CACHE_FORMAT.encode())
        for name in (
            "MobileFaceSwap_224.pdparams",
            "arcface.pdparams",
            "landmarks/scrfd_10g_bnkps.onnx",
        ):
            with (model_dir / name).open("rb") as source:
                digest.update(hashlib.file_digest(source, "sha256").digest())
        self.fingerprint = digest.hexdigest()
        self.directory = directory / self.fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, preset: FacePreset) -> str:
        digest = hashlib.sha256(preset.image.read_bytes()).hexdigest()
        if preset.sha256 is not None and digest != preset.sha256:
            raise ValueError("preset image SHA-256 does not match the catalog")
        return digest

    def load(self, preset: FacePreset):
        key = self.key(preset)
        path = self.directory / f"{key}.pt"
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if (payload["format"], payload["models"], payload["source"]) != (
                CACHE_FORMAT,
                self.fingerprint,
                key,
            ):
                raise ValueError("conditioned model cache metadata mismatch")
            model = fuse_model(MobileFaceSwapUNet().eval())
            model.load_state_dict(payload["weights"], strict=True)
            if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("conditioned model cache contains non-finite weights")
            model.requires_grad_(False)
            return model
        except Exception:
            LOGGER.warning("Ignoring invalid conditioned cache %s", path, exc_info=True)
            return None

    def save(self, preset: FacePreset, model):
        key = self.key(preset)
        payload = {
            "format": CACHE_FORMAT,
            "models": self.fingerprint,
            "source": key,
            "weights": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, suffix=".tmp", delete=False
            ) as file:
                temporary = Path(file.name)
                torch.save(payload, file)
            temporary.replace(self.directory / f"{key}.pt")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class MobileFaceSwapUNet(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = [3, 32, 64, 128, 256, 512]
        self.Encoder = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(i, i, 4, stride=2, padding=1, groups=i), nn.Conv2d(i, o, 1))
                for i, o in pairwise(encoder)
            ]
        )
        self.Decoder = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(i, i, 3, padding=1, groups=i), nn.Conv2d(i, o, 1))
                for i, o in zip([512, 512, 256, 128], [256, 128, 64, 32], strict=True)
            ]
        )
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 8, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Tanh(),
        )
        channels = [512, 128, 64, 32, 8, 2]
        layers = [
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(i, i, 3, padding=1, groups=i),
                nn.Conv2d(i, o, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.1),
            )
            for i, o in pairwise(channels)
        ]
        self.mask = nn.Sequential(*layers, nn.Conv2d(2, 1, 3, padding=1), nn.Sigmoid())
        self.relu = nn.LeakyReLU(0.1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, data):
        x = (data - 0.5) / 0.5
        encoded = []
        for layer in self.Encoder:
            x = self.relu(layer(x))
            encoded.append(x)
        mask = self.mask(x)
        y = encoded[-1]
        for index, layer in enumerate(self.Decoder):
            y = self.relu(layer(self.up(y)))
            if index != len(self.Decoder) - 1:
                y = torch.cat((y, encoded[len(self.Decoder) - 1 - index]), dim=1)
        output = (1 + self.final(y)) / 2
        return output * mask + (1 - mask) * data, mask


def from_paddle(source, *, fuse: bool = True) -> MobileFaceSwapUNet:
    """Load every conditioned tensor; reject missing or unexpected parameters."""
    model = MobileFaceSwapUNet().eval()
    converted = {}
    for name, value in source.swap_model.state_dict().items():
        name = name.replace("._mean", ".running_mean").replace("._variance", ".running_var")
        converted[name] = torch.from_numpy(value.numpy().copy())
    for name, value in model.state_dict().items():
        if name.endswith(".num_batches_tracked"):
            converted[name] = value
    model.load_state_dict(converted, strict=True)
    model.requires_grad_(False)
    if fuse:
        fuse_model(model)
    return model


class CudaGraphModel(nn.Module):
    """One fixed crop and one identity; owned by the bounded source-model cache.

    The face executor serializes calls. Each identity has separate weights, input,
    output and graph storage, so replay never mixes different source identities.
    """

    def __init__(self, model, device, dtype):
        super().__init__()
        with CUDA_CAPTURE_LOCK:
            self._capture(model, device, dtype)

    def _capture(self, model, device, dtype):
        self.model = model
        self.values = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype).contiguous(
            memory_format=torch.channels_last
        )
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                model(self.values)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"),
        ):
            generated, mask = model(self.values)
            self.outputs = generated.float(), mask.float()
        torch.cuda.current_stream(device).wait_stream(stream)

    def forward(self, values):
        if values.shape != self.values.shape:
            raise ValueError("CUDA graph expects one 224px face crop")
        self.values.copy_(values)
        self.graph.replay()
        return self.outputs


class TorchMobileFaceSwapRenderer(MobileFaceSwapRenderer):
    """Per-frame synthesis on MPS/CUDA; first-use source conditioning on Paddle CPU."""

    def __init__(self, model_dir: Path, provider: str, *, precision: str = "fp32"):
        if provider not in {"mps", "cuda"}:
            raise ValueError("torch MobileFaceSwap provider must be mps or cuda")
        if precision not in {"fp32", "fp16"} or (precision == "fp16" and provider != "cuda"):
            raise ValueError("MobileFaceSwap fp16 is supported only on CUDA")
        if provider == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable")
        if provider == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        super().__init__(
            model_dir, "cpu", landmarks_provider="coreml" if provider == "mps" else "cuda"
        )
        self.cuda_graphs = (
            provider == "cuda" and os.environ.get("MOBILEFACESWAP_CUDA_GRAPHS") == "1"
        )
        self.device = torch.device(provider)
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.torch_sources: OrderedDict[tuple[Path, str | None], nn.Module] = OrderedDict()
        self.source_cache_size = int(os.environ.get("MOBILEFACESWAP_SOURCE_CACHE_SIZE", "16"))
        if not 1 <= self.source_cache_size <= 128:
            raise ValueError("MOBILEFACESWAP_SOURCE_CACHE_SIZE must be 1..128")
        self.require_cache = os.environ.get("MOBILEFACESWAP_REQUIRE_CACHE") == "1"
        cache_dir = os.environ.get("MOBILEFACESWAP_CACHE_DIR")
        self.disk_cache = ConditionedModelCache(Path(cache_dir), model_dir) if cache_dir else None
        if self.require_cache and self.disk_cache is None:
            raise ValueError("MOBILEFACESWAP_REQUIRE_CACHE requires MOBILEFACESWAP_CACHE_DIR")
        self.transfer_buffers = None

    def torch_source_model(self, preset: FacePreset) -> nn.Module:
        key = (preset.image, preset.sha256)
        if key not in self.torch_sources:
            model = self.disk_cache.load(preset) if self.disk_cache else None
            if model is None:
                if getattr(self, "require_cache", False):
                    raise RuntimeError(
                        "preset cache unavailable; run prepare_mobilefaceswap_cache.py"
                    )
                model = from_paddle(self.source_model(preset))
                if self.disk_cache:
                    try:
                        self.disk_cache.save(preset, model)
                    except OSError:
                        LOGGER.warning("Could not save conditioned model cache", exc_info=True)
            model = model.to(self.device, dtype=self.dtype, memory_format=torch.channels_last)
            if getattr(self, "cuda_graphs", False):
                model = CudaGraphModel(model, self.device, self.dtype)
            self.torch_sources[key] = model
            if len(self.torch_sources) > getattr(self, "source_cache_size", 16):
                self.torch_sources.popitem(last=False)
        self.torch_sources.move_to_end(key)
        return self.torch_sources[key]

    def warmup(self, presets):
        if self.require_cache:
            missing = [
                preset.key
                for preset in presets
                if not (self.disk_cache.directory / f"{self.disk_cache.key(preset)}.pt").is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} preset caches missing; run prepare_mobilefaceswap_cache.py"
                )
        neutral = [
            p for p in presets if p.attributes.exp == "none" and p.attributes.glasses == "off"
        ]
        for preset in neutral[: self.source_cache_size]:
            self.render_crop(np.zeros((224, 224, 3), np.uint8), preset)
        LOGGER.info("MobileFaceSwap warmup complete: %d source models", len(self.torch_sources))

    def render_crop(self, crop: np.ndarray, preset: FacePreset) -> tuple[np.ndarray, np.ndarray]:
        model = self.torch_source_model(preset)
        with torch.inference_mode():
            values = torch.from_numpy(image_tensor(crop)).to(
                self.device, dtype=self.dtype, memory_format=torch.channels_last
            )
            generated, mask = model(values)
            # CPU copies synchronize the GPU before paste-back and timing completion.
            return generated.float().cpu().numpy(), mask.float().cpu().numpy()

    def render_crops(self, crops, presets):
        if self.device.type != "cuda" or len(crops) <= 1:
            return super().render_crops(crops, presets)
        # Different source identities have different weights: keep their forward
        # passes separate, but transfer crops/results together and wait once per
        # bounded chunk. Copy each graph output before replay can overwrite it.
        if self.transfer_buffers is None:
            self.transfer_buffers = (
                torch.empty((8, 3, 224, 224), dtype=torch.float32, pin_memory=True),
                torch.empty((8, 3, 224, 224), dtype=self.dtype, device=self.device).contiguous(
                    memory_format=torch.channels_last
                ),
                torch.empty((8, 4, 224, 224), dtype=torch.float32, device=self.device),
                torch.empty((8, 4, 224, 224), dtype=torch.float32, pin_memory=True),
            )
        host_input, device_input, device_output, host_output = self.transfer_buffers
        results = []
        for start in range(0, len(crops), 8):
            chunk = crops[start : start + 8]
            models = []
            for preset in presets[start : start + len(chunk)]:
                try:
                    models.append(self.torch_source_model(preset))
                except Exception as error:
                    models.append(error)
            with torch.inference_mode():
                for index, crop in enumerate(chunk):
                    np.copyto(host_input[index].numpy(), image_tensor(crop)[0])
                device_input[: len(chunk)].copy_(host_input[: len(chunk)], non_blocking=True)
                errors = []
                for index, model in enumerate(models):
                    try:
                        if isinstance(model, Exception):
                            raise model
                        generated, mask = model(device_input[index : index + 1])
                        device_output[index : index + 1, :3].copy_(generated)
                        device_output[index : index + 1, 3:].copy_(mask)
                        errors.append(None)
                    except Exception as error:
                        errors.append(error)
                host_output[: len(chunk)].copy_(device_output[: len(chunk)], non_blocking=True)
                torch.cuda.current_stream(self.device).synchronize()
                # Detach from reusable pinned buffers before the next chunk/frame.
                values = host_output[: len(chunk)].numpy().copy()
                results.extend(
                    error if error is not None else (values[i : i + 1, :3], values[i : i + 1, 3:])
                    for i, error in enumerate(errors)
                )
        return results
