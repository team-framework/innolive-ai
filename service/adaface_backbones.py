"""AdaFace-compatible backbones used by the recognition runtime.

The fixed ViT-KP-RPE implementation follows the MIT-licensed CVLFace model
definition while keeping only the inference path used by the published
WebFace12M checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import nn

ADAFACE_ARCHITECTURES = ("ir18", "ir50", "ir101", "vit_base_kprpe")

_IR_UNITS = {
    "ir18": (2, 2, 2, 2),
    "ir50": (3, 4, 14, 3),
    "ir101": (3, 13, 30, 3),
}
_VIT_DEPTH = 24
_VIT_EMBED_DIM = 512
_VIT_HEADS = 16
_VIT_PATCHES_PER_SIDE = 14
_VIT_RPE_BUCKETS = 49


def build_adaface_backbone(architecture: str) -> nn.Module:
    if architecture == "vit_base_kprpe":
        return ViTBaseKPRPE()
    try:
        return AdaFaceIR(_IR_UNITS[architecture])
    except KeyError as error:
        choices = ", ".join(ADAFACE_ARCHITECTURES)
        raise ValueError(
            f"unsupported AdaFace architecture {architecture!r}; choose {choices}"
        ) from error


def checkpoint_backbone_state(
    architecture: str,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if architecture == "vit_base_kprpe":
        prefix = "net."
    elif any(key.startswith("model.") for key in state):
        prefix = "model."
    else:
        return dict(state)
    return {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }


class AdaFaceIR(nn.Module):
    def __init__(self, units: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        channels = ((64, 64), (64, 128), (128, 256), (256, 512))
        body = []
        for (input_channels, output_channels), block_count in zip(channels, units, strict=True):
            body.append(BasicBlockIR(input_channels, output_channels, 2))
            body.extend(
                BasicBlockIR(output_channels, output_channels, 1) for _ in range(block_count - 1)
            )
        self.body = nn.Sequential(*body)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        _keypoints: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.output_layer(self.body(self.input_layer(inputs)))
        norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
        return features / norms, norms


class BasicBlockIR(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        if input_channels == output_channels:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.PReLU(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.res_layer(inputs) + self.shortcut_layer(inputs)


class ViTBaseKPRPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        patch_count = _VIT_PATCHES_PER_SIDE**2
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count, _VIT_EMBED_DIM))
        self.patch_embed = PatchEmbed()
        self.blocks = nn.ModuleList(ViTBlock() for _ in range(_VIT_DEPTH))
        self.norm = nn.LayerNorm(_VIT_EMBED_DIM)
        self.feature = nn.Sequential(
            nn.Linear(_VIT_EMBED_DIM * patch_count, _VIT_EMBED_DIM, bias=False),
            nn.BatchNorm1d(_VIT_EMBED_DIM, eps=2e-5),
            nn.Linear(_VIT_EMBED_DIM, _VIT_EMBED_DIM, bias=False),
            nn.BatchNorm1d(_VIT_EMBED_DIM, eps=2e-5),
        )
        self.keypoint_linear = nn.Linear(
            10,
            _VIT_RPE_BUCKETS * _VIT_HEADS * _VIT_DEPTH,
        )
        self.register_buffer("_patch_centers", _patch_centers(), persistent=False)
        self.register_buffer("_rpe_bucket_ids", _product_bucket_ids(), persistent=False)

    def forward(
        self,
        inputs: torch.Tensor,
        keypoints: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keypoints is None or keypoints.shape != (inputs.shape[0], 5, 2):
            raise ValueError("ViT-KP-RPE requires five normalized keypoints per image")

        features = self.patch_embed(inputs) + self.pos_embed
        contexts = self._keypoint_contexts(keypoints, features.dtype)
        for block, context in zip(self.blocks, contexts, strict=True):
            features = block(features, context, self._rpe_bucket_ids)
        features = self.norm(features.float()).flatten(1)
        features = self.feature(features)
        norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
        return features / norms, norms

    def _keypoint_contexts(
        self,
        keypoints: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        centers = self._patch_centers.to(dtype=dtype)
        relative = (centers - keypoints.unsqueeze(1)).flatten(2)
        contexts = self.keypoint_linear(relative)
        contexts = contexts.view(
            len(keypoints),
            -1,
            _VIT_DEPTH,
            _VIT_HEADS,
            _VIT_RPE_BUCKETS,
        )
        return tuple(contexts[:, :, index].permute(0, 2, 1, 3) for index in range(_VIT_DEPTH))


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, _VIT_EMBED_DIM, kernel_size=8, stride=8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1:] != (3, 112, 112):
            raise ValueError(f"ViT-KP-RPE expects Bx3x112x112 input, got {tuple(inputs.shape)}")
        return self.proj(inputs).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(_VIT_EMBED_DIM)
        self.norm2 = nn.LayerNorm(_VIT_EMBED_DIM)
        self.attn = KPRPEAttention()
        self.mlp = MLP()

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        inputs = inputs + self.attn(self.norm1(inputs), context, bucket_ids)
        return inputs + self.mlp(self.norm2(inputs))


class KPRPEAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = _VIT_HEADS
        self.scale = (_VIT_EMBED_DIM // _VIT_HEADS) ** -0.5
        self.qkv = nn.Linear(_VIT_EMBED_DIM, _VIT_EMBED_DIM * 3, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(_VIT_EMBED_DIM, _VIT_EMBED_DIM)
        self.proj_drop = nn.Dropout(0.0)

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, dimensions = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.num_heads, -1)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        indices = bucket_ids.view(1, 1, tokens, tokens).expand(batch, self.num_heads, -1, -1)
        attention = attention + torch.gather(context, -1, indices)
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, dimensions)
        return self.proj_drop(self.proj(output))


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_VIT_EMBED_DIM, _VIT_EMBED_DIM * 3)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(_VIT_EMBED_DIM * 3, _VIT_EMBED_DIM)
        self.drop = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(inputs)))))


def _patch_centers() -> torch.Tensor:
    edges = torch.linspace(0, 1, _VIT_PATCHES_PER_SIDE + 1)
    coordinates = (edges[:-1] + edges[1:]) / 2
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.stack((columns, rows), dim=-1).reshape(1, -1, 1, 2)


def _product_bucket_ids() -> torch.Tensor:
    coordinates = torch.arange(_VIT_PATCHES_PER_SIDE)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    positions = torch.stack((rows, columns), dim=-1).reshape(-1, 2)
    differences = positions[:, None] - positions[None, :]
    indexed = _piecewise_index(differences)
    beta = 3
    side = beta * 2 + 1
    return ((indexed[..., 0] + beta) * side + indexed[..., 1] + beta).long()


def _piecewise_index(relative_positions: torch.Tensor) -> torch.Tensor:
    alpha = 1.9
    beta = 3.8
    gamma = 15.2
    absolute = relative_positions.abs()
    outside = absolute > alpha
    output = relative_positions.clone()
    values = relative_positions[outside]
    magnitudes = absolute[outside]
    mapped_magnitude = (
        (alpha + torch.log(magnitudes / alpha) / math.log(gamma / alpha) * (beta - alpha))
        .round()
        .clip(max=beta)
    )
    output[outside] = (torch.sign(values) * mapped_magnitude).to(output.dtype)
    return output
