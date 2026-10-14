"""Error-localized prompt-gated 3D residual refinement.

The module upgrades the existing two-input residual SegResNet without changing
its prediction at initialization.  Frozen SAM logits are treated as a dense
prompt: their probability, uncertainty, and local 3D boundary are exposed to
the network.  A learned gate can suppress or amplify the residual correction.

Ground truth is never an inference input.  It is used only to supervise the
correction gate during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.nets import SegResNet


FIRST_WEIGHT = "network.convInit.conv.weight"
FINAL_WEIGHT = "network.conv_final.2.conv.weight"
FINAL_BIAS = "network.conv_final.2.conv.bias"


@dataclass(frozen=True)
class ErrorGatedOutput:
    logits: torch.Tensor
    residual: torch.Tensor
    gate_logits: torch.Tensor
    residual_scale: torch.Tensor


def dense_prompt_features(coarse_logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return label-free dense-prompt features at the native resolution."""
    if coarse_logits.ndim != 5 or coarse_logits.shape[1] != 1:
        raise ValueError("coarse_logits must have shape [B, 1, D, H, W]")
    probability = torch.sigmoid(coarse_logits)
    uncertainty = 4.0 * probability * (1.0 - probability)
    local_max = F.max_pool3d(probability, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool3d(-probability, kernel_size=3, stride=1, padding=1)
    boundary = local_max - local_min
    return torch.tanh(coarse_logits / 5.0), uncertainty, boundary


def correction_target(
    coarse_logits: torch.Tensor,
    label: torch.Tensor,
    *,
    dilation_radius: int = 2,
) -> torch.Tensor:
    """Dilated coarse-error map used only as an auxiliary training target."""
    if coarse_logits.shape != label.shape:
        raise ValueError("coarse_logits and label must have identical shapes")
    if dilation_radius < 0:
        raise ValueError("dilation_radius must be non-negative")
    error = torch.logical_xor(coarse_logits > 0, label > 0.5).to(label.dtype)
    if dilation_radius:
        kernel = 2 * dilation_radius + 1
        error = F.max_pool3d(error, kernel_size=kernel, stride=1, padding=dilation_radius)
    return error


class ErrorGatedResidualRefiner(nn.Module):
    """SegResNet residual refiner with uncertainty/boundary prompts and a gate."""

    def __init__(self, init_filters: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=2,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=dropout,
        )
        final = self.network.conv_final[2].conv
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward_with_aux(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> ErrorGatedOutput:
        if image.shape != coarse_logits.shape:
            raise ValueError("image and coarse_logits must have identical shapes")
        coarse_feature, uncertainty, boundary = dense_prompt_features(coarse_logits)
        heads = self.network(
            torch.cat([image, coarse_feature, uncertainty, boundary], dim=1)
        )
        residual, gate_logits = heads.split(1, dim=1)
        residual_scale = 1.0 + torch.tanh(gate_logits)
        logits = coarse_logits + residual_scale * residual
        return ErrorGatedOutput(
            logits=logits,
            residual=residual,
            gate_logits=gate_logits,
            residual_scale=residual_scale,
        )

    def forward(self, image: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(image, coarse_logits).logits


def bridge_residual_refiner_state_dict(
    model: ErrorGatedResidualRefiner,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    """Load a legacy 2-input/1-output refiner with exact functional parity.

    The old image/coarse kernels and residual head are copied.  New prompt
    channels and the gate head are zeroed, making residual_scale exactly one.
    """
    target = model.state_dict()
    missing = sorted(set(target) - set(source_state))
    unexpected = sorted(set(source_state) - set(target))
    allowed_mismatch = {FIRST_WEIGHT, FINAL_WEIGHT, FINAL_BIAS}
    if set(missing) - allowed_mismatch or set(unexpected) - allowed_mismatch:
        raise ValueError(
            f"incompatible legacy state: missing={missing}, unexpected={unexpected}"
        )
    for name, value in source_state.items():
        if name in allowed_mismatch:
            continue
        if name not in target or target[name].shape != value.shape:
            raise ValueError(f"incompatible tensor {name}: {value.shape}")
        target[name].copy_(value)

    old_first = source_state[FIRST_WEIGHT]
    new_first = target[FIRST_WEIGHT]
    if old_first.shape[1] != 2 or new_first.shape[1] != 4:
        raise ValueError("unexpected first-convolution channel count")
    if old_first.shape[0] != new_first.shape[0] or old_first.shape[2:] != new_first.shape[2:]:
        raise ValueError("unexpected first-convolution shape")
    new_first.zero_()
    new_first[:, :2].copy_(old_first)

    old_final = source_state[FINAL_WEIGHT]
    new_final = target[FINAL_WEIGHT]
    if old_final.shape[0] != 1 or new_final.shape[0] != 2 or old_final.shape[1:] != new_final.shape[1:]:
        raise ValueError("unexpected final-convolution shape")
    new_final.zero_()
    new_final[0].copy_(old_final[0])
    old_bias = source_state.get(FINAL_BIAS)
    new_bias = target.get(FINAL_BIAS)
    if old_bias is not None and new_bias is not None:
        new_bias.zero_()
        new_bias[0].copy_(old_bias[0])
    elif old_bias is not None or new_bias is not None:
        raise ValueError("source/target final bias mismatch")

    model.load_state_dict(target, strict=True)
    return {
        "source_input_channels": 2,
        "target_input_channels": 4,
        "source_output_channels": 1,
        "target_output_channels": 2,
        "copied_legacy_tensors": len(source_state),
        "new_input_channels_zero_initialized": True,
        "gate_head_zero_initialized": True,
        "initial_residual_scale": 1.0,
    }
