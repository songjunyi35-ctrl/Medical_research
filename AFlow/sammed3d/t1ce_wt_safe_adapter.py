"""Zero-initialized safe adapters on a frozen robust T1ce/WT refiner."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from monai.networks.nets import SegResNet


@dataclass(frozen=True)
class SafeAdapterOutput:
    logits: torch.Tensor
    base_logits: torch.Tensor
    residual: torch.Tensor
    gate_logits: torch.Tensor
    residual_scale: torch.Tensor


class FrozenBaseSafeAdapter(nn.Module):
    """Add a small uncertainty-conditioned adapter without rewriting the base."""

    MODES = frozenset({"plain", "gated"})

    def __init__(
        self,
        base: nn.Module,
        *,
        mode: str,
        init_filters: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.mode = mode
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.adapter = SegResNet(
            spatial_dims=3,
            in_channels=5,
            out_channels=2,
            init_filters=init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=dropout,
        )
        final = self.adapter.conv_final[2].conv
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def train(self, mode: bool = True) -> "FrozenBaseSafeAdapter":
        super().train(mode)
        self.base.eval()
        return self

    def forward_with_aux(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> SafeAdapterOutput:
        if image.shape != coarse_logits.shape:
            raise ValueError("image and coarse_logits must have identical shapes")
        with torch.no_grad():
            base_logits = self.base(image, coarse_logits)
        base_probability = torch.sigmoid(base_logits)
        coarse_probability = torch.sigmoid(coarse_logits)
        base_uncertainty = 4.0 * base_probability * (1.0 - base_probability)
        disagreement = torch.abs(base_probability - coarse_probability)
        features = torch.cat(
            [
                image,
                torch.tanh(coarse_logits / 5.0),
                torch.tanh(base_logits / 5.0),
                base_uncertainty,
                disagreement,
            ],
            dim=1,
        )
        residual, gate_logits = self.adapter(features).split(1, dim=1)
        if self.mode == "plain":
            residual_scale = torch.ones_like(gate_logits)
        else:
            residual_scale = 2.0 * torch.sigmoid(gate_logits)
        logits = base_logits + residual_scale * residual
        return SafeAdapterOutput(
            logits=logits,
            base_logits=base_logits,
            residual=residual,
            gate_logits=gate_logits,
            residual_scale=residual_scale,
        )

    def forward(self, image: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(image, coarse_logits).logits

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
