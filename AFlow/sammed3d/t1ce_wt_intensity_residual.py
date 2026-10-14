"""Intensity-aware uncertainty-gated residual correction for the plain locator."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator


SCHEMA = "aflow.sammed3d.t1ce_wt_intensity_residual"


@dataclass(frozen=True)
class IntensityResidualOutput:
    segmentation_logits: torch.Tensor
    base_logits: torch.Tensor
    residual_logits: torch.Tensor
    uncertainty: torch.Tensor
    explicit_intensity_features: torch.Tensor


def intensity_features(image: torch.Tensor, *, kernel_size: int = 7) -> torch.Tensor:
    """Return clipped intensity, local z-contrast, and gradient magnitude."""

    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError("image must have shape [B,1,D,H,W]")
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than one")
    padding = kernel_size // 2
    local_mean = F.avg_pool3d(image, kernel_size, stride=1, padding=padding)
    local_square_mean = F.avg_pool3d(
        image.square(), kernel_size, stride=1, padding=padding
    )
    local_std = (local_square_mean - local_mean.square()).clamp_min(1e-4).sqrt()
    local_z = ((image - local_mean) / local_std).clamp(-5.0, 5.0) / 5.0

    dz = F.pad(image[:, :, 1:] - image[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(image[:, :, :, :, 1:] - image[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    gradient = (dx.square() + dy.square() + dz.square() + 1e-6).sqrt().tanh()
    clipped = image.clamp(-5.0, 5.0) / 5.0
    result = torch.cat((clipped, local_z, gradient), dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("intensity features contain non-finite values")
    return result


class IntensityUncertaintyResidualLocator(nn.Module):
    """Plain locator plus a zero-start residual head on uncertain voxels.

    ``use_intensity=False`` is the matched control: the architecture and parameter
    initialization are unchanged, but all three explicit intensity channels are
    exactly zero.
    """

    def __init__(
        self,
        *,
        use_intensity: bool,
        init_filters: int = 16,
        dropout: float = 0.1,
        residual_channels: int = 8,
        intensity_kernel: int = 7,
    ) -> None:
        super().__init__()
        if residual_channels <= 0:
            raise ValueError("residual_channels must be positive")
        self.use_intensity = bool(use_intensity)
        self.init_filters = int(init_filters)
        self.dropout = float(dropout)
        self.residual_channels = int(residual_channels)
        self.intensity_kernel = int(intensity_kernel)
        self.base = AutomaticPromptLocator(
            center_aware=False,
            init_filters=self.init_filters,
            dropout=self.dropout,
        )
        self.residual_head = nn.Sequential(
            nn.Conv3d(6, self.residual_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, self.residual_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                self.residual_channels,
                self.residual_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, self.residual_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.residual_channels, 1, kernel_size=1),
        )
        final = self.residual_head[-1]
        assert isinstance(final, nn.Conv3d)
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward(self, image: torch.Tensor) -> IntensityResidualOutput:
        base_logits = self.base(image).segmentation_logits
        detached_logits = base_logits.detach()
        probability = detached_logits.sigmoid()
        uncertainty = (4.0 * probability * (1.0 - probability)).detach()
        explicit = intensity_features(image, kernel_size=self.intensity_kernel)
        if not self.use_intensity:
            explicit = torch.zeros_like(explicit)
        context = torch.cat(
            (
                (detached_logits / 5.0).tanh(),
                probability,
                uncertainty,
                explicit,
            ),
            dim=1,
        )
        residual_logits = self.residual_head(context)
        segmentation_logits = base_logits + uncertainty * residual_logits
        return IntensityResidualOutput(
            segmentation_logits=segmentation_logits,
            base_logits=base_logits,
            residual_logits=residual_logits,
            uncertainty=uncertainty,
            explicit_intensity_features=explicit,
        )

