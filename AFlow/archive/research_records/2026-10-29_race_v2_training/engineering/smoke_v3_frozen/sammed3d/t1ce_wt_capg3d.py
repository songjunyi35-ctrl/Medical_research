"""Image-only 3D prompt localization for automatic SAM-Med3D inference.

CAPG-3D adds a centre-heatmap head to a compact 3D segmentation locator.  At
inference neither arm sees a label: the plain arm takes the maximum of its
coarse WT logit, while CAPG-3D takes the maximum of the learned centre map.
The selected coordinate and the coarse mask logits can then be used as SAM's
point and dense-mask prompts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from monai.networks.nets import SegResNet


SCHEMA = "aflow.sammed3d.t1ce_wt_capg3d"


@dataclass(frozen=True)
class LocatorOutput:
    segmentation_logits: torch.Tensor
    center_logits: torch.Tensor | None


class AutomaticPromptLocator(nn.Module):
    """Matched plain/CAPG locator with a shared SegResNet backbone."""

    def __init__(
        self,
        *,
        center_aware: bool,
        init_filters: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.center_aware = bool(center_aware)
        self.init_filters = int(init_filters)
        self.dropout = float(dropout)
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2 if self.center_aware else 1,
            init_filters=self.init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=self.dropout,
        )
        # Keep the paired segmentation heads exactly aligned at initialization.
        # A sparse prior is appropriate for the auxiliary Gaussian centre map.
        final = self.network.conv_final[2].conv
        if final.bias is not None:
            with torch.no_grad():
                final.bias[0].zero_()
                if self.center_aware:
                    final.bias[1].fill_(-4.0)

    def forward(self, image: torch.Tensor) -> LocatorOutput:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image must have shape [B,1,D,H,W]")
        logits = self.network(image)
        if self.center_aware:
            return LocatorOutput(logits[:, :1], logits[:, 1:2])
        return LocatorOutput(logits, None)


def foreground_anchor(label: torch.Tensor) -> torch.Tensor:
    """Return a centroid-nearest foreground voxel for each binary mask.

    A raw centre of mass can lie outside a disconnected or crescent-shaped
    tumour.  Projecting it back onto foreground guarantees that the training
    target represents a valid positive click.
    """

    if label.ndim != 5 or label.shape[1] != 1:
        raise ValueError("label must have shape [B,1,D,H,W]")
    anchors: list[torch.Tensor] = []
    for batch_index in range(label.shape[0]):
        coordinates = torch.nonzero(label[batch_index, 0] > 0.5, as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("center target requires a non-empty foreground mask")
        centroid = coordinates.float().mean(dim=0)
        distance = (coordinates.float() - centroid).square().sum(dim=1)
        anchors.append(coordinates[torch.argmin(distance)])
    return torch.stack(anchors, dim=0)


def gaussian_center_target(label: torch.Tensor, *, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a Gaussian heatmap around a valid positive-click anchor."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    anchors = foreground_anchor(label)
    depth, height, width = label.shape[-3:]
    z = torch.arange(depth, device=label.device, dtype=torch.float32).view(1, depth, 1, 1)
    y = torch.arange(height, device=label.device, dtype=torch.float32).view(1, 1, height, 1)
    x = torch.arange(width, device=label.device, dtype=torch.float32).view(1, 1, 1, width)
    anchor = anchors.float()
    squared_distance = (
        (z - anchor[:, 0, None, None, None]).square()
        + (y - anchor[:, 1, None, None, None]).square()
        + (x - anchor[:, 2, None, None, None]).square()
    )
    target = torch.exp(-squared_distance / (2.0 * float(sigma) ** 2)).unsqueeze(1)
    return target, anchors


def center_heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft Dice plus a small MSE term for a sparse Gaussian heatmap."""

    if logits.shape != target.shape:
        raise ValueError("center logits and target shapes differ")
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    soft_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    mse = (probability - target).square().flatten(1).mean(dim=1)
    return (1.0 - soft_dice + 0.1 * mse).mean()


def automatic_prompt_points(output: LocatorOutput) -> torch.Tensor:
    """Select one label-free positive point in model-grid coordinates."""

    source = output.center_logits if output.center_logits is not None else output.segmentation_logits
    if source.ndim != 5 or source.shape[1] != 1:
        raise ValueError("prompt source must have shape [B,1,D,H,W]")
    flattened = source[:, 0].flatten(1)
    flat_index = torch.argmax(flattened, dim=1)
    height, width = source.shape[-2:]
    z = torch.div(flat_index, height * width, rounding_mode="floor")
    remainder = flat_index % (height * width)
    y = torch.div(remainder, width, rounding_mode="floor")
    x = remainder % width
    return torch.stack((z, y, x), dim=1).to(torch.float32).unsqueeze(1)
