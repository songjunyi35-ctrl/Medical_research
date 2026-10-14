"""Three-dimensional boundary targets and metrics for T1ce/WT refinement."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def boundary_target(label: torch.Tensor, *, radius: int = 1) -> torch.Tensor:
    """Return a symmetric morphological boundary band around a binary target."""
    if label.ndim != 5 or label.shape[1] != 1:
        raise ValueError("label must have shape [B, 1, D, H, W]")
    if radius < 1:
        raise ValueError("radius must be at least one")
    target = (label > 0.5).to(label.dtype)
    kernel = 2 * radius + 1
    dilated = F.max_pool3d(target, kernel_size=kernel, stride=1, padding=radius)
    eroded = -F.max_pool3d(-target, kernel_size=kernel, stride=1, padding=radius)
    return (dilated > eroded).to(label.dtype)


def boundary_dice_per_case(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    prediction = logits > 0
    truth = target > 0.5
    intersection = (prediction & truth).flatten(1).sum(1).float()
    denominator = prediction.flatten(1).sum(1) + truth.flatten(1).sum(1)
    return torch.where(
        denominator > 0,
        2.0 * intersection / denominator.float(),
        torch.ones_like(intersection),
    )
