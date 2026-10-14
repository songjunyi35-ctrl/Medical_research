"""Frozen binary 3D Voxel-wise Modulated Dice loss for T1ce/WT.

The model emits one foreground raw logit.  This loss constructs explicit
background and foreground probability/target channels, evaluates the paper's
squared-denominator Dice over ground-truth-present classes in each volume, and
adds the frozen ``0.1 * BCEWithLogits`` term.  The dynamic modulation is always
detached; the ordinary probability path in the Dice numerator and denominator
remains differentiable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


SCHEMA = "aflow.sammed3d.t1ce_wt_vm_dice"
VM_DICE_ARM_GAMMAS = {
    "paper_dice_gamma0": 0.0,
    "vm_dice_gamma1": 1.0,
}
ARM_GAMMAS = VM_DICE_ARM_GAMMAS
CLASS_NAMES = ("background", "foreground")
VM_DICE_EPSILON = 1e-6
EPSILON = VM_DICE_EPSILON
BCE_WEIGHT = 0.1


@dataclass(frozen=True)
class VMDiceOutput:
    """Total objective and its complete small-tensor decomposition."""

    total: torch.Tensor
    loss: torch.Tensor
    bce: torch.Tensor
    per_case_loss: torch.Tensor
    per_volume_score: torch.Tensor
    class_scores: torch.Tensor
    modulation: torch.Tensor
    present_classes: torch.Tensor
    present_class_counts: torch.Tensor
    gamma: float

    @property
    def vm_dice(self) -> torch.Tensor:
        """Backward-compatible explicit name for the pure VM Dice loss."""

        return self.loss


def _validate_raw_binary_inputs(logits: torch.Tensor, target: torch.Tensor) -> None:
    for name, value in (("logits", logits), ("target", target)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 5 or value.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,D,H,W]")
        if value.shape[0] <= 0 or any(size <= 0 for size in value.shape[2:]):
            raise ValueError(f"{name} dimensions must be positive")
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shapes")
    if logits.device != target.device:
        raise ValueError("logits and target must be on the same device")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if target.is_floating_point() and not bool(torch.isfinite(target).all()):
        raise ValueError("target must be finite")
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("target must be binary")


def _validate_gamma(gamma: float) -> float:
    value = float(gamma)
    if value not in tuple(VM_DICE_ARM_GAMMAS.values()):
        raise ValueError("frozen VM Dice protocol supports only gamma=0 or gamma=1")
    return value


def binary_probability_and_target_channels(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 channels ordered as background, foreground."""

    _validate_raw_binary_inputs(logits, target)
    foreground_probability = torch.sigmoid(logits.float())
    foreground_target = target.float()
    probabilities = torch.cat(
        (1.0 - foreground_probability, foreground_probability), dim=1
    )
    targets = torch.cat((1.0 - foreground_target, foreground_target), dim=1)
    return probabilities, targets


def voxel_modulation(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Return the frozen detached modulation for either paired arm."""

    value = _validate_gamma(gamma)
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have identical shapes")
    if probabilities.ndim != 5 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("probabilities must have shape [B,2,D,H,W]")
    if value == 0.0:
        modulation = torch.ones_like(probabilities)
    else:
        modulation = torch.abs(targets - probabilities.detach())
    if modulation.requires_grad:
        raise RuntimeError("VM Dice modulation must be detached")
    return modulation


def vm_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float,
) -> VMDiceOutput:
    """Compute the preregistered gamma0/gamma1 VM Dice plus BCE objective."""

    value = _validate_gamma(gamma)
    probabilities, targets = binary_probability_and_target_channels(logits, target)
    modulation = voxel_modulation(probabilities, targets, gamma=value)
    spatial_dims = tuple(range(2, probabilities.ndim))
    numerator = 2.0 * (modulation * targets * probabilities).sum(dim=spatial_dims) + EPSILON
    denominator = (
        modulation * (targets.square() + probabilities.square())
    ).sum(dim=spatial_dims) + EPSILON
    class_scores = numerator / denominator
    present_classes = targets.sum(dim=spatial_dims) > 0
    present_class_counts = present_classes.sum(dim=1)
    if not bool((present_class_counts > 0).all()):
        raise RuntimeError("each binary volume must contain at least one present class")
    per_volume_score = (
        class_scores * present_classes.to(class_scores.dtype)
    ).sum(dim=1) / present_class_counts.to(class_scores.dtype)
    per_case_loss = 1.0 - per_volume_score
    vm_dice = per_case_loss.mean()
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="mean"
    )
    total = vm_dice + BCE_WEIGHT * bce
    scalars = (vm_dice, bce, total)
    if any(item.ndim != 0 or not bool(torch.isfinite(item)) for item in scalars):
        raise RuntimeError("VM Dice produced a non-finite or non-scalar loss")
    if not 0.0 <= float(vm_dice.detach()) <= 1.0:
        raise RuntimeError("VM Dice component must lie in [0,1]")
    return VMDiceOutput(
        total=total,
        loss=vm_dice,
        bce=bce,
        per_case_loss=per_case_loss,
        per_volume_score=per_volume_score,
        class_scores=class_scores,
        modulation=modulation,
        present_classes=present_classes,
        present_class_counts=present_class_counts,
        gamma=value,
    )


def vm_dice_loss_for_arm(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    arm: str,
) -> VMDiceOutput:
    """Resolve one of the two frozen paired-arm names."""

    if arm not in VM_DICE_ARM_GAMMAS:
        raise ValueError(f"unknown VM Dice arm: {arm!r}")
    return vm_dice_loss(logits, target, gamma=VM_DICE_ARM_GAMMAS[arm])


def voxel_modulated_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
) -> VMDiceOutput:
    """Public API shared by the paired trainer and validation evaluator.

    ``output.loss`` is the pure VM Dice component.  The frozen compound
    objective is available as ``output.total``; the formal trainer may also
    form ``output.loss + 0.1 * BCE`` explicitly.
    """

    return vm_dice_loss(logits, target, gamma=gamma)


__all__ = [
    "ARM_GAMMAS",
    "BCE_WEIGHT",
    "CLASS_NAMES",
    "EPSILON",
    "SCHEMA",
    "VM_DICE_ARM_GAMMAS",
    "VM_DICE_EPSILON",
    "VMDiceOutput",
    "binary_probability_and_target_channels",
    "vm_dice_loss",
    "vm_dice_loss_for_arm",
    "voxel_modulated_dice_loss",
    "voxel_modulation",
]
