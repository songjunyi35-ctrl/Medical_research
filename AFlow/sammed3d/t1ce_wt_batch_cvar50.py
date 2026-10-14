"""Per-draw DiceCE and frozen Batch-CVaR50 aggregation for T1ce/WT.

The module changes no model or voxel-level loss semantics.  It exposes the
historical ``Dice + 0.1 * BCE`` as eight independent scan-draw losses and then
either averages all draws or averages the four largest detached values.  The
selected original losses remain attached to the graph.
"""
from __future__ import annotations

from dataclasses import dataclass

from monai.losses import DiceLoss
import torch
import torch.nn.functional as F


SCHEMA = "aflow.sammed3d.t1ce_wt_batch_cvar50"
BATCH_CVAR50_ARMS = ("uniform_mean8", "batch_cvar50_top4")
BATCH_SIZE = 8
TOP_COUNT = 4
CE_WEIGHT = 0.1
DICE_SMOOTH_NUMERATOR = 1e-5
DICE_SMOOTH_DENOMINATOR = 1e-5


_PER_DRAW_DICE = DiceLoss(
    include_background=True,
    to_onehot_y=False,
    sigmoid=True,
    softmax=False,
    squared_pred=False,
    jaccard=False,
    smooth_nr=DICE_SMOOTH_NUMERATOR,
    smooth_dr=DICE_SMOOTH_DENOMINATOR,
    batch=False,
    weight=None,
    reduction="none",
)


@dataclass(frozen=True)
class PerDrawDiceCEOutput:
    """The historical loss decomposed into one scalar per scan draw."""

    per_draw_loss: torch.Tensor
    per_draw_dice: torch.Tensor
    per_draw_ce: torch.Tensor


@dataclass(frozen=True)
class BatchAggregationOutput:
    """One frozen aggregation and its auditable discrete selection."""

    total: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor
    selected_losses: torch.Tensor
    arm: str

    @property
    def loss(self) -> torch.Tensor:
        return self.total


@dataclass(frozen=True)
class BatchCVaR50Output:
    """Combined convenience output for a complete training loss call."""

    total: torch.Tensor
    per_draw_loss: torch.Tensor
    per_draw_dice: torch.Tensor
    per_draw_ce: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor
    selected_losses: torch.Tensor
    arm: str

    @property
    def loss(self) -> torch.Tensor:
        return self.total


def _validate_logits_target(logits: torch.Tensor, target: torch.Tensor) -> None:
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


def per_draw_dicece(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> PerDrawDiceCEOutput:
    """Return float32 historical DiceCE values without batch reduction."""

    _validate_logits_target(logits, target)
    # Preserve the caller's autocast context.  The historical DiceCELoss is
    # evaluated inside CUDA bfloat16 autocast; explicitly casting logits to
    # float32 here changes its scalar reduction even though gradients remain
    # almost identical.  Float32 outputs are enforced after the same kernels.
    dice_raw = _PER_DRAW_DICE(logits, target)
    per_draw_dice = dice_raw.reshape(logits.shape[0], -1).mean(dim=1)
    ce_raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        weight=None,
        pos_weight=None,
        reduction="none",
    )
    per_draw_dice = per_draw_dice.float()
    per_draw_ce = ce_raw.flatten(1).mean(dim=1).float()
    per_draw_loss = per_draw_dice + CE_WEIGHT * per_draw_ce
    values = (per_draw_loss, per_draw_dice, per_draw_ce)
    if any(value.shape != (logits.shape[0],) for value in values):
        raise RuntimeError("per-draw DiceCE reduction produced an unexpected shape")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("per-draw DiceCE produced a non-finite value")
    return PerDrawDiceCEOutput(
        per_draw_loss=per_draw_loss,
        per_draw_dice=per_draw_dice,
        per_draw_ce=per_draw_ce,
    )


def per_draw_dicece_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> PerDrawDiceCEOutput:
    """Compatibility alias used by the paired trainer."""

    return per_draw_dicece(logits, target)


def aggregate(
    losses: torch.Tensor,
    arm: str,
) -> BatchAggregationOutput:
    """Apply uniform mean8 or stable largest4-of8 aggregation."""

    if arm not in BATCH_CVAR50_ARMS:
        raise ValueError(f"unknown Batch-CVaR50 arm: {arm!r}")
    if not isinstance(losses, torch.Tensor):
        raise TypeError("losses must be a torch.Tensor")
    if losses.ndim != 1 or losses.shape[0] != BATCH_SIZE:
        raise ValueError("frozen Batch-CVaR50 aggregation requires exactly eight draws")
    if not losses.is_floating_point() or not bool(torch.isfinite(losses).all()):
        raise ValueError("losses must be finite floating-point values")

    if arm == "uniform_mean8":
        selected_indices = torch.arange(BATCH_SIZE, device=losses.device)
    else:
        selected_indices = torch.argsort(
            losses.detach(), descending=True, stable=True
        )[:TOP_COUNT]
    selected_losses = losses.index_select(0, selected_indices)
    selected_mask = torch.zeros(
        BATCH_SIZE, device=losses.device, dtype=torch.bool
    )
    selected_mask[selected_indices] = True
    total = selected_losses.mean()
    if total.ndim != 0 or not bool(torch.isfinite(total)):
        raise RuntimeError("Batch-CVaR50 aggregation produced a non-finite loss")
    return BatchAggregationOutput(
        total=total,
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        selected_losses=selected_losses,
        arm=arm,
    )


def aggregate_batch_case_loss(
    per_draw_loss: torch.Tensor,
    *,
    arm: str,
) -> BatchAggregationOutput:
    """Keyword-arm compatibility alias used by orchestration code."""

    return aggregate(per_draw_loss, arm)


def batch_cvar50_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    arm: str,
) -> BatchCVaR50Output:
    """Compute per-draw DiceCE and one frozen batch aggregation."""

    components = per_draw_dicece(logits, target)
    reduced = aggregate(components.per_draw_loss, arm)
    return BatchCVaR50Output(
        total=reduced.total,
        per_draw_loss=components.per_draw_loss,
        per_draw_dice=components.per_draw_dice,
        per_draw_ce=components.per_draw_ce,
        selected_indices=reduced.selected_indices,
        selected_mask=reduced.selected_mask,
        selected_losses=reduced.selected_losses,
        arm=arm,
    )


__all__ = [
    "BATCH_CVAR50_ARMS",
    "BATCH_SIZE",
    "BatchAggregationOutput",
    "BatchCVaR50Output",
    "CE_WEIGHT",
    "DICE_SMOOTH_DENOMINATOR",
    "DICE_SMOOTH_NUMERATOR",
    "PerDrawDiceCEOutput",
    "SCHEMA",
    "TOP_COUNT",
    "aggregate",
    "aggregate_batch_case_loss",
    "batch_cvar50_loss",
    "per_draw_dicece",
    "per_draw_dicece_loss",
]
