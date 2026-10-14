"""Foreground hard-binned mL1-ACE for the binary T1ce/WT locator.

The frozen project adaptation uses one sigmoid foreground channel and twenty
hard probability bins.  Binning is performed independently for every volume.
The integer assignment is intentionally detached, while the probability sums
remain attached to the graph so the occupied-bin calibration gaps can update
the raw segmentation logits.

This module never constructs a ``[batch, bins, *volume]`` membership tensor.
Only one flattened volume, its integer bin index, and twenty-bin statistics are
live inside the streaming loop.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


SCHEMA = "aflow.sammed3d.t1ce_wt_foreground_hard_ml1ace"
NUM_BINS = 20
BUCKETIZE_RIGHT = False
ML1ACE_WEIGHT = 1.0


@dataclass(frozen=True)
class HardML1ACEBatchStatistics:
    """Auditable per-volume foreground calibration statistics.

    ``loss`` and ``per_case_ace`` follow the frozen training rule: a volume
    with empty foreground ground truth contributes zero.  Bin sums and counts
    are nevertheless retained, making it possible for evaluation code to
    aggregate the exact observed probability distribution.
    """

    loss: torch.Tensor
    per_case_ace: torch.Tensor
    per_case_ece: torch.Tensor
    per_case_mce: torch.Tensor
    counts: torch.Tensor
    probability_sums: torch.Tensor
    target_sums: torch.Tensor
    mean_probabilities: torch.Tensor
    mean_targets: torch.Tensor
    occupied: torch.Tensor
    target_nonempty: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.counts.shape[0])


@dataclass(frozen=True)
class HardML1ACEAggregate:
    """Macro, official-micro, and truly pooled foreground ACE summaries."""

    case_count: int
    macro_ace: torch.Tensor
    macro_ece: torch.Tensor
    macro_mce: torch.Tensor
    official_micro_ace: torch.Tensor
    pooled_voxel_ace: torch.Tensor
    bin_counts: torch.Tensor
    occupied_case_counts: torch.Tensor
    official_micro_mean_probabilities: torch.Tensor
    official_micro_mean_targets: torch.Tensor
    pooled_mean_probabilities: torch.Tensor
    pooled_mean_targets: torch.Tensor
    occupied: torch.Tensor


def _validate_binary_volume_tensor(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,D,H,W]")
    if value.shape[0] <= 0 or any(size <= 0 for size in value.shape[2:]):
        raise ValueError(f"{name} dimensions must be positive")


def _validate_probability_and_target(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _validate_binary_volume_tensor(probability, name="probability")
    _validate_binary_volume_tensor(target, name="target")
    if probability.shape != target.shape:
        raise ValueError("probability and target must have identical shapes")
    if probability.device != target.device:
        raise ValueError("probability and target must be on the same device")
    if not probability.is_floating_point():
        raise TypeError("probability must be floating point")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("probability must be finite")
    if not bool(((probability >= 0.0) & (probability <= 1.0)).all()):
        raise ValueError("probability values must lie in [0,1]")
    if target.is_floating_point() and not bool(torch.isfinite(target).all()):
        raise ValueError("target must be finite")
    if not bool(((target == 0) | (target == 1)).all()):
        raise ValueError("target must be binary")


def hard_ml1ace_from_probabilities(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> HardML1ACEBatchStatistics:
    """Compute the frozen per-volume foreground hard mL1-ACE statistics.

    The official boundary construction and ``torch.bucketize`` call are kept
    literal.  In particular, ``right=False`` assigns an exact internal
    boundary to the lower bin; replacing this operation with ``floor`` is not
    equivalent.
    """

    _validate_probability_and_target(probability, target)
    probability_float = probability.float()
    target_float = target.float()
    boundaries = torch.linspace(
        0.0,
        1.0 + torch.finfo(torch.float32).eps,
        NUM_BINS + 1,
        device=probability.device,
        dtype=torch.float32,
    )

    case_aces: list[torch.Tensor] = []
    case_eces: list[torch.Tensor] = []
    case_mces: list[torch.Tensor] = []
    all_counts: list[torch.Tensor] = []
    all_probability_sums: list[torch.Tensor] = []
    all_target_sums: list[torch.Tensor] = []
    all_mean_probabilities: list[torch.Tensor] = []
    all_mean_targets: list[torch.Tensor] = []
    all_occupied: list[torch.Tensor] = []
    all_target_nonempty: list[torch.Tensor] = []

    for case_index in range(probability.shape[0]):
        case_probability = probability_float[case_index, 0].reshape(-1)
        case_target = target_float[case_index, 0].reshape(-1)
        # Assignment is the sole non-differentiable part.  Probability values
        # below remain attached in scatter_add and therefore in the bin means.
        bin_index = torch.bucketize(
            case_probability.detach(),
            boundaries[1:],
            right=BUCKETIZE_RIGHT,
        )
        counts = torch.bincount(bin_index, minlength=NUM_BINS)
        probability_sums = torch.zeros(
            NUM_BINS, device=probability.device, dtype=torch.float32
        ).scatter_add(0, bin_index, case_probability)
        target_sums = torch.zeros_like(probability_sums).scatter_add(
            0, bin_index, case_target
        )
        occupied = counts > 0
        denominator = counts.clamp_min(1).to(torch.float32)
        mean_probability = probability_sums / denominator
        mean_target = target_sums / denominator
        gaps = torch.abs(mean_probability - mean_target)

        raw_ace = gaps[occupied].mean()
        raw_ece = (
            gaps[occupied]
            * counts[occupied].to(torch.float32)
            / counts.sum().to(torch.float32)
        ).sum()
        raw_mce = gaps[occupied].max()
        target_nonempty = case_target.sum() > 0
        zero = raw_ace.new_zeros(())
        case_aces.append(torch.where(target_nonempty, raw_ace, zero))
        case_eces.append(torch.where(target_nonempty, raw_ece, zero))
        case_mces.append(torch.where(target_nonempty, raw_mce, zero))
        all_counts.append(counts)
        all_probability_sums.append(probability_sums)
        all_target_sums.append(target_sums)
        all_mean_probabilities.append(mean_probability)
        all_mean_targets.append(mean_target)
        all_occupied.append(occupied)
        all_target_nonempty.append(target_nonempty)

    per_case_ace = torch.stack(case_aces)
    result = HardML1ACEBatchStatistics(
        loss=per_case_ace.mean(),
        per_case_ace=per_case_ace,
        per_case_ece=torch.stack(case_eces),
        per_case_mce=torch.stack(case_mces),
        counts=torch.stack(all_counts),
        probability_sums=torch.stack(all_probability_sums),
        target_sums=torch.stack(all_target_sums),
        mean_probabilities=torch.stack(all_mean_probabilities),
        mean_targets=torch.stack(all_mean_targets),
        occupied=torch.stack(all_occupied),
        target_nonempty=torch.stack(all_target_nonempty),
    )
    scalar_values = (result.loss, *result.per_case_ace.unbind())
    if any(value.ndim != 0 or not bool(torch.isfinite(value)) for value in scalar_values):
        raise RuntimeError("hard mL1-ACE produced a non-finite or non-scalar loss")
    if not 0.0 <= float(result.loss.detach()) <= 1.0:
        raise RuntimeError("hard mL1-ACE must lie in [0,1]")
    return result


def foreground_hard_ml1ace(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> HardML1ACEBatchStatistics:
    """Apply float32 sigmoid to raw logits and compute hard mL1-ACE."""

    _validate_binary_volume_tensor(logits, name="logits")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    probability = torch.sigmoid(logits.float())
    return hard_ml1ace_from_probabilities(probability, target)


def _as_statistics_sequence(
    statistics: HardML1ACEBatchStatistics
    | Sequence[HardML1ACEBatchStatistics],
) -> tuple[HardML1ACEBatchStatistics, ...]:
    if isinstance(statistics, HardML1ACEBatchStatistics):
        return (statistics,)
    values = tuple(statistics)
    if not values or any(not isinstance(value, HardML1ACEBatchStatistics) for value in values):
        raise TypeError("statistics must contain HardML1ACEBatchStatistics values")
    return values


def aggregate_foreground_calibration(
    statistics: HardML1ACEBatchStatistics
    | Sequence[HardML1ACEBatchStatistics],
) -> HardML1ACEAggregate:
    """Aggregate frozen macro, official-micro, and pooled-voxel ACE.

    The preregistered ``official_micro_ace`` intentionally follows the source
    repository: per-bin prediction means are averaged equally across scans,
    whereas per-bin ground-truth means are weighted by each scan's bin count.
    ``pooled_voxel_ace`` instead weights both quantities by voxel count.
    """

    values = _as_statistics_sequence(statistics)
    devices = {value.counts.device for value in values}
    if len(devices) != 1:
        raise ValueError("all statistics must be on the same device")
    if any(value.counts.shape[1:] != (NUM_BINS,) for value in values):
        raise ValueError("all statistics must use the frozen twenty-bin layout")

    per_case_ace = torch.cat([value.per_case_ace for value in values], dim=0)
    per_case_ece = torch.cat([value.per_case_ece for value in values], dim=0)
    per_case_mce = torch.cat([value.per_case_mce for value in values], dim=0)
    counts = torch.cat([value.counts for value in values], dim=0)
    probability_sums = torch.cat(
        [value.probability_sums for value in values], dim=0
    )
    target_sums = torch.cat([value.target_sums for value in values], dim=0)
    mean_probabilities = torch.cat(
        [value.mean_probabilities for value in values], dim=0
    )
    occupied_by_case = counts > 0

    occupied_case_counts = occupied_by_case.sum(dim=0)
    bin_counts = counts.sum(dim=0)
    occupied = bin_counts > 0
    occupied_case_denominator = occupied_case_counts.clamp_min(1).to(torch.float32)
    official_mean_probability = (
        mean_probabilities * occupied_by_case.to(torch.float32)
    ).sum(dim=0) / occupied_case_denominator
    official_mean_target = target_sums.sum(dim=0) / bin_counts.clamp_min(1).to(
        torch.float32
    )
    pooled_mean_probability = probability_sums.sum(dim=0) / bin_counts.clamp_min(
        1
    ).to(torch.float32)
    pooled_mean_target = official_mean_target

    official_micro_ace = torch.abs(
        official_mean_probability[occupied] - official_mean_target[occupied]
    ).mean()
    pooled_voxel_ace = torch.abs(
        pooled_mean_probability[occupied] - pooled_mean_target[occupied]
    ).mean()
    return HardML1ACEAggregate(
        case_count=int(per_case_ace.numel()),
        macro_ace=per_case_ace.mean(),
        macro_ece=per_case_ece.mean(),
        macro_mce=per_case_mce.mean(),
        official_micro_ace=official_micro_ace,
        pooled_voxel_ace=pooled_voxel_ace,
        bin_counts=bin_counts,
        occupied_case_counts=occupied_case_counts,
        official_micro_mean_probabilities=official_mean_probability,
        official_micro_mean_targets=official_mean_target,
        pooled_mean_probabilities=pooled_mean_probability,
        pooled_mean_targets=pooled_mean_target,
        occupied=occupied,
    )


__all__ = [
    "BUCKETIZE_RIGHT",
    "HardML1ACEAggregate",
    "HardML1ACEBatchStatistics",
    "ML1ACE_WEIGHT",
    "NUM_BINS",
    "SCHEMA",
    "aggregate_foreground_calibration",
    "foreground_hard_ml1ace",
    "hard_ml1ace_from_probabilities",
]
