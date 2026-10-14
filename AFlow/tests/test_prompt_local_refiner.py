from __future__ import annotations

import numpy as np
import torch

from scripts.train_t1ce_wt_prompt_local_refiner import (
    DistributedWeightedSampler,
    combined_patch_dice,
    gaussian_heatmap,
    patch_origin,
)


def test_patch_origin_centres_and_clamps_fixed_cube() -> None:
    assert patch_origin(torch.tensor([64, 64, 64]), 64) == (32, 32, 32)
    assert patch_origin(torch.tensor([2, 126, 50]), 64) == (0, 64, 18)


def test_click_heatmap_peaks_at_local_click() -> None:
    heatmap = gaussian_heatmap(torch.tensor([40, 50, 60]), 1, (32, 32, 32), 64, 3.0)
    assert tuple(torch.nonzero(heatmap == heatmap.max())[0].tolist()) == (8, 18, 28)
    assert float(heatmap.max()) == 1.0
    assert not bool(gaussian_heatmap(torch.tensor([-1, -1, -1]), -1, (0, 0, 0), 64, 3.0).any())


def test_combined_patch_dice_counts_outside_voxels() -> None:
    value = combined_patch_dice(
        np.array([3]), np.array([5]), np.array([4]),
        np.array([7]), np.array([8]), np.array([10]),
    )
    assert np.allclose(value, np.array([20 / 27]))


def test_distributed_weighted_sampler_is_deterministic_and_disjoint_by_position() -> None:
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    rank0 = DistributedWeightedSampler(weights, replicas=2, rank_index=0, seed=17)
    rank1 = DistributedWeightedSampler(weights, replicas=2, rank_index=1, seed=17)
    rank0.set_epoch(4)
    rank1.set_epoch(4)
    generator = torch.Generator().manual_seed(21)
    expected = torch.multinomial(weights, 4, replacement=True, generator=generator).tolist()
    assert list(rank0) == expected[0::2]
    assert list(rank1) == expected[1::2]
