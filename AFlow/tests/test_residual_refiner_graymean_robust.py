from __future__ import annotations

import pytest
import torch

from scripts.train_t1ce_wt_residual_refiner_graymean_robust import (
    corrupt_coarse_batch,
)


def test_coarse_corruption_zero_probability_is_exact_identity() -> None:
    coarse = torch.randn((2, 1, 7, 7, 7))
    result = corrupt_coarse_batch(coarse.clone(), max_shift=4, probability=0)
    assert torch.equal(result, coarse)


def test_coarse_corruption_translates_without_wrap() -> None:
    coarse = torch.zeros((1, 1, 7, 7, 7))
    coarse[..., 3, 3, 3] = 1
    torch.manual_seed(0)
    result = corrupt_coarse_batch(coarse.clone(), max_shift=1, probability=1)
    assert not torch.equal(result, coarse)
    assert int(torch.count_nonzero(result)) == 1


def test_coarse_corruption_rejects_invalid_configuration() -> None:
    coarse = torch.zeros((1, 1, 2, 2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        corrupt_coarse_batch(coarse, max_shift=-1, probability=1)
    with pytest.raises(ValueError, match="\[0, 1\]"):
        corrupt_coarse_batch(coarse, max_shift=1, probability=2)
