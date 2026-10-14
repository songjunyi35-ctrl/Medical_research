from __future__ import annotations

import pytest
import torch

from scripts.evaluate_t1ce_wt_consensus_teachers_inner import consensus_logits


def test_consensus_is_exact_equal_raw_logit_mean() -> None:
    values = [
        torch.tensor([[[[[1.0, 2.0]]]]]),
        torch.tensor([[[[[4.0, 5.0]]]]]),
        torch.tensor([[[[[7.0, 8.0]]]]]),
    ]
    expected = torch.tensor([[[[[4.0, 5.0]]]]])
    assert torch.equal(consensus_logits(values), expected)


def test_consensus_requires_exactly_three_equal_shapes() -> None:
    value = torch.zeros(2, 1, 3, 3, 3)
    with pytest.raises(ValueError, match="exactly three"):
        consensus_logits([value, value])
    with pytest.raises(ValueError, match="shapes differ"):
        consensus_logits([value, value, torch.zeros(1, 1, 3, 3, 3)])


def test_consensus_rejects_nonfinite_logits() -> None:
    good = torch.zeros(1, 1, 2, 2, 2)
    bad = good.clone()
    bad.flatten()[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        consensus_logits([good, good, bad])
