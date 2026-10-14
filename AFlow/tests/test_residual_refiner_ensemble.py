from __future__ import annotations

import pytest
import torch

from scripts.evaluate_t1ce_wt_residual_refiner_ensemble_validation import mean_logits


def test_equal_logit_mean() -> None:
    values = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([6.0])]
    assert torch.equal(mean_logits(values), torch.tensor([3.0]))
    with pytest.raises(ValueError, match="at least"):
        mean_logits([])
    with pytest.raises(ValueError, match="identical"):
        mean_logits([torch.zeros(1), torch.zeros(2)])
