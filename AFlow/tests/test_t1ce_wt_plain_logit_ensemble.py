from __future__ import annotations

import pytest
import torch

from scripts.evaluate_t1ce_wt_plain_logit_ensemble_validation import (
    blend_logits,
    point_from_logits,
)


def test_blend_logits_uses_declared_ema_weight() -> None:
    historical = torch.tensor([[[[[2.0]]]]])
    ema = torch.tensor([[[[[6.0]]]]])
    assert blend_logits(historical, ema, 0.25).item() == pytest.approx(3.0)
    assert blend_logits(historical, ema, 0.5).item() == pytest.approx(4.0)


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_blend_logits_rejects_invalid_weight(weight: float) -> None:
    value = torch.zeros((1, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        blend_logits(value, value, weight)


def test_point_from_logits_returns_zyx_argmax() -> None:
    logits = torch.zeros((1, 1, 3, 4, 5))
    logits[0, 0, 2, 1, 4] = 7.0
    assert point_from_logits(logits).tolist() == [[2, 1, 4]]
