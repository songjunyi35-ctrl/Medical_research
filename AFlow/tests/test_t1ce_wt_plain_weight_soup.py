from __future__ import annotations

import pytest
import torch

from scripts.make_t1ce_wt_plain_weight_soup import interpolate_state_dict


def test_weight_soup_interpolates_float_and_copies_integer() -> None:
    historical = {
        "weight": torch.tensor([2.0]),
        "count": torch.tensor([3], dtype=torch.int64)
    }
    ema = {
        "weight": torch.tensor([6.0]),
        "count": torch.tensor([9], dtype=torch.int64)
    }
    result = interpolate_state_dict(historical, ema, 0.25)
    assert result["weight"].item() == pytest.approx(3.0)
    assert result["count"].item() == 3


def test_weight_soup_rejects_different_keys() -> None:
    with pytest.raises(ValueError, match="keys"):
        interpolate_state_dict({"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])}, 0.25)
