from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.train_t1ce_wt_plain_ema_full_cohort import update_ema


class Tiny(nn.Module):
    def __init__(self, weight: float, running: float, count: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight]))
        self.register_buffer("running", torch.tensor([running]))
        self.register_buffer("count", torch.tensor([count], dtype=torch.int64))


def test_update_ema_averages_parameters_and_float_buffers() -> None:
    ema = Tiny(2.0, 4.0, 1)
    raw = Tiny(6.0, 8.0, 7)
    update_ema(ema, raw, 0.75)
    assert ema.weight.item() == pytest.approx(3.0)
    assert ema.running.item() == pytest.approx(5.0)
    assert ema.count.item() == 7


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.1])
def test_update_ema_rejects_invalid_decay(decay: float) -> None:
    with pytest.raises(ValueError):
        update_ema(Tiny(1.0, 1.0, 1), Tiny(2.0, 2.0, 2), decay)


def test_update_ema_rejects_different_parameter_names() -> None:
    with pytest.raises(ValueError, match="parameter names"):
        update_ema(Tiny(1.0, 1.0, 1), nn.Linear(1, 1), 0.9)
