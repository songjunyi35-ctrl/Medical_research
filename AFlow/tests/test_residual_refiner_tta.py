from __future__ import annotations

import torch
from torch import nn

from scripts.evaluate_t1ce_wt_residual_refiner_tta_validation import (
    reflection_combinations,
    tta_logits,
)


class EquivariantAdd(nn.Module):
    def forward(self, image: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return image + coarse


def test_reflection_set_is_complete_and_tta_inverts_transforms() -> None:
    combinations = reflection_combinations()
    assert len(combinations) == 8
    assert len(set(combinations)) == 8
    image = torch.randn((1, 1, 3, 4, 5))
    coarse = torch.randn_like(image)
    assert torch.allclose(tta_logits(EquivariantAdd(), image, coarse), image + coarse)
