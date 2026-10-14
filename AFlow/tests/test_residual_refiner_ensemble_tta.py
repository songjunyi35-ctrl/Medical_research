from __future__ import annotations

import torch
from torch import nn

from scripts.evaluate_t1ce_wt_residual_refiner_ensemble_tta_validation import (
    ensemble_tta_logits,
)


class ScaleModel(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, image: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return self.scale * image + coarse


def test_equal_model_and_reflection_mean() -> None:
    image = torch.randn((1, 1, 3, 4, 5))
    coarse = torch.randn_like(image)
    result = ensemble_tta_logits(
        [ScaleModel(1.0), ScaleModel(3.0)],
        image,
        coarse,
    )
    assert torch.allclose(result, 2.0 * image + coarse)
