from __future__ import annotations

import pytest
import torch

from scripts.train_t1ce_wt_residual_refiner_graymean import soft_gray_mean_mse


def test_soft_gray_mean_loss_is_near_zero_for_matching_region() -> None:
    image = torch.tensor([[[[[2.0, -1.0], [-1.0, -1.0]]]]])
    label = torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]])
    logits = torch.where(label > 0.5, torch.tensor(30.0), torch.tensor(-30.0))
    loss = soft_gray_mean_mse(logits, image, label)
    assert float(loss) < 1e-10


def test_soft_gray_mean_loss_has_finite_nonzero_gradient() -> None:
    image = torch.tensor([[[[[3.0, -1.0], [-1.0, -1.0]]]]])
    label = torch.tensor([[[[[1.0, 0.0], [0.0, 0.0]]]]])
    logits = torch.zeros_like(image, requires_grad=True)
    loss = soft_gray_mean_mse(logits, image, label)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert float(torch.linalg.vector_norm(logits.grad)) > 0


def test_soft_gray_mean_loss_rejects_invalid_inputs() -> None:
    value = torch.zeros((1, 1, 2, 2, 2))
    with pytest.raises(ValueError, match="temperature"):
        soft_gray_mean_mse(value, value, value, temperature=0)
    with pytest.raises(ValueError, match="identical shapes"):
        soft_gray_mean_mse(value, value[..., :1], value)
