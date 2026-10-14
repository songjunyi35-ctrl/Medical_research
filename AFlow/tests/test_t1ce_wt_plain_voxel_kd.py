from __future__ import annotations

import pytest
import torch

from scripts.train_t1ce_wt_plain_voxel_kd_continuation import binary_voxel_kl


def test_binary_voxel_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[[[[0.0, 1.0]]]]])
    value = binary_voxel_kl(logits, logits, temperature=2.0)
    assert value.item() == pytest.approx(0.0, abs=1e-7)


def test_binary_voxel_kl_is_positive_and_differentiable() -> None:
    student = torch.tensor([[[[[0.0, 0.0]]]]], requires_grad=True)
    teacher = torch.tensor([[[[[2.0, -2.0]]]]])
    value = binary_voxel_kl(student, teacher, temperature=2.0)
    assert value.item() > 0.0
    value.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_binary_voxel_kl_rejects_nonpositive_temperature() -> None:
    logits = torch.zeros((1, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        binary_voxel_kl(logits, logits, temperature=0.0)
