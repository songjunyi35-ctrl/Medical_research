from __future__ import annotations

import pytest
import torch

from scripts.train_t1ce_wt_plain_reco_kd_continuation import (
    region_normalized_logit_kd,
)


def _label() -> torch.Tensor:
    label = torch.zeros((1, 1, 9, 9, 9))
    label[:, :, 3:6, 3:6, 3:6] = 1
    return label


def test_region_kd_is_zero_for_identical_logits() -> None:
    logits = torch.randn((1, 1, 9, 9, 9))
    assert region_normalized_logit_kd(logits, logits, _label()).item() == pytest.approx(0.0)


def test_region_kd_is_positive_for_foreground_and_background_errors() -> None:
    student = torch.zeros((1, 1, 9, 9, 9), requires_grad=True)
    teacher = torch.zeros_like(student)
    teacher[:, :, 4, 4, 4] = 2
    teacher[:, :, 0, 0, 0] = -2
    value = region_normalized_logit_kd(student, teacher, _label())
    assert value.item() > 0
    value.backward()
    assert student.grad is not None


def test_region_kd_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        region_normalized_logit_kd(
            torch.zeros((1, 1, 4, 4, 4)),
            torch.zeros((1, 1, 5, 5, 5)),
            torch.zeros((1, 1, 4, 4, 4))
        )
