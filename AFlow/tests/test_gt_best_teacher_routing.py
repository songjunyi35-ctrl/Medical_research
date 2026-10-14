from __future__ import annotations

import pytest
import torch

from scripts.train_t1ce_wt_gt_best_teacher_routing import (
    per_case_soft_dice_loss,
    select_gt_best_teacher_logits,
)


def test_soft_dice_loss_exact_simple_cases() -> None:
    label = torch.tensor([[[[[1.0, 0.0]]]]])
    logits = torch.logit(torch.tensor([[[[[0.8, 0.2]]]]]))
    expected = 1.0 - (2.0 * 0.8 + 1e-6) / (0.8 + 0.2 + 1.0 + 1e-6)
    assert float(per_case_soft_dice_loss(logits, label)) == pytest.approx(expected, abs=1e-7)


def test_best_teacher_selection_is_per_case_and_detached() -> None:
    label = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).reshape(2, 1, 1, 1, 2)
    logits0 = torch.logit(
        torch.tensor([[0.9, 0.1], [0.8, 0.2]]).reshape(2, 1, 1, 1, 2)
    ).requires_grad_(True)
    logits1 = torch.logit(
        torch.tensor([[0.2, 0.8], [0.1, 0.9]]).reshape(2, 1, 1, 1, 2)
    ).requires_grad_(True)
    logits2 = torch.zeros_like(logits0, requires_grad=True)
    selected, indices, losses = select_gt_best_teacher_logits([logits0, logits1, logits2], label)
    assert indices.tolist() == [0, 1]
    assert torch.equal(selected[0], logits0.detach()[0])
    assert torch.equal(selected[1], logits1.detach()[1])
    assert not selected.requires_grad
    assert not losses.requires_grad


def test_exact_tie_selects_first_teacher() -> None:
    label = torch.ones(1, 1, 1, 1, 2)
    first = torch.ones_like(label)
    second = first.clone()
    third = -torch.ones_like(label)
    selected, indices, _ = select_gt_best_teacher_logits([first, second, third], label)
    assert indices.tolist() == [0]
    assert torch.equal(selected, first)


def test_routing_contract_rejects_bad_inputs() -> None:
    label = torch.zeros(1, 1, 2, 2, 2)
    value = torch.zeros_like(label)
    with pytest.raises(ValueError, match="exactly three"):
        select_gt_best_teacher_logits([value, value], label)
    with pytest.raises(ValueError, match="shapes differ"):
        select_gt_best_teacher_logits([value, value, torch.zeros(2, 1, 2, 2, 2)], label)
    bad = value.clone()
    bad.flatten()[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        select_gt_best_teacher_logits([value, value, bad], label)
