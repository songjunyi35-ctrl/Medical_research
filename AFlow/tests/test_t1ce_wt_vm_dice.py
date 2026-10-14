from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sammed3d.t1ce_wt_vm_dice import (
    BCE_WEIGHT,
    CLASS_NAMES,
    VM_DICE_ARM_GAMMAS,
    VM_DICE_EPSILON,
    binary_probability_and_target_channels,
    vm_dice_loss_for_arm,
    voxel_modulated_dice_loss,
    voxel_modulation,
)


def _manual_vm_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    foreground_probability = torch.sigmoid(logits.float())
    foreground_target = target.float()
    probability = torch.cat((1 - foreground_probability, foreground_probability), dim=1)
    classes = torch.cat((1 - foreground_target, foreground_target), dim=1)
    modulation = (
        torch.ones_like(probability)
        if gamma == 0
        else torch.abs(classes - probability.detach())
    )
    dimensions = (2, 3, 4)
    scores = (
        2 * (modulation * classes * probability).sum(dim=dimensions)
        + VM_DICE_EPSILON
    ) / (
        (modulation * (classes.square() + probability.square())).sum(dim=dimensions)
        + VM_DICE_EPSILON
    )
    present = classes.sum(dim=dimensions) > 0
    per_case = 1 - (scores * present).sum(dim=1) / present.sum(dim=1)
    return per_case.mean(), per_case, scores, modulation


def _logits_from_probabilities(values: list[float]) -> torch.Tensor:
    probability = torch.tensor(values, dtype=torch.float32)
    return torch.logit(probability).reshape(1, 1, 1, 1, -1)


def test_frozen_constants_arm_names_and_class_order() -> None:
    assert VM_DICE_ARM_GAMMAS == {
        "paper_dice_gamma0": 0.0,
        "vm_dice_gamma1": 1.0,
    }
    assert CLASS_NAMES == ("background", "foreground")
    assert VM_DICE_EPSILON == 1e-6
    assert BCE_WEIGHT == 0.1


def test_gamma0_matches_manual_squared_denominator_dice_exactly() -> None:
    logits = _logits_from_probabilities([0.2, 0.8, 0.6, 0.1]).requires_grad_(True)
    target = torch.tensor([[[[[0, 1, 1, 0]]]]], dtype=torch.float32)
    observed = voxel_modulated_dice_loss(logits, target, gamma=0.0)
    expected, per_case, scores, modulation = _manual_vm_dice(
        logits, target, gamma=0.0
    )
    assert torch.equal(observed.modulation, torch.ones_like(observed.modulation))
    torch.testing.assert_close(observed.loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed.per_case_loss, per_case, rtol=0, atol=0)
    torch.testing.assert_close(observed.class_scores, scores, rtol=0, atol=0)
    torch.testing.assert_close(observed.modulation, modulation, rtol=0, atol=0)

    # The frozen paper control uses p^2 in the denominator and is deliberately
    # not the historical unsquared MONAI Dice.
    probability, classes = binary_probability_and_target_channels(logits, target)
    unsquared_score = (
        2 * (probability * classes).sum((2, 3, 4)) + VM_DICE_EPSILON
    ) / (
        (probability + classes).sum((2, 3, 4)) + VM_DICE_EPSILON
    )
    unsquared_loss = 1 - unsquared_score.mean()
    assert not torch.isclose(observed.loss, unsquared_loss)


def test_gamma1_matches_manual_toy_and_weights_higher_error_more() -> None:
    logits = _logits_from_probabilities([0.9, 0.2, 0.7, 0.1]).requires_grad_(True)
    target = torch.tensor([[[[[1, 1, 0, 0]]]]], dtype=torch.float32)
    observed = voxel_modulated_dice_loss(logits, target, gamma=1.0)
    expected, per_case, scores, modulation = _manual_vm_dice(
        logits, target, gamma=1.0
    )
    torch.testing.assert_close(observed.loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(observed.per_case_loss, per_case, rtol=0, atol=0)
    torch.testing.assert_close(observed.class_scores, scores, rtol=0, atol=0)
    torch.testing.assert_close(observed.modulation, modulation, rtol=0, atol=0)

    foreground_weights = observed.modulation[0, 1, 0, 0]
    assert foreground_weights[1] > foreground_weights[0]  # FN: p=.2 versus p=.9
    assert foreground_weights[2] > foreground_weights[3]  # FP: p=.7 versus p=.1


def test_modulation_is_detached_but_regular_probability_path_has_gradient() -> None:
    logits = _logits_from_probabilities([0.15, 0.45, 0.75, 0.95]).requires_grad_(True)
    target = torch.tensor([[[[[0, 1, 0, 1]]]]], dtype=torch.float32)
    output = voxel_modulated_dice_loss(logits, target, gamma=1.0)
    assert output.modulation.requires_grad is False
    assert output.class_scores.requires_grad is True
    gradient = torch.autograd.grad(output.loss, logits)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0

    probability, classes = binary_probability_and_target_channels(logits, target)
    direct = voxel_modulation(probability, classes, gamma=1.0)
    assert direct.requires_grad is False
    assert torch.equal(direct, torch.abs(classes - probability.detach()))


def test_present_class_rule_for_empty_full_and_mixed_foreground() -> None:
    logits = torch.zeros((3, 1, 1, 1, 4), dtype=torch.float32)
    target = torch.tensor(
        [
            [[[[0, 0, 0, 0]]]],
            [[[[1, 1, 1, 1]]]],
            [[[[0, 1, 0, 1]]]],
        ],
        dtype=torch.float32,
    )
    output = voxel_modulated_dice_loss(logits, target, gamma=1.0)
    assert output.present_classes.tolist() == [
        [True, False],
        [False, True],
        [True, True],
    ]
    assert output.present_class_counts.tolist() == [1, 1, 2]
    manual, per_case, _, _ = _manual_vm_dice(logits, target, gamma=1.0)
    torch.testing.assert_close(output.loss, manual)
    torch.testing.assert_close(output.per_case_loss, per_case)


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_foreground_background_exchange_is_symmetric_when_gamma_matches(
    gamma: float,
) -> None:
    logits = torch.tensor([[[[[-2.0, -0.4, 0.3, 1.7, 2.5]]]]])
    target = torch.tensor([[[[[0, 1, 0, 1, 1]]]]], dtype=torch.float32)
    original = voxel_modulated_dice_loss(logits, target, gamma)
    exchanged = voxel_modulated_dice_loss(-logits, 1.0 - target, gamma)
    # sigmoid(-z) and 1-sigmoid(z) can differ by one float32 ULP.
    torch.testing.assert_close(original.loss, exchanged.loss, rtol=0, atol=1e-7)
    torch.testing.assert_close(original.total, exchanged.total, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        original.class_scores,
        torch.flip(exchanged.class_scores, dims=(1,)),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "target_value,logit_value,expected_present",
    [
        (0.0, -80.0, [True, False]),
        (1.0, 80.0, [False, True]),
    ],
)
def test_perfect_empty_or_full_prediction_has_zero_loss(
    target_value: float,
    logit_value: float,
    expected_present: list[bool],
) -> None:
    logits = torch.full((1, 1, 1, 2, 2), logit_value)
    target = torch.full_like(logits, target_value)
    for gamma in (0.0, 1.0):
        output = voxel_modulated_dice_loss(logits, target, gamma=gamma)
        assert output.present_classes[0].tolist() == expected_present
        assert output.loss.item() == pytest.approx(0.0, abs=1e-7)
        assert output.total.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("target_value", [0.0, 1.0])
def test_single_voxel_uses_only_its_present_class(target_value: float) -> None:
    logits = torch.zeros((1, 1, 1, 1, 1), requires_grad=True)
    target = torch.full_like(logits, target_value)
    output = voxel_modulated_dice_loss(logits, target, gamma=1.0)
    expected_score = (0.5 + VM_DICE_EPSILON) / (0.625 + VM_DICE_EPSILON)
    assert output.present_class_counts.item() == 1
    assert output.loss.item() == pytest.approx(1 - expected_score, abs=1e-7)
    output.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().item() > 0


def test_reduction_is_present_class_per_volume_then_batch_mean() -> None:
    logits = torch.tensor(
        [
            [[[[0.0, -1.0, -2.0, -3.0]]]],
            [[[[2.0, 1.0, 0.0, -1.0]]]],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [[[[0, 0, 0, 0]]]],
            [[[[0, 1, 1, 0]]]],
        ],
        dtype=torch.float32,
    )
    batch = voxel_modulated_dice_loss(logits, target, gamma=1.0)
    singles = [
        voxel_modulated_dice_loss(
            logits[index : index + 1], target[index : index + 1], gamma=1.0
        ).loss
        for index in range(2)
    ]
    torch.testing.assert_close(batch.loss, torch.stack(singles).mean())


def test_bfloat16_logits_use_float32_loss_and_frozen_bce_weight() -> None:
    logits = torch.tensor(
        [[[[[-2.0, -0.5, 0.5, 2.0]]]]], dtype=torch.bfloat16
    ).requires_grad_(True)
    target = torch.tensor([[[[[0, 0, 1, 1]]]]], dtype=torch.float32)
    output = voxel_modulated_dice_loss(logits, target, gamma=1.0)
    expected_bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="mean"
    )
    assert output.loss.dtype == torch.float32
    assert output.class_scores.dtype == torch.float32
    assert output.bce.dtype == torch.float32
    torch.testing.assert_close(output.bce, expected_bce, rtol=0, atol=0)
    torch.testing.assert_close(
        output.total, output.loss + 0.1 * expected_bce, rtol=0, atol=0
    )
    output.total.backward()
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16
    assert torch.isfinite(logits.grad).all()


def test_gamma1_finite_difference_and_detached_autograd_agree_in_direction() -> None:
    logits = torch.tensor(
        [[[[[-1.7, -0.6, 0.2, 0.9, 2.1]]]]], requires_grad=True
    )
    target = torch.tensor([[[[[0, 1, 0, 1, 1]]]]], dtype=torch.float32)
    direction = torch.tensor([[[[[0.4, -0.7, 0.3, -0.2, 0.5]]]]])
    output = voxel_modulated_dice_loss(logits, target, 1.0)
    gradient = torch.autograd.grad(output.loss, logits)[0]
    autograd_direction = float((gradient * direction).sum())
    step = 1e-3
    upper = voxel_modulated_dice_loss(
        (logits.detach() + step * direction), target, 1.0
    ).loss
    lower = voxel_modulated_dice_loss(
        (logits.detach() - step * direction), target, 1.0
    ).loss
    finite_difference_direction = float((upper - lower) / (2 * step))
    assert autograd_direction != 0.0
    assert finite_difference_direction != 0.0
    assert autograd_direction * finite_difference_direction > 0.0


def test_arm_helper_and_loss_do_not_consume_rng() -> None:
    logits = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]])
    target = torch.tensor([[[[[0, 1, 1]]]]], dtype=torch.float32)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    gamma0 = vm_dice_loss_for_arm(logits, target, arm="paper_dice_gamma0")
    gamma1 = vm_dice_loss_for_arm(logits, target, arm="vm_dice_gamma1")
    assert gamma0.gamma == 0.0
    assert gamma1.gamma == 1.0
    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                torch.cuda.get_rng_state_all(), cuda_before, strict=True
            )
        )


@pytest.mark.parametrize(
    "logits,target,gamma,error",
    [
        (torch.zeros((1, 2, 1, 1, 1)), torch.zeros((1, 2, 1, 1, 1)), 1.0, ValueError),
        (torch.zeros((1, 1, 1, 1, 1)), torch.zeros((1, 1, 1, 1, 2)), 1.0, ValueError),
        (torch.zeros((1, 1, 1, 1, 1)), torch.full((1, 1, 1, 1, 1), 0.5), 1.0, ValueError),
        (torch.zeros((1, 1, 1, 1, 1)), torch.zeros((1, 1, 1, 1, 1)), 2.0, ValueError),
    ],
)
def test_invalid_contract_is_rejected(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        voxel_modulated_dice_loss(logits, target, gamma)
