from __future__ import annotations

import copy

from monai.losses import DiceCELoss
import pytest
import torch

from sammed3d.t1ce_wt_batch_cvar50 import (
    BATCH_CVAR50_ARMS,
    BATCH_SIZE,
    CE_WEIGHT,
    DICE_SMOOTH_DENOMINATOR,
    DICE_SMOOTH_NUMERATOR,
    TOP_COUNT,
    aggregate,
    aggregate_batch_case_loss,
    batch_cvar50_loss,
    per_draw_dicece,
    per_draw_dicece_loss,
)


def _legacy_loss() -> DiceCELoss:
    return DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1,
    )


def _manual_per_draw(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target_float = target.float()
    intersection = (probability * target_float).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target_float.flatten(1).sum(1)
    dice = 1 - (
        2 * intersection + DICE_SMOOTH_NUMERATOR
    ) / (denominator + DICE_SMOOTH_DENOMINATOR)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target_float, reduction="none"
    ).flatten(1).mean(1)
    return dice + CE_WEIGHT * ce


def _toy_batch(*, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260860)
    logits = torch.randn((BATCH_SIZE, 1, 2, 3, 4), dtype=torch.float32)
    logits.requires_grad_(requires_grad)
    target = (torch.rand((BATCH_SIZE, 1, 2, 3, 4)) > 0.72).float()
    return logits, target


def test_frozen_constants_and_arm_order() -> None:
    assert BATCH_CVAR50_ARMS == ("uniform_mean8", "batch_cvar50_top4")
    assert BATCH_SIZE == 8
    assert TOP_COUNT == 4
    assert CE_WEIGHT == 0.1
    assert DICE_SMOOTH_NUMERATOR == DICE_SMOOTH_DENOMINATOR == 1e-5


def test_per_draw_dicece_matches_manual_historical_formula() -> None:
    logits, target = _toy_batch()
    output = per_draw_dicece(logits, target)
    expected = _manual_per_draw(logits, target)
    assert output.per_draw_loss.shape == (BATCH_SIZE,)
    assert output.per_draw_dice.shape == (BATCH_SIZE,)
    assert output.per_draw_ce.shape == (BATCH_SIZE,)
    assert output.per_draw_loss.dtype == torch.float32
    torch.testing.assert_close(output.per_draw_loss, expected, rtol=1e-6, atol=1e-7)
    alias = per_draw_dicece_loss(logits, target)
    torch.testing.assert_close(alias.per_draw_loss, output.per_draw_loss)


def test_uniform_mean_matches_legacy_dicece_loss_and_raw_logit_gradient() -> None:
    logits, target = _toy_batch(requires_grad=True)
    legacy = _legacy_loss()(logits, target)
    components = per_draw_dicece(logits, target)
    uniform = aggregate(components.per_draw_loss, "uniform_mean8")
    torch.testing.assert_close(uniform.total, legacy, rtol=0, atol=1e-7)

    legacy_gradient = torch.autograd.grad(legacy, logits, retain_graph=True)[0]
    uniform_gradient = torch.autograd.grad(uniform.total, logits)[0]
    torch.testing.assert_close(
        uniform_gradient, legacy_gradient, rtol=0, atol=1e-7
    )
    assert uniform.selected_indices.tolist() == list(range(8))
    assert uniform.selected_mask.tolist() == [True] * 8


def test_uniform_full_model_adamw_step_matches_legacy_path() -> None:
    torch.manual_seed(91)
    legacy_model = torch.nn.Conv3d(1, 1, kernel_size=1, bias=True)
    uniform_model = copy.deepcopy(legacy_model)
    kwargs = {
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
    }
    legacy_optimizer = torch.optim.AdamW(legacy_model.parameters(), **kwargs)
    uniform_optimizer = torch.optim.AdamW(uniform_model.parameters(), **kwargs)
    image = torch.randn((8, 1, 3, 4, 5))
    target = (torch.rand((8, 1, 3, 4, 5)) > 0.7).float()

    legacy_optimizer.zero_grad(set_to_none=True)
    legacy_loss = _legacy_loss()(legacy_model(image), target)
    legacy_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        legacy_model.parameters(), max_norm=5.0, error_if_nonfinite=True
    )
    legacy_optimizer.step()

    uniform_optimizer.zero_grad(set_to_none=True)
    per_draw = per_draw_dicece(uniform_model(image), target).per_draw_loss
    uniform_loss = aggregate(per_draw, "uniform_mean8").total
    uniform_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        uniform_model.parameters(), max_norm=5.0, error_if_nonfinite=True
    )
    uniform_optimizer.step()

    torch.testing.assert_close(uniform_loss, legacy_loss, rtol=0, atol=1e-7)
    for left, right in zip(
        legacy_model.parameters(), uniform_model.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-7)
        legacy_state = legacy_optimizer.state[left]
        uniform_state = uniform_optimizer.state[right]
        assert torch.equal(legacy_state["step"], uniform_state["step"])
        torch.testing.assert_close(
            legacy_state["exp_avg"], uniform_state["exp_avg"], rtol=0, atol=1e-7
        )
        torch.testing.assert_close(
            legacy_state["exp_avg_sq"],
            uniform_state["exp_avg_sq"],
            rtol=0,
            atol=1e-7,
        )


def test_cvar_selects_stable_lexicographic_largest_four() -> None:
    losses = torch.tensor([3.0, 5.0, 5.0, 1.0, 4.0, 5.0, 2.0, 5.0])
    output = aggregate(losses, "batch_cvar50_top4")
    assert output.selected_indices.tolist() == [1, 2, 5, 7]
    assert output.selected_losses.tolist() == [5.0, 5.0, 5.0, 5.0]
    assert output.selected_mask.tolist() == [False, True, True, False, False, True, False, True]
    assert output.total.item() == 5.0


def test_selection_is_detached_and_only_selected_losses_receive_gradient() -> None:
    losses = torch.tensor(
        [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4], requires_grad=True
    )
    output = aggregate_batch_case_loss(
        losses, arm="batch_cvar50_top4"
    )
    output.loss.backward()
    assert output.selected_indices.tolist() == [0, 2, 4, 6]
    expected = torch.tensor([0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0])
    assert torch.equal(losses.grad, expected)
    assert output.selected_indices.requires_grad is False
    assert output.selected_mask.requires_grad is False


def test_duplicate_draws_remain_separate_and_can_both_be_selected() -> None:
    # Draws 0 and 1 represent the same case ID after independent augmentation.
    # Aggregation receives draw positions and must never deduplicate them.
    losses = torch.tensor([0.91, 0.90, 0.8, 0.7, 0.2, 0.3, 0.4, 0.1])
    case_ids = ["duplicate", "duplicate", "c2", "c3", "c4", "c5", "c6", "c7"]
    output = aggregate(losses, "batch_cvar50_top4")
    selected_ids = [case_ids[index] for index in output.selected_indices.tolist()]
    assert output.selected_indices.tolist() == [0, 1, 2, 3]
    assert selected_ids.count("duplicate") == 2


def test_non_tied_permutation_preserves_cvar_value() -> None:
    losses = torch.tensor([0.11, 0.82, 0.23, 0.74, 0.35, 0.66, 0.47, 0.58])
    permutation = torch.tensor([5, 0, 7, 2, 6, 1, 3, 4])
    original = aggregate(losses, "batch_cvar50_top4")
    permuted = aggregate(losses[permutation], "batch_cvar50_top4")
    assert torch.equal(original.total, permuted.total)


def test_bfloat16_model_logits_produce_float32_draw_losses() -> None:
    logits, target = _toy_batch()
    logits = logits.to(torch.bfloat16).requires_grad_(True)
    output = batch_cvar50_loss(
        logits, target, arm="batch_cvar50_top4"
    )
    assert output.per_draw_loss.dtype == torch.float32
    assert output.total.dtype == torch.float32
    assert output.selected_indices.dtype == torch.int64
    output.loss.backward()
    assert logits.grad is not None
    assert logits.grad.dtype == torch.bfloat16
    assert torch.isfinite(logits.grad).all()


def test_aggregation_and_selection_do_not_consume_rng() -> None:
    losses = torch.linspace(0.1, 0.8, 8)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    aggregate(losses, "uniform_mean8")
    aggregate(losses, "batch_cvar50_top4")
    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                torch.cuda.get_rng_state_all(), cuda_before, strict=True
            )
        )


@pytest.mark.parametrize(
    "losses,arm,error",
    [
        (torch.ones(7), "uniform_mean8", ValueError),
        (torch.ones(9), "batch_cvar50_top4", ValueError),
        (torch.ones(8), "unknown", ValueError),
        (torch.tensor([0.0, 1.0, 2.0, 3.0, float("nan"), 5.0, 6.0, 7.0]), "uniform_mean8", ValueError),
    ],
)
def test_invalid_aggregation_contract_is_rejected(
    losses: torch.Tensor,
    arm: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        aggregate(losses, arm)


def test_invalid_logit_target_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        per_draw_dicece(
            torch.zeros((8, 2, 2, 2, 2)), torch.zeros((8, 2, 2, 2, 2))
        )
    with pytest.raises(ValueError, match="binary"):
        per_draw_dicece(
            torch.zeros((8, 1, 2, 2, 2)), torch.full((8, 1, 2, 2, 2), 0.5)
        )
