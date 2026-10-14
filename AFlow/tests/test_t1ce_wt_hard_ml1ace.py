from __future__ import annotations

import copy

from monai.losses import DiceCELoss
import pytest
import torch

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_hard_ml1ace import (
    BUCKETIZE_RIGHT,
    ML1ACE_WEIGHT,
    NUM_BINS,
    aggregate_foreground_calibration,
    foreground_hard_ml1ace,
    hard_ml1ace_from_probabilities,
)


def _official_single_channel_reference(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Literal single-channel adaptation of the pinned official loss."""

    probability = torch.sigmoid(logits.float()).flatten(start_dim=2)
    target_float = target.float().flatten(start_dim=2)
    boundaries = torch.linspace(
        0.0,
        1.0 + torch.finfo(torch.float32).eps,
        NUM_BINS + 1,
        device=logits.device,
    )
    batch_size, channels = probability.shape[:2]
    mean_probability = torch.zeros(
        batch_size, channels, NUM_BINS, device=logits.device
    )
    mean_target = torch.zeros_like(mean_probability)
    counts = torch.zeros_like(mean_probability)
    for batch_index in range(batch_size):
        for channel_index in range(channels):
            index = torch.bucketize(
                probability[batch_index, channel_index],
                boundaries[1:],
                right=False,
            )
            counts[batch_index, channel_index] = torch.zeros_like(
                boundaries[1:]
            ).scatter_add(
                0,
                index,
                torch.ones_like(probability[batch_index, channel_index]),
            )
            mean_probability[batch_index, channel_index] = torch.empty_like(
                boundaries[1:]
            ).scatter_reduce(
                0,
                index,
                probability[batch_index, channel_index],
                reduce="mean",
                include_self=False,
            )
            mean_target[batch_index, channel_index] = torch.empty_like(
                boundaries[1:]
            ).scatter_reduce(
                0,
                index,
                target_float[batch_index, channel_index],
                reduce="mean",
                include_self=False,
            )
    mean_probability = mean_probability.masked_fill(counts == 0, torch.nan)
    mean_target = mean_target.masked_fill(counts == 0, torch.nan)
    per_case = torch.nanmean(torch.abs(mean_probability - mean_target), dim=-1)
    nonempty = target_float.sum(dim=-1) > 0
    per_case = per_case * nonempty.float()
    return per_case.mean(), per_case[:, 0], counts[:, 0], mean_probability[:, 0]


def test_frozen_constants_and_official_reference_loss_and_gradient_match() -> None:
    assert NUM_BINS == 20
    assert BUCKETIZE_RIGHT is False
    assert ML1ACE_WEIGHT == 1.0
    logits = torch.tensor(
        [
            [[[[0.2, -1.7], [2.3, 0.0]], [[-0.5, 1.1], [3.0, -2.0]]]],
            [[[[1.4, -0.9], [0.7, -2.4]], [[2.1, 0.3], [-1.1, 1.8]]]],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [[[[1, 0], [1, 0]], [[0, 1], [1, 0]]]],
            [[[[1, 0], [0, 0]], [[1, 0], [0, 1]]]],
        ],
        dtype=torch.float32,
    )

    observed = foreground_hard_ml1ace(logits, target)
    expected_loss, expected_cases, expected_counts, expected_means = (
        _official_single_channel_reference(logits, target)
    )
    torch.testing.assert_close(observed.loss, expected_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        observed.per_case_ace, expected_cases, rtol=1e-6, atol=1e-7
    )
    assert torch.equal(observed.counts.to(expected_counts), expected_counts)
    torch.testing.assert_close(
        observed.mean_probabilities[observed.occupied],
        expected_means[observed.occupied],
        rtol=1e-6,
        atol=1e-7,
    )

    observed_gradient = torch.autograd.grad(observed.loss, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_loss, logits)[0]
    torch.testing.assert_close(
        observed_gradient, expected_gradient, rtol=1e-5, atol=1e-7
    )


def test_official_right_false_boundary_assignment_including_zero_and_one() -> None:
    probability = torch.tensor(
        [[[[[0.0, 0.05, 0.10, 0.50, 0.95, 1.0]]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[[[0, 0, 0, 1, 1, 1]]]]], dtype=torch.float32)
    result = hard_ml1ace_from_probabilities(probability, target)
    expected = torch.zeros(NUM_BINS, dtype=torch.int64)
    expected[0] = 2
    expected[1] = 1
    expected[9] = 1
    expected[18] = 1
    expected[19] = 1
    assert torch.equal(result.counts[0].cpu(), expected)
    assert result.counts.sum().item() == probability.numel()
    assert result.occupied.sum().item() == 5

    result.loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()
    assert probability.grad.abs().sum().item() > 0.0


def test_empty_bins_are_ignored_and_empty_ground_truth_case_loss_is_zero() -> None:
    probability = torch.tensor(
        [
            [[[[0.02, 0.02, 0.98, 0.98]]]],
            [[[[0.02, 0.02, 0.98, 0.98]]]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [[[[0, 0, 0, 0]]]],
            [[[[0, 0, 1, 1]]]],
        ],
        dtype=torch.float32,
    )
    result = hard_ml1ace_from_probabilities(probability, target)
    assert result.occupied.sum(dim=1).tolist() == [2, 2]
    assert result.target_nonempty.tolist() == [False, True]
    assert result.per_case_ace[0].item() == 0.0
    assert result.per_case_ece[0].item() == 0.0
    assert result.per_case_mce[0].item() == 0.0
    assert result.per_case_ace[1].item() == pytest.approx(0.02, abs=1e-7)
    assert result.loss.item() == pytest.approx(0.01, abs=1e-7)

    gradient = torch.autograd.grad(result.loss, probability)[0]
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))
    assert gradient[1].abs().sum().item() > 0.0


def test_joint_voxel_permutation_preserves_every_calibration_summary() -> None:
    probability = torch.tensor(
        [[[[[0.04, 0.07, 0.13, 0.51, 0.72, 0.97, 0.42, 0.28]]]]]
    )
    target = torch.tensor([[[[[0, 0, 1, 1, 0, 1, 0, 1]]]]], dtype=torch.float32)
    permutation = torch.tensor([5, 1, 7, 2, 4, 0, 6, 3])
    permuted_probability = probability.flatten()[permutation].reshape_as(probability)
    permuted_target = target.flatten()[permutation].reshape_as(target)

    left = hard_ml1ace_from_probabilities(probability, target)
    right = hard_ml1ace_from_probabilities(permuted_probability, permuted_target)
    assert torch.equal(left.counts, right.counts)
    torch.testing.assert_close(left.probability_sums, right.probability_sums)
    torch.testing.assert_close(left.target_sums, right.target_sums)
    torch.testing.assert_close(left.loss, right.loss)
    torch.testing.assert_close(
        aggregate_foreground_calibration(left).official_micro_ace,
        aggregate_foreground_calibration(right).official_micro_ace,
    )


def test_bfloat16_logits_are_converted_to_float32_before_sigmoid_and_binning() -> None:
    logits_bfloat16 = torch.tensor(
        [[[[[-3.0, -1.0, 0.0, 1.0, 3.0]]]]], dtype=torch.bfloat16
    ).requires_grad_(True)
    target = torch.tensor([[[[[0, 0, 1, 1, 1]]]]], dtype=torch.float32)
    observed = foreground_hard_ml1ace(logits_bfloat16, target)
    reference_logits = logits_bfloat16.detach().float().requires_grad_(True)
    expected = foreground_hard_ml1ace(reference_logits, target)

    assert observed.loss.dtype == torch.float32
    assert observed.probability_sums.dtype == torch.float32
    assert observed.mean_probabilities.dtype == torch.float32
    assert torch.equal(observed.counts, expected.counts)
    assert torch.equal(observed.loss, expected.loss)
    observed.loss.backward()
    assert logits_bfloat16.grad is not None
    assert logits_bfloat16.grad.dtype == torch.bfloat16
    assert torch.isfinite(logits_bfloat16.grad).all()


def test_assignment_is_detached_but_ace_gradient_reaches_raw_logits() -> None:
    logits = torch.tensor(
        [[[[[-2.0, -0.7, 0.2, 0.8, 2.2, 3.0]]]]], requires_grad=True
    )
    target = torch.tensor([[[[[0, 0, 0, 1, 1, 0]]]]], dtype=torch.float32)
    result = foreground_hard_ml1ace(logits, target)
    gradient = torch.autograd.grad(result.loss, logits)[0]
    assert torch.isfinite(result.loss)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0
    assert (gradient != 0).sum().item() > 0
    assert result.counts.requires_grad is False
    assert result.probability_sums.requires_grad is True


def test_loss_and_aggregate_helpers_do_not_consume_rng() -> None:
    probability = torch.tensor(
        [[[[[0.02, 0.08, 0.55, 0.91]]]]], dtype=torch.float32
    )
    target = torch.tensor([[[[[0, 0, 1, 1]]]]], dtype=torch.float32)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    statistics = hard_ml1ace_from_probabilities(probability, target)
    aggregate_foreground_calibration(statistics)
    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                torch.cuda.get_rng_state_all(), cuda_before, strict=True
            )
        )


def test_zero_weight_full_model_adamw_update_exactly_matches_control() -> None:
    torch.manual_seed(20260860)
    control = AutomaticPromptLocator(
        center_aware=False, init_filters=8, dropout=0.1
    ).train()
    hace = copy.deepcopy(control).train()
    optimizer_kwargs = {
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
    }
    control_optimizer = torch.optim.AdamW(control.parameters(), **optimizer_kwargs)
    hace_optimizer = torch.optim.AdamW(hace.parameters(), **optimizer_kwargs)
    image = torch.randn((1, 1, 16, 16, 16))
    target = (torch.rand((1, 1, 16, 16, 16)) > 0.9).float()
    supervised = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1,
    )

    control_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(999)
    control_logits = control(image).segmentation_logits
    control_loss = supervised(control_logits, target)
    control_loss.backward()
    torch.nn.utils.clip_grad_norm_(control.parameters(), max_norm=5.0)
    control_optimizer.step()

    hace_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(999)
    hace_logits = hace(image).segmentation_logits
    raw_hace = foreground_hard_ml1ace(hace_logits, target).loss
    hace_loss = supervised(hace_logits, target) + 0.0 * raw_hace
    hace_loss.backward()
    torch.nn.utils.clip_grad_norm_(hace.parameters(), max_norm=5.0)
    hace_optimizer.step()

    for name, value in control.state_dict().items():
        assert torch.equal(value, hace.state_dict()[name]), name


def test_macro_official_micro_and_pooled_voxel_ace_are_distinct_and_exact() -> None:
    # Both scans occupy the same probability bin, but they have unequal voxel
    # counts and different scan-level mean probabilities.  This isolates the
    # three frozen aggregation rules.
    first = hard_ml1ace_from_probabilities(
        torch.tensor([[[[[0.06, 0.06]]]]]),
        torch.tensor([[[[[0, 1]]]]], dtype=torch.float32),
    )
    second = hard_ml1ace_from_probabilities(
        torch.tensor([[[[[0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09]]]]]),
        torch.tensor([[[[[1, 1, 0, 0, 0, 0, 0, 0]]]]], dtype=torch.float32),
    )
    aggregate = aggregate_foreground_calibration((first, second))

    assert aggregate.case_count == 2
    assert aggregate.macro_ace.item() == pytest.approx(0.30, abs=1e-7)
    assert aggregate.official_micro_ace.item() == pytest.approx(0.225, abs=1e-7)
    assert aggregate.pooled_voxel_ace.item() == pytest.approx(0.216, abs=1e-7)
    assert len(
        {
            round(float(aggregate.macro_ace), 6),
            round(float(aggregate.official_micro_ace), 6),
            round(float(aggregate.pooled_voxel_ace), 6),
        }
    ) == 3
    occupied_bin = int(torch.nonzero(aggregate.occupied, as_tuple=False)[0])
    assert aggregate.occupied_case_counts[occupied_bin].item() == 2
    assert aggregate.bin_counts[occupied_bin].item() == 10
    assert aggregate.official_micro_mean_probabilities[occupied_bin].item() == pytest.approx(
        0.075, abs=1e-7
    )
    assert aggregate.pooled_mean_probabilities[occupied_bin].item() == pytest.approx(
        0.084, abs=1e-7
    )
    assert aggregate.official_micro_mean_targets[occupied_bin].item() == pytest.approx(
        0.3, abs=1e-7
    )


@pytest.mark.parametrize(
    "probability,target,error",
    [
        (torch.ones((1, 2, 1, 1, 1)), torch.ones((1, 2, 1, 1, 1)), ValueError),
        (torch.ones((1, 1, 1, 1, 1)) * 1.1, torch.ones((1, 1, 1, 1, 1)), ValueError),
        (torch.ones((1, 1, 1, 1, 1)), torch.ones((1, 1, 1, 1, 2)), ValueError),
        (torch.ones((1, 1, 1, 1, 1)), torch.ones((1, 1, 1, 1, 1)) * 0.5, ValueError),
    ],
)
def test_invalid_probability_or_target_contract_is_rejected(
    probability: torch.Tensor,
    target: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        hard_ml1ace_from_probabilities(probability, target)
