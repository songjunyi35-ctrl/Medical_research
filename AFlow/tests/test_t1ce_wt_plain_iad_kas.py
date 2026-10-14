from __future__ import annotations

from collections import OrderedDict

from monai.losses import DiceCELoss
import pytest
import torch
import torch.nn.functional as F

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_iad_kas import (
    KAS_DICE_EPSILON,
    KAS_GAMMA,
    KAS_TEMPERATURE,
    KASTrainingModel,
    KASTrainingOutput,
    PROJECTOR_INITIALIZATION_SEED,
    PROJECTOR_INITIAL_SHA256,
    kas_assimilation_loss,
    projector_parameter_sha256,
)


def _plain(
    *, center_aware: bool = False, init_filters: int = 8
) -> AutomaticPromptLocator:
    torch.manual_seed(11)
    return AutomaticPromptLocator(
        center_aware=center_aware,
        init_filters=init_filters,
        dropout=0.1,
    )


def _small_model() -> KASTrainingModel:
    return KASTrainingModel.from_plain_model(_plain())


def _checkpoint(plain: AutomaticPromptLocator) -> dict:
    return {
        "architecture": {
            "class": "AutomaticPromptLocator",
            "center_aware": False,
            "init_filters": plain.init_filters,
            "dropout": plain.dropout,
        },
        "model_state_dict": plain.state_dict(),
    }


def _manual_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    return (left * right).sum(dim=1) / (
        left.norm(dim=1) * right.norm(dim=1)
    ).clamp_min(1e-8)


def test_frozen_equal_weight_kas_formula_matches_manual_per_volume_result() -> None:
    final = torch.tensor(
        [
            [[[[0.5, -1.0], [2.0, 0.25]]]],
            [[[[1.5, 0.75], [-0.5, -2.0]]]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    projected = torch.tensor(
        [
            [[[[0.25, -0.5], [1.0, -0.25]]]],
            [[[[0.5, 1.25], [-1.5, -0.75]]]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = KASTrainingOutput(
        final_logits=final,
        deepest_feature=torch.ones((2, 1, 1, 1, 1), requires_grad=True),
        projected_logits=projected,
    )
    observed = kas_assimilation_loss(output)

    expected_mse = F.mse_loss(projected, final, reduction="mean")
    final_probability = torch.sigmoid(final)
    projected_probability = torch.sigmoid(projected)
    intersection = (final_probability * projected_probability).flatten(1).sum(1)
    denominator = final_probability.flatten(1).sum(1) + projected_probability.flatten(1).sum(1)
    expected_dice = (
        1
        - (2 * intersection + KAS_DICE_EPSILON)
        / (denominator + KAS_DICE_EPSILON)
    ).mean()
    negative = (1 - projected_probability) + (1 - final_probability)
    positive_cosine = _manual_cosine(final, projected)
    negative_cosine = _manual_cosine(final, negative)
    pair = torch.stack(
        (positive_cosine / KAS_TEMPERATURE, negative_cosine / KAS_TEMPERATURE),
        dim=1,
    )
    expected_contrastive = (
        torch.logsumexp(pair, dim=1) - positive_cosine / KAS_TEMPERATURE
    ).mean()

    assert KAS_GAMMA == 0.01
    assert KAS_TEMPERATURE == 1.0
    assert torch.equal(observed.mse, expected_mse)
    assert torch.equal(observed.dice, expected_dice)
    torch.testing.assert_close(
        observed.contrastive, expected_contrastive, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        observed.total,
        observed.mse + observed.dice + observed.contrastive,
        rtol=0.0,
        atol=0.0,
    )

    # The frozen aggregation is per-volume and then a batch mean, so adding a
    # second case cannot create cross-case cosine pairs.
    singles = [
        kas_assimilation_loss(
            KASTrainingOutput(
                final_logits=final[index : index + 1],
                deepest_feature=output.deepest_feature[index : index + 1],
                projected_logits=projected[index : index + 1],
            )
        ).contrastive
        for index in range(2)
    ]
    torch.testing.assert_close(
        observed.contrastive, torch.stack(singles).mean(), rtol=1e-6, atol=1e-7
    )


def test_projector_is_registered_training_only_and_in_optimizer() -> None:
    assert PROJECTOR_INITIALIZATION_SEED == 20260861
    full = KASTrainingModel(init_filters=16, dropout=0.1)
    assert full.projector.in_channels == 128
    assert full.projector.out_channels == 1
    assert full.projector.kernel_size == (1, 1, 1)
    assert full.projector.bias is not None
    assert projector_parameter_sha256(full.projector) == PROJECTOR_INITIAL_SHA256
    assert sum(value.numel() for value in full.deploy_state_dict().values()) == 4_700_897

    names = [name for name, _ in full.training_only_named_parameters()]
    assert names == ["projector.weight", "projector.bias"]
    assert [parameter for _, parameter in full.training_only_named_parameters()] == list(
        full.training_only_parameters()
    )
    assert "projector.weight" in full.state_dict()
    assert "projector.bias" in full.state_dict()
    assert not any(name.startswith("projector.") for name in full.deploy_state_dict())

    optimizer = torch.optim.AdamW(full.parameters(), lr=1e-5, weight_decay=1e-5)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimizer_ids for parameter in full.training_only_parameters())


def test_projector_seed_is_dedicated_and_its_initialization_is_rng_isolated() -> None:
    torch.manual_seed(101)
    first = KASTrainingModel(init_filters=16, dropout=0.1)
    first_hash = projector_parameter_sha256(first.projector)
    torch.manual_seed(909)
    second = KASTrainingModel(init_filters=16, dropout=0.1)
    second_hash = projector_parameter_sha256(second.projector)
    assert first_hash == second_hash == PROJECTOR_INITIAL_SHA256

    # Isolate the projector constructor itself from the network constructor:
    # replacing an already-built projector under the frozen fork must restore
    # the caller's CPU state exactly.
    state_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(PROJECTOR_INITIALIZATION_SEED)
        replacement = torch.nn.Conv3d(128, 1, kernel_size=1, bias=True)
    assert torch.equal(torch.get_rng_state(), state_before)
    assert projector_parameter_sha256(replacement) == PROJECTOR_INITIAL_SHA256

    # Relative to constructing the same plain network, adding the projector
    # consumes no caller RNG.  This is the property used by both plain-model
    # and plain-checkpoint wrapper construction before the formal RNG reset.
    torch.manual_seed(303)
    AutomaticPromptLocator(center_aware=False, init_filters=8, dropout=0.1)
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(303)
    KASTrainingModel(init_filters=8, dropout=0.1)
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_single_forward_shapes_and_plain_final_logits_are_exact() -> None:
    plain = _plain().eval()
    model = KASTrainingModel.from_plain_model(plain).eval()
    image = torch.randn((2, 1, 16, 24, 32))
    with torch.inference_mode():
        expected = plain(image).segmentation_logits
        output = model(image)
    assert torch.equal(output.final_logits, expected)
    assert output.deepest_feature.shape == (2, 64, 2, 3, 4)
    assert output.projected_logits.shape == expected.shape == (2, 1, 16, 24, 32)


def test_kas_only_backward_reaches_projector_encoder_and_final_branches() -> None:
    torch.manual_seed(19)
    model = _small_model().train()
    output = model(torch.randn((1, 1, 16, 16, 16))).retain_kas_grads()
    loss = model.kas_loss(output)
    assert all(
        torch.isfinite(value)
        for value in (loss.mse, loss.dice, loss.contrastive, loss.total)
    )
    loss.total.backward()

    for name, parameter in model.training_only_named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0.0, name
    for name, value in (
        ("deepest_feature", output.deepest_feature),
        ("final_logits", output.final_logits),
    ):
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name
        assert value.grad.abs().sum().item() > 0.0, name


def test_gamma_zero_full_model_adamw_update_exactly_matches_control() -> None:
    torch.manual_seed(2026)
    control = AutomaticPromptLocator(
        center_aware=False, init_filters=8, dropout=0.1
    ).train()
    kas = KASTrainingModel.from_plain_model(control).train()
    control_optimizer = torch.optim.AdamW(
        control.parameters(),
        lr=1e-5,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    kas_optimizer = torch.optim.AdamW(
        kas.parameters(),
        lr=1e-5,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )
    image = torch.randn((1, 1, 16, 16, 16))
    label = (torch.rand((1, 1, 16, 16, 16)) > 0.8).float()
    supervised_loss = DiceCELoss(
        sigmoid=True,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.1,
    )

    control_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(999)
    control_logits = control(image).segmentation_logits
    control_loss = supervised_loss(control_logits, label)
    control_loss.backward()
    torch.nn.utils.clip_grad_norm_(control.parameters(), max_norm=5.0)
    control_optimizer.step()

    kas_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(999)
    kas_output = kas(image)
    raw_kas = kas.kas_loss(kas_output)
    kas_loss = supervised_loss(kas_output.final_logits, label) + 0.0 * raw_kas.total
    kas_loss.backward()
    torch.nn.utils.clip_grad_norm_(kas.parameters(), max_norm=5.0)
    kas_optimizer.step()

    deploy_state = kas.deploy_state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(value, deploy_state[name]), name


def test_plain_checkpoint_training_roundtrip_and_deploy_are_exact(tmp_path) -> None:
    plain = _plain().eval()
    checkpoint = _checkpoint(plain)
    mapped = KASTrainingModel.from_plain_checkpoint(checkpoint).eval()
    path = tmp_path / "plain.pth"
    torch.save(checkpoint, path)
    loaded = KASTrainingModel.from_plain_checkpoint(path).eval()

    image = torch.randn((1, 1, 16, 16, 16))
    with torch.inference_mode():
        expected = plain(image).segmentation_logits
        assert torch.equal(mapped(image).final_logits, expected)
        assert torch.equal(loaded(image).final_logits, expected)

    clone = KASTrainingModel(init_filters=8, dropout=0.1).eval()
    clone.load_state_dict(mapped.state_dict(), strict=True)
    with torch.inference_mode():
        left = mapped(image)
        right = clone(image)
    assert torch.equal(left.final_logits, right.final_logits)
    assert torch.equal(left.deepest_feature, right.deepest_feature)
    assert torch.equal(left.projected_logits, right.projected_logits)

    deploy_state = mapped.deploy_state_dict(cpu=True)
    assert isinstance(deploy_state, OrderedDict)
    assert tuple(deploy_state) == tuple(plain.state_dict())
    deploy = mapped.to_deploy_model().eval()
    with torch.inference_mode():
        assert torch.equal(
            mapped(image).final_logits,
            deploy(image).segmentation_logits,
        )

    rng_before = torch.get_rng_state().clone()
    assert mapped.deploy_max_abs_difference(image) == 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert mapped.training is False


@pytest.mark.parametrize(
    "image,exception",
    [
        (torch.randn((1, 2, 16, 16, 16)), ValueError),
        (torch.randn((1, 1, 15, 16, 16)), ValueError),
        (torch.ones((1, 1, 16, 16, 16), dtype=torch.int64), TypeError),
    ],
)
def test_invalid_images_are_rejected(
    image: torch.Tensor, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        _small_model()(image)


def test_invalid_models_checkpoints_and_loss_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="center_aware"):
        KASTrainingModel.from_plain_model(_plain(center_aware=True))
    with pytest.raises(TypeError, match="AutomaticPromptLocator"):
        KASTrainingModel.from_plain_model(torch.nn.Identity())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="architecture"):
        KASTrainingModel.from_plain_checkpoint({"model_state_dict": {}})
    with pytest.raises(TypeError, match="mapping"):
        KASTrainingModel.from_plain_checkpoint(3.0)  # type: ignore[arg-type]

    checkpoint = _checkpoint(_plain())
    broken = dict(checkpoint)
    broken_state = OrderedDict(checkpoint["model_state_dict"])
    broken_state.pop(next(iter(broken_state)))
    broken["model_state_dict"] = broken_state
    with pytest.raises(RuntimeError, match="key mismatch"):
        KASTrainingModel.from_plain_checkpoint(broken)

    final = torch.randn((1, 1, 4, 4, 4))
    deep = torch.randn((1, 8, 1, 1, 1))
    with pytest.raises(TypeError, match="KASTrainingOutput"):
        kas_assimilation_loss(final)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identical shapes"):
        kas_assimilation_loss(
            KASTrainingOutput(final, deep, torch.randn((1, 1, 2, 4, 4)))
        )
    with pytest.raises(TypeError, match="floating"):
        kas_assimilation_loss(
            KASTrainingOutput(
                final.to(torch.int64), deep, torch.ones_like(final, dtype=torch.int64)
            )
        )
    invalid = final.clone()
    invalid[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        kas_assimilation_loss(KASTrainingOutput(invalid, deep, final.clone()))


def test_invalid_constructor_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        KASTrainingModel(init_filters=0)
    with pytest.raises(ValueError, match="dropout"):
        KASTrainingModel(dropout=1.0)
