from __future__ import annotations

from collections import OrderedDict
import hashlib

from monai.losses import DiceCELoss
import pytest
import torch

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_dcd import (
    DB4_DEC_HI,
    DB4_DEC_LO,
    DB4_FILTER_BANK_SHA256,
    DCDDetailLoss,
    DCDTrainingModel,
    ENCODER_CHANNELS,
    ENCODER_RESOLUTIONS,
    LEVELS,
    SELECTED_BANDS,
    SUBBAND_NAMES,
    build_db4_filter_bank_3d,
    dwt3,
    encoder_feature_gradient_audit,
    idwt3,
    multilevel_detail_loss,
)
from scripts.train_t1ce_wt_plain_dcd_continuation import (
    EXPECTED_DCD_WEIGHTS,
    binary_voxel_kl,
    dcd_weight_for_global_epoch,
)


def _plain(*, center_aware: bool = False, init_filters: int = 8) -> AutomaticPromptLocator:
    torch.manual_seed(11)
    return AutomaticPromptLocator(
        center_aware=center_aware,
        init_filters=init_filters,
        dropout=0.1,
    )


def _small_model() -> DCDTrainingModel:
    return DCDTrainingModel.from_plain_model(_plain())


def _synthetic_features(
    *, requires_grad: bool, seed: int, spatial: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn((1, 1, spatial, spatial, spatial), generator=generator).requires_grad_(
            requires_grad
        )
        for _ in range(4)
    )  # type: ignore[return-value]


def test_canonical_db4_taps_filter_hash_and_band_order_are_frozen() -> None:
    assert len(DB4_DEC_LO) == len(DB4_DEC_HI) == 8
    assert SUBBAND_NAMES == (
        "LLL",
        "LLH",
        "LHL",
        "LHH",
        "HLL",
        "HLH",
        "HHL",
        "HHH",
    )
    assert SELECTED_BANDS == (1, 2, 3, 4, 5, 6)
    assert ENCODER_CHANNELS == (16, 32, 64, 128)
    assert ENCODER_RESOLUTIONS == (128, 64, 32, 16)
    bank = build_db4_filter_bank_3d()
    assert bank.shape == (8, 1, 8, 8, 8)
    assert bank.dtype == torch.float32
    assert hashlib.sha256(bank.numpy().tobytes()).hexdigest() == DB4_FILTER_BANK_SHA256

    low = torch.tensor(DB4_DEC_LO, dtype=torch.float32)
    high = torch.tensor(DB4_DEC_HI, dtype=torch.float32)
    assert torch.equal(bank[0, 0], low[:, None, None] * low[None, :, None] * low[None, None, :])
    assert torch.equal(bank[1, 0], low[:, None, None] * low[None, :, None] * high[None, None, :])
    assert torch.equal(bank[7, 0], high[:, None, None] * high[None, :, None] * high[None, None, :])


def test_full_band_float32_reconstruction_meets_frozen_gate() -> None:
    torch.manual_seed(314159)
    features = torch.randn((2, 3, 16, 16, 16), dtype=torch.float32)
    bank = build_db4_filter_bank_3d()
    coefficients = dwt3(features, bank)
    reconstruction = idwt3(coefficients, bank, features.shape[-3:])
    assert reconstruction.shape == features.shape
    assert float((features - reconstruction).abs().max()) < 1e-5


def test_three_level_shapes_and_exact_level_reconstruction() -> None:
    torch.manual_seed(17)
    current = torch.randn((1, 2, 16, 16, 16), dtype=torch.float32)
    bank = build_db4_filter_bank_3d()
    observed = []
    for _ in range(LEVELS):
        input_shape = tuple(current.shape[-3:])
        coefficients = dwt3(current, bank)
        observed.append(tuple(coefficients.shape[-3:]))
        reconstruction = idwt3(coefficients, bank, input_shape)
        assert float((current - reconstruction).abs().max()) < 1e-5
        current = coefficients[:, :, 0]
    assert observed == [(12, 12, 12), (10, 10, 10), (9, 9, 9)]


def test_identical_features_have_exact_zero_dcd_loss() -> None:
    teacher = _synthetic_features(requires_grad=False, seed=23)
    student = tuple(value.detach().clone().requires_grad_(True) for value in teacher)
    result = DCDDetailLoss()(student, teacher)
    assert result.total.item() == 0.0
    assert all(value.item() == 0.0 for value in result.stage_losses)
    assert all(
        value.item() == 0.0
        for stage in result.level_losses
        for value in stage
    )


def test_stage_sum_level_mean_and_all_feature_gradients_are_auditable() -> None:
    student = _synthetic_features(requires_grad=True, seed=29)
    teacher = _synthetic_features(requires_grad=False, seed=31)
    result = DCDDetailLoss()(student, teacher)
    assert torch.equal(result.total, torch.stack(result.stage_losses).sum())
    for stage_loss, level_losses in zip(
        result.stage_losses, result.level_losses, strict=True
    ):
        assert torch.equal(stage_loss, torch.stack(level_losses).mean())
        assert torch.isfinite(stage_loss)
        assert stage_loss.item() > 0.0
    result.total.backward()
    audit = encoder_feature_gradient_audit(student)
    assert tuple(audit) == ("stage0", "stage1", "stage2", "stage3")
    assert all(item["finite"] and item["nonzero"] for item in audit.values())
    assert all(value.grad is None for value in teacher)


def test_multilevel_loss_rejects_trainable_teacher_or_shape_mismatch() -> None:
    bank = build_db4_filter_bank_3d()
    student = torch.randn((1, 1, 8, 8, 8), requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        multilevel_detail_loss(
            student,
            torch.randn((1, 1, 8, 8, 8), requires_grad=True),
            bank,
        )
    with pytest.raises(ValueError, match="shapes"):
        multilevel_detail_loss(
            student,
            torch.randn((1, 1, 16, 16, 16)),
            bank,
        )


def test_single_encode_decode_forward_matches_plain_exactly() -> None:
    plain = _plain().eval()
    model = DCDTrainingModel.from_plain_model(plain).eval()
    image = torch.randn((1, 1, 16, 16, 16))
    with torch.inference_mode():
        plain_logits = plain(image).segmentation_logits
        output = model(image)
    assert torch.equal(output.final_logits, plain_logits)
    assert [tuple(value.shape) for value in output.encoder_features] == [
        (1, 8, 16, 16, 16),
        (1, 16, 8, 8, 8),
        (1, 32, 4, 4, 4),
        (1, 64, 2, 2, 2),
    ]


def test_encode_features_does_not_run_the_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _small_model().eval()

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("detail teacher must not run its decoder")

    monkeypatch.setattr(model.network, "decode", forbidden_decode)
    with torch.inference_mode():
        features = model.encode_features(torch.randn((1, 1, 16, 16, 16)))
    assert len(features) == 4


def test_plain_checkpoint_load_and_deploy_export_are_strict(tmp_path) -> None:
    plain = _plain().eval()
    checkpoint = {
        "architecture": {
            "class": "AutomaticPromptLocator",
            "center_aware": False,
            "init_filters": 8,
            "dropout": 0.1,
        },
        "model_state_dict": plain.state_dict(),
    }
    mapped = DCDTrainingModel.from_plain_checkpoint(checkpoint).eval()
    path = tmp_path / "plain.pth"
    torch.save(checkpoint, path)
    loaded = DCDTrainingModel.from_plain_checkpoint(path).eval()
    assert isinstance(mapped.deploy_state_dict(cpu=True), OrderedDict)
    assert tuple(mapped.deploy_state_dict()) == tuple(plain.state_dict())
    assert not any(name.startswith("dcd_loss") for name in mapped.deploy_state_dict())
    assert not [name for name, _ in mapped.named_parameters() if not name.startswith("network.")]

    image = torch.randn((1, 1, 16, 16, 16))
    with torch.inference_mode():
        expected = plain(image).segmentation_logits
        assert torch.equal(mapped(image).final_logits, expected)
        assert torch.equal(loaded(image).final_logits, expected)

    broken = dict(checkpoint)
    broken_state = OrderedDict(checkpoint["model_state_dict"])
    broken_state.pop(next(iter(broken_state)))
    broken["model_state_dict"] = broken_state
    with pytest.raises(RuntimeError, match="key mismatch"):
        DCDTrainingModel.from_plain_checkpoint(broken)


def test_deploy_roundtrip_parameter_count_rng_and_zero_weight_parity() -> None:
    model = _small_model().eval()
    image = torch.randn((1, 1, 16, 16, 16))
    rng_before = torch.get_rng_state().clone()
    assert model.deploy_max_abs_difference(image) == 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is False

    full_model = DCDTrainingModel(init_filters=16, dropout=0.1)
    assert sum(value.numel() for value in full_model.deploy_state_dict().values()) == 4_700_897

    # A zero-weight DCD branch must make an exactly identical optimizer update.
    torch.manual_seed(37)
    base_parameter = torch.nn.Parameter(torch.randn((1, 1, 8, 8, 8)))
    dcd_parameter = torch.nn.Parameter(base_parameter.detach().clone())
    teachers = _synthetic_features(requires_grad=False, seed=41)
    base_optimizer = torch.optim.SGD((base_parameter,), lr=0.01)
    dcd_optimizer = torch.optim.SGD((dcd_parameter,), lr=0.01)

    base_optimizer.zero_grad(set_to_none=True)
    base_parameter.square().mean().backward()
    base_optimizer.step()

    dcd_optimizer.zero_grad(set_to_none=True)
    student_features = (dcd_parameter,) * 4
    raw_dcd = DCDDetailLoss()(student_features, teachers).total
    (dcd_parameter.square().mean() + 0.0 * raw_dcd).backward()
    dcd_optimizer.step()
    assert torch.equal(base_parameter, dcd_parameter)


def test_frozen_schedule_and_full_model_zero_weight_adamw_parity() -> None:
    assert {
        epoch: dcd_weight_for_global_epoch(epoch) for epoch in EXPECTED_DCD_WEIGHTS
    } == EXPECTED_DCD_WEIGHTS
    assert sum(dcd_weight_for_global_epoch(epoch) for epoch in range(169, 201)) / 32 == pytest.approx(
        0.00421875
    )

    torch.manual_seed(2026)
    base = AutomaticPromptLocator(
        center_aware=False, init_filters=8, dropout=0.1
    ).train()
    dcd = DCDTrainingModel.from_plain_model(base).train()
    base_optimizer = torch.optim.AdamW(
        base.parameters(), lr=2e-4, weight_decay=1e-5,
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
    )
    dcd_optimizer = torch.optim.AdamW(
        dcd.parameters(), lr=2e-4, weight_decay=1e-5,
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
    )
    image = torch.randn((1, 1, 16, 16, 16))
    label = (torch.rand((1, 1, 16, 16, 16)) > 0.8).float()
    frozen_teacher_logits = torch.randn_like(label)
    loss_function = DiceCELoss(
        sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.1
    )

    base_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(999)
    base_logits = base(image).segmentation_logits
    base_loss = loss_function(base_logits, label) + 0.3 * binary_voxel_kl(
        base_logits.float(), frozen_teacher_logits, temperature=2.0
    )
    base_loss.backward()
    torch.nn.utils.clip_grad_norm_(base.parameters(), max_norm=5.0)
    base_optimizer.step()

    dcd_optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(999)
    dcd_output = dcd(image)
    # Reuse the 16^3 first-stage feature four times so the parity test remains
    # small while still attaching a real DCD graph to the deploy network.
    parity_features = (dcd_output.encoder_features[0],) * 4
    teacher_features = tuple(
        feature.detach().clone() + 0.01 for feature in parity_features
    )
    raw_dcd = dcd.detail_loss(parity_features, teacher_features).total
    dcd_loss = loss_function(dcd_output.final_logits, label) + 0.3 * binary_voxel_kl(
        dcd_output.final_logits.float(), frozen_teacher_logits, temperature=2.0
    ) + 0.0 * raw_dcd
    dcd_loss.backward()
    torch.nn.utils.clip_grad_norm_(dcd.parameters(), max_norm=5.0)
    dcd_optimizer.step()

    deploy_state = dcd.deploy_state_dict()
    for name, value in base.state_dict().items():
        assert torch.equal(value, deploy_state[name]), name
