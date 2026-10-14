from __future__ import annotations

from collections import OrderedDict
import copy

import pytest
import torch

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_dmrfb_adapter import (
    DMRFBAdapterLocator,
    FORMAL_DEPLOY_PARAMETERS,
)
from scripts.make_t1ce_wt_dmrfb_weight_interpolation import (
    CHECKPOINT_SCHEMA,
    EXPECTED_STATE_KEYS,
    FIXED_INPUT_SHAPE,
    INTERIOR_ALPHAS,
    alpha_token,
    build_alpha0_model,
    fixed_input,
    interpolate_state_dict,
    load_alpha1_model,
    state_sha256,
    validate_state_geometry,
)


def _plain(*, init_filters: int = 8) -> AutomaticPromptLocator:
    torch.manual_seed(23)
    return AutomaticPromptLocator(
        center_aware=False, init_filters=init_filters, dropout=0.1
    )


def _plain_checkpoint(plain: AutomaticPromptLocator) -> dict:
    return {
        "architecture": {
            "class": "AutomaticPromptLocator",
            "center_aware": False,
            "init_filters": plain.init_filters,
            "dropout": plain.dropout,
        },
        "model_state_dict": plain.state_dict(),
    }


def _small_endpoints() -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    endpoint0_model = DMRFBAdapterLocator.from_plain_model(
        _plain(), arm="dmrfb_dilation2"
    ).eval()
    endpoint1_model = copy.deepcopy(endpoint0_model).eval()
    with torch.no_grad():
        for index, parameter in enumerate(endpoint1_model.parameters()):
            parameter.add_(0.001 * (index + 1))
    return (
        endpoint0_model.deploy_state_dict(cpu=True),
        endpoint1_model.deploy_state_dict(cpu=True),
    )


def test_frozen_alpha_grid_tokens_and_expected_geometry() -> None:
    assert INTERIOR_ALPHAS == (0.25, 0.5, 0.75)
    assert [alpha_token(value) for value in INTERIOR_ALPHAS] == ["025", "050", "075"]
    assert EXPECTED_STATE_KEYS == 90
    assert CHECKPOINT_SCHEMA == "aflow.sammed3d.t1ce_wt_dmrfb_weight_interpolation_checkpoint"


def test_float_interpolation_uses_float32_then_cast_and_nonfloat_copies() -> None:
    endpoint0 = OrderedDict(
        float16=torch.tensor([0.1, 0.4], dtype=torch.float16),
        float32=torch.tensor([-1.0, 3.0], dtype=torch.float32),
        integer=torch.tensor([2, 7], dtype=torch.int64),
    )
    endpoint1 = OrderedDict(
        float16=torch.tensor([0.9, -0.2], dtype=torch.float16),
        float32=torch.tensor([5.0, 1.0], dtype=torch.float32),
        integer=torch.tensor([2, 7], dtype=torch.int64),
    )
    observed = interpolate_state_dict(
        endpoint0, endpoint1, alpha=0.25, expected_keys=None
    )
    expected16 = (
        0.75 * endpoint0["float16"].float()
        + 0.25 * endpoint1["float16"].float()
    ).to(torch.float16)
    expected32 = 0.75 * endpoint0["float32"] + 0.25 * endpoint1["float32"]
    assert torch.equal(observed["float16"], expected16)
    assert torch.equal(observed["float32"], expected32)
    assert torch.equal(observed["integer"], endpoint0["integer"])
    assert observed["integer"].data_ptr() != endpoint0["integer"].data_ptr()


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_unregistered_alpha_is_rejected(alpha: float) -> None:
    left = OrderedDict(value=torch.ones(1))
    right = OrderedDict(value=torch.zeros(1))
    with pytest.raises(ValueError, match="alpha"):
        interpolate_state_dict(left, right, alpha=alpha, expected_keys=None)


def test_geometry_rejects_key_shape_dtype_and_nonfloat_value_mismatch() -> None:
    base = OrderedDict(a=torch.ones(2), b=torch.tensor([1], dtype=torch.int64))
    with pytest.raises(ValueError, match="keys"):
        validate_state_geometry(base, OrderedDict(a=torch.ones(2)), expected_keys=None)
    with pytest.raises(ValueError, match="shape"):
        validate_state_geometry(
            base,
            OrderedDict(a=torch.ones(3), b=torch.tensor([1], dtype=torch.int64)),
            expected_keys=None,
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_state_geometry(
            base,
            OrderedDict(a=torch.ones(2, dtype=torch.float64), b=torch.tensor([1], dtype=torch.int64)),
            expected_keys=None,
        )
    with pytest.raises(ValueError, match="Non-floating|non-floating"):
        validate_state_geometry(
            base,
            OrderedDict(a=torch.ones(2), b=torch.tensor([2], dtype=torch.int64)),
            expected_keys=None,
        )


def test_real_dmrfb_state_geometry_has_90_identical_keys_and_three_reloadable_interiors(
    tmp_path,
) -> None:
    endpoint0, endpoint1 = _small_endpoints()
    geometry = validate_state_geometry(endpoint0, endpoint1)
    assert geometry["state_keys"] == EXPECTED_STATE_KEYS
    assert geometry["total_values"] == sum(value.numel() for value in endpoint0.values())
    assert geometry["endpoint0_state_sha256"] == state_sha256(endpoint0)
    assert geometry["endpoint1_state_sha256"] == state_sha256(endpoint1)
    assert geometry["endpoint0_state_sha256"] != geometry["endpoint1_state_sha256"]

    image = torch.randn((1, 1, 16, 16, 16))
    for alpha in INTERIOR_ALPHAS:
        state = interpolate_state_dict(endpoint0, endpoint1, alpha=alpha)
        path = tmp_path / f"alpha{alpha_token(alpha)}.pth"
        torch.save({"model_state_dict": state}, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        model = DMRFBAdapterLocator(
            arm="dmrfb_dilation2", init_filters=8, dropout=0.1
        ).eval()
        model.load_state_dict(loaded["model_state_dict"], strict=True)
        with torch.inference_mode():
            logits = model(image).final_logits
        assert torch.isfinite(logits).all()


def test_alpha0_builder_has_zero_adapter_and_exact_plain_geometry() -> None:
    plain = _plain(init_filters=16)
    checkpoint = _plain_checkpoint(plain)
    alpha0 = build_alpha0_model(checkpoint)
    assert torch.count_nonzero(alpha0.adapter.projection.weight).item() == 0
    assert alpha0.adapter.projection.bias is not None
    assert torch.count_nonzero(alpha0.adapter.projection.bias).item() == 0
    assert sum(parameter.numel() for parameter in alpha0.parameters()) == FORMAL_DEPLOY_PARAMETERS
    for name, value in plain.network.state_dict().items():
        assert torch.equal(value, alpha0.network.state_dict()[name]), name


def test_alpha1_loader_requires_frozen_schema_epoch_arm_access_and_strict_state() -> None:
    model = DMRFBAdapterLocator(
        arm="dmrfb_dilation2", init_filters=16, dropout=0.1
    ).eval()
    checkpoint = {
        "schema": "aflow.sammed3d.t1ce_wt_dmrfb_adapter_checkpoint",
        "arm": "dmrfb_dilation2",
        "global_epoch": 216,
        "model_state_dict": model.deploy_state_dict(cpu=True),
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    loaded = load_alpha1_model(checkpoint)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    broken = dict(checkpoint)
    broken["global_epoch"] = 208
    with pytest.raises(ValueError, match="epoch216"):
        load_alpha1_model(broken)
    broken = dict(checkpoint)
    broken["test_access"] = True
    with pytest.raises(ValueError, match="test_access"):
        load_alpha1_model(broken)


def test_fixed_input_is_rng_isolated_deterministic_and_finite() -> None:
    torch.manual_seed(303)
    before = torch.get_rng_state().clone()
    first = fixed_input()
    second = fixed_input()
    assert tuple(first.shape) == FIXED_INPUT_SHAPE
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.equal(torch.get_rng_state(), before)
