from __future__ import annotations

import torch

from sammed3d.t1ce_wt_safe_adapter import FrozenBaseSafeAdapter
from scripts.train_sammed3d_t1ce_wt_residual_refiner import ResidualRefiner


def make_model(mode: str) -> FrozenBaseSafeAdapter:
    base = ResidualRefiner(init_filters=8, dropout=0.0)
    return FrozenBaseSafeAdapter(
        base,
        mode=mode,
        init_filters=8,
        dropout=0.0,
    )


def test_zero_initialized_adapter_exactly_reproduces_frozen_base() -> None:
    torch.manual_seed(11)
    image = torch.randn(1, 1, 16, 16, 16)
    coarse = torch.randn_like(image)
    for mode in ("plain", "gated"):
        model = make_model(mode).eval()
        with torch.inference_mode():
            expected = model.base(image, coarse)
            output = model.forward_with_aux(image, coarse)
        torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(output.residual, torch.zeros_like(coarse), rtol=0, atol=0)
        torch.testing.assert_close(output.residual_scale, torch.ones_like(coarse), rtol=0, atol=0)


def test_base_remains_frozen_and_in_eval_mode() -> None:
    model = make_model("gated")
    model.train()
    assert model.training is True
    assert model.adapter.training is True
    assert model.base.training is False
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.requires_grad for parameter in model.adapter.parameters())


def test_plain_and_gated_share_capacity_but_gate_changes_scale() -> None:
    torch.manual_seed(19)
    plain = make_model("plain").eval()
    torch.manual_seed(19)
    gated = make_model("gated").eval()
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in gated.parameters())
    with torch.no_grad():
        plain.adapter.conv_final[2].conv.bias[:] = torch.tensor([1.0, 2.0])
        gated.adapter.conv_final[2].conv.bias[:] = torch.tensor([1.0, 2.0])
    image = torch.zeros(1, 1, 16, 16, 16)
    coarse = torch.zeros_like(image)
    with torch.inference_mode():
        plain_output = plain.forward_with_aux(image, coarse)
        gated_output = gated.forward_with_aux(image, coarse)
    assert torch.all(plain_output.residual_scale == 1)
    assert torch.all(gated_output.residual_scale > 1)
    assert torch.all(gated_output.logits > plain_output.logits)


def test_invalid_mode_is_rejected() -> None:
    base = ResidualRefiner(init_filters=8, dropout=0.0)
    try:
        FrozenBaseSafeAdapter(base, mode="unknown", init_filters=8, dropout=0.0)
    except ValueError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("invalid mode was accepted")
