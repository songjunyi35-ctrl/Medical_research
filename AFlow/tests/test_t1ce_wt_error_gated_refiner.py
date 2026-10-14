from __future__ import annotations

import torch

from sammed3d.t1ce_wt_error_gated_refiner import (
    ErrorGatedResidualRefiner,
    bridge_residual_refiner_state_dict,
    correction_target,
    dense_prompt_features,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import ResidualRefiner


def test_dense_prompt_features_are_bounded_and_label_free() -> None:
    coarse = torch.tensor([[[[[-3.0, 0.0, 3.0]]]]])
    coarse_feature, uncertainty, boundary = dense_prompt_features(coarse)
    assert coarse_feature.shape == coarse.shape
    assert uncertainty.shape == coarse.shape
    assert boundary.shape == coarse.shape
    assert torch.all((uncertainty >= 0) & (uncertainty <= 1))
    assert uncertainty[0, 0, 0, 0, 1] == 1
    assert torch.all(boundary >= 0)


def test_correction_target_is_dilated_xor() -> None:
    coarse = torch.full((1, 1, 7, 7, 7), -1.0)
    label = torch.zeros_like(coarse)
    label[:, :, 3, 3, 3] = 1
    target = correction_target(coarse, label, dilation_radius=1)
    assert int(target.sum()) == 27
    assert target[0, 0, 3, 3, 3] == 1
    assert target[0, 0, 0, 0, 0] == 0


def test_bridge_has_exact_eval_parity() -> None:
    torch.manual_seed(7)
    legacy = ResidualRefiner(init_filters=8, dropout=0.0).eval()
    upgraded = ErrorGatedResidualRefiner(init_filters=8, dropout=0.0).eval()
    report = bridge_residual_refiner_state_dict(upgraded, legacy.state_dict())
    image = torch.randn(1, 1, 16, 16, 16)
    coarse = torch.randn_like(image)
    with torch.inference_mode():
        expected = legacy(image, coarse)
        output = upgraded.forward_with_aux(image, coarse)
    torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(output.residual_scale, torch.ones_like(coarse), rtol=0, atol=0)
    assert report["gate_head_zero_initialized"] is True


def test_gate_can_suppress_or_amplify_residual() -> None:
    model = ErrorGatedResidualRefiner(init_filters=8, dropout=0.0).eval()
    final = model.network.conv_final[2].conv
    with torch.no_grad():
        final.weight.zero_()
        final.bias[0] = 1.0
        final.bias[1] = -2.0
    image = torch.zeros(1, 1, 16, 16, 16)
    coarse = torch.zeros_like(image)
    low = model.forward_with_aux(image, coarse)
    with torch.no_grad():
        final.bias[1] = 2.0
    high = model.forward_with_aux(image, coarse)
    assert torch.all(low.residual_scale < 1)
    assert torch.all(high.residual_scale > 1)
    assert torch.all(high.logits > low.logits)
