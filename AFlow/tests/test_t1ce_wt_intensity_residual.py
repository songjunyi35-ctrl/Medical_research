from __future__ import annotations

import torch

from sammed3d.t1ce_wt_intensity_residual import (
    IntensityUncertaintyResidualLocator,
    intensity_features,
)


def test_intensity_features_shape_finite_and_nontrivial() -> None:
    image = torch.linspace(-2, 2, 9**3).reshape(1, 1, 9, 9, 9)
    features = intensity_features(image, kernel_size=3)
    assert features.shape == (1, 3, 9, 9, 9)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features[:, 1:]) > 0


def test_control_and_intensity_start_at_identical_base_output() -> None:
    torch.manual_seed(20260821)
    control = IntensityUncertaintyResidualLocator(
        use_intensity=False, init_filters=8, residual_channels=4, intensity_kernel=3
    ).eval()
    torch.manual_seed(20260821)
    intensity = IntensityUncertaintyResidualLocator(
        use_intensity=True, init_filters=8, residual_channels=4, intensity_kernel=3
    ).eval()
    for left, right in zip(control.state_dict().values(), intensity.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    image = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        control_output = control(image)
        intensity_output = intensity(image)
    torch.testing.assert_close(
        control_output.segmentation_logits, control_output.base_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        intensity_output.segmentation_logits, intensity_output.base_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        control_output.segmentation_logits,
        intensity_output.segmentation_logits,
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(control_output.explicit_intensity_features) == 0
    assert torch.count_nonzero(intensity_output.explicit_intensity_features) > 0


def test_uncertainty_gate_is_bounded_and_residual_head_trains() -> None:
    torch.manual_seed(7)
    model = IntensityUncertaintyResidualLocator(
        use_intensity=True, init_filters=8, residual_channels=4, intensity_kernel=3
    )
    image = torch.randn(1, 1, 16, 16, 16)
    output = model(image)
    assert torch.all(output.uncertainty >= 0)
    assert torch.all(output.uncertainty <= 1)
    loss = output.segmentation_logits.square().mean()
    loss.backward()
    final = model.residual_head[-1]
    assert final.weight.grad is not None
    assert torch.count_nonzero(final.weight.grad) > 0

