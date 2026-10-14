from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.t1ce_wt_dmrfb_adapter import (
    ADAPTER_CONSTRUCTION_SEED,
    DMRFBAdapterLocator,
    DMRFBBottleneckAdapter,
    DMRFB_ARMS,
    FORMAL_ADAPTER_PARAMETERS,
    FORMAL_DEPLOY_PARAMETERS,
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


def test_frozen_branch_metadata_parameter_counts_and_zero_projection() -> None:
    assert DMRFB_ARMS == ("same_rf_control", "dmrfb_dilation2")
    assert ADAPTER_CONSTRUCTION_SEED == 20260901
    control = DMRFBBottleneckAdapter(channels=128, arm="same_rf_control")
    candidate = DMRFBBottleneckAdapter(channels=128, arm="dmrfb_dilation2")
    assert [branch.kernel_size for branch in control.branches] == [
        (1, 1, 1), (3, 3, 3), (3, 3, 3)
    ]
    assert [branch.dilation for branch in control.branches] == [
        (1, 1, 1), (1, 1, 1), (1, 1, 1)
    ]
    assert [branch.dilation for branch in candidate.branches] == [
        (1, 1, 1), (1, 1, 1), (2, 2, 2)
    ]
    assert control.branches[2].padding == (1, 1, 1)
    assert candidate.branches[2].padding == (2, 2, 2)
    assert control.norm.num_groups == 128
    assert control.norm.num_channels == 384
    assert control.norm.eps == 1e-5
    assert isinstance(control.activation, torch.nn.GELU)
    assert control.activation.approximate == "none"
    assert sum(parameter.numel() for parameter in control.parameters()) == FORMAL_ADAPTER_PARAMETERS
    assert sum(parameter.numel() for parameter in candidate.parameters()) == FORMAL_ADAPTER_PARAMETERS
    assert torch.count_nonzero(control.projection.weight).item() == 0
    assert control.projection.bias is not None
    assert torch.count_nonzero(control.projection.bias).item() == 0


def test_two_arms_have_identical_initial_state_tensors_under_any_caller_seed() -> None:
    torch.manual_seed(101)
    control = DMRFBBottleneckAdapter(channels=128, arm="same_rf_control")
    torch.manual_seed(909)
    candidate = DMRFBBottleneckAdapter(channels=128, arm="dmrfb_dilation2")
    assert tuple(control.state_dict()) == tuple(candidate.state_dict())
    for name, value in control.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name


def test_adapter_is_exact_identity_at_initialization_for_both_arms() -> None:
    torch.manual_seed(17)
    features = torch.randn((2, 128, 4, 5, 6))
    for arm in DMRFB_ARMS:
        adapter = DMRFBBottleneckAdapter(channels=128, arm=arm).eval()
        with torch.inference_mode():
            output = adapter(features)
        assert torch.equal(output, features)


def test_plain_control_candidate_final_logits_are_exact_before_training() -> None:
    plain = _plain().eval()
    control = DMRFBAdapterLocator.from_plain_model(
        plain, arm="same_rf_control"
    ).eval()
    candidate = DMRFBAdapterLocator.from_plain_model(
        plain, arm="dmrfb_dilation2"
    ).eval()
    plain_network_state = plain.network.state_dict()
    for name, value in plain_network_state.items():
        assert torch.equal(value, control.network.state_dict()[name]), name
        assert torch.equal(value, candidate.network.state_dict()[name]), name
    image = torch.randn((1, 1, 16, 16, 16))
    with torch.inference_mode():
        expected = plain(image).segmentation_logits
        left = control(image)
        right = candidate(image)
    assert torch.equal(left.final_logits, expected)
    assert torch.equal(left.segmentation_logits, expected)
    assert left.center_logits is None
    assert torch.equal(right.final_logits, expected)
    assert right.center_logits is None


def test_adapter_constructor_does_not_perturb_paired_caller_rng() -> None:
    torch.manual_seed(303)
    AutomaticPromptLocator(center_aware=False, init_filters=8, dropout=0.1)
    expected = torch.get_rng_state().clone()
    torch.manual_seed(303)
    DMRFBAdapterLocator(
        arm="same_rf_control", init_filters=8, dropout=0.1
    )
    assert torch.equal(torch.get_rng_state(), expected)


def test_step1_projection_then_step2_all_adapter_gradients_are_nonzero() -> None:
    torch.manual_seed(41)
    adapter = DMRFBBottleneckAdapter(
        channels=128, arm="dmrfb_dilation2"
    ).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=1e-3,
        weight_decay=0.08,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )

    first = torch.randn((2, 128, 4, 4, 4))
    first_target = torch.randn_like(first)
    optimizer.zero_grad(set_to_none=True)
    first_loss = F.mse_loss(adapter(first), first_target)
    first_loss.backward()
    projection_gradient_norm = 0.0
    for name, parameter in adapter.projection.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        projection_gradient_norm += float(parameter.grad.float().square().sum())
    assert projection_gradient_norm > 0
    for module in (adapter.branches, adapter.norm):
        for parameter in module.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
                assert parameter.grad.abs().sum().item() == 0.0
    optimizer.step()

    second = torch.randn((2, 128, 4, 4, 4)) + 0.25
    second_target = torch.randn_like(second) - 0.5
    optimizer.zero_grad(set_to_none=True)
    second_loss = F.mse_loss(adapter(second), second_target)
    second_loss.backward()
    for index, branch in enumerate(adapter.branches):
        gradient = branch.weight.grad
        assert gradient is not None, f"branch{index}"
        assert torch.isfinite(gradient).all(), f"branch{index}"
        assert gradient.float().norm().item() > 0, f"branch{index}"
    for module_name, module in (
        ("norm", adapter.norm),
        ("projection", adapter.projection),
    ):
        module_norm = 0.0
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"{module_name}.{name}"
            assert torch.isfinite(parameter.grad).all(), f"{module_name}.{name}"
            module_norm += float(parameter.grad.float().square().sum())
        assert module_norm > 0, module_name
    optimizer.step()
    for name, parameter in adapter.named_parameters():
        assert torch.isfinite(parameter).all(), name
        state = optimizer.state[parameter]
        for state_name, tensor in state.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all(), f"{name}.{state_name}"


def test_base_and_adapter_parameter_iterators_are_disjoint_and_complete() -> None:
    model = DMRFBAdapterLocator(
        arm="same_rf_control", init_filters=16, dropout=0.1
    )
    base_named = list(model.base_named_parameters())
    adapter_named = list(model.adapter_named_parameters())
    base = list(model.base_parameters())
    adapter = list(model.adapter_parameters())
    assert [parameter for _, parameter in base_named] == base
    assert [parameter for _, parameter in adapter_named] == adapter
    assert not ({id(value) for value in base} & {id(value) for value in adapter})
    assert {name for name, _ in base_named + adapter_named} == {
        name for name, _ in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        (
            {"params": base, "lr": 1e-5, "weight_decay": 1e-5},
            {"params": adapter, "lr": 1e-3, "weight_decay": 0.08},
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len({id(value) for value in optimizer_parameters}) == len(
        optimizer_parameters
    )
    assert {id(value) for value in optimizer_parameters} == {
        id(value) for value in model.parameters()
    }
    assert sum(value.numel() for value in base) == 4_700_897
    assert sum(value.numel() for value in adapter) == 57_088


def test_formal_deploy_parameter_count_and_retained_adapter_state() -> None:
    model = DMRFBAdapterLocator(
        arm="dmrfb_dilation2", init_filters=16, dropout=0.1
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == FORMAL_DEPLOY_PARAMETERS
    state = model.deploy_state_dict(cpu=True)
    assert isinstance(state, OrderedDict)
    assert any(name.startswith("adapter.") for name in state)
    assert sum(value.numel() for value in state.values()) == FORMAL_DEPLOY_PARAMETERS


def test_plain_checkpoint_mapping_path_and_broken_state_are_strict(tmp_path) -> None:
    plain = _plain().eval()
    checkpoint = _checkpoint(plain)
    mapped = DMRFBAdapterLocator.from_plain_checkpoint(
        checkpoint, arm="same_rf_control"
    ).eval()
    path = tmp_path / "plain.pth"
    torch.save(checkpoint, path)
    loaded = DMRFBAdapterLocator.from_plain_checkpoint(
        path, arm="same_rf_control"
    ).eval()
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
        DMRFBAdapterLocator.from_plain_checkpoint(
            broken, arm="same_rf_control"
        )


def test_deploy_roundtrip_is_exact_retains_adapter_and_restores_rng() -> None:
    model = DMRFBAdapterLocator(
        arm="dmrfb_dilation2", init_filters=8, dropout=0.1
    ).eval()
    with torch.no_grad():
        model.adapter.projection.weight.normal_(mean=0.0, std=1e-3)
        assert model.adapter.projection.bias is not None
        model.adapter.projection.bias.fill_(1e-4)
    image = torch.randn((1, 1, 16, 16, 16))
    deploy = model.to_deploy_model().eval()
    with torch.inference_mode():
        expected = model(image).final_logits
        observed = deploy(image).final_logits
    assert torch.equal(expected, observed)
    rng_before = torch.get_rng_state().clone()
    assert model.deploy_max_abs_difference(image) == 0.0
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is False


@pytest.mark.parametrize(
    "arm,error",
    [("unknown", ValueError)],
)
def test_invalid_adapter_contract_is_rejected(
    arm: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        DMRFBBottleneckAdapter(channels=128, arm=arm)
    with pytest.raises(ValueError, match="shape"):
        DMRFBBottleneckAdapter(channels=128, arm="same_rf_control")(
            torch.randn((1, 64, 4, 4, 4))
        )
    with pytest.raises(ValueError, match="center_aware"):
        DMRFBAdapterLocator.from_plain_model(
            _plain(center_aware=True), arm="same_rf_control"
        )
