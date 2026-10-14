from __future__ import annotations

import copy

import pytest
import torch

from scripts.gcsam_optimizer import PaperGCSAM, centralize_gradient_, centralized_global_grad_norm


def test_centralization_contract() -> None:
    multi = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    one_d = torch.tensor([1.0, 2.0, 4.0])
    expected_one_d = one_d.clone()
    centralize_gradient_(multi)
    centralize_gradient_(one_d)
    assert torch.allclose(multi.mean(dim=(1, 2)), torch.zeros(2), atol=1e-7, rtol=0)
    assert torch.equal(one_d, expected_one_d)


def test_first_step_uses_centralized_norm_and_restores_exactly() -> None:
    weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [4.0, 8.0]]))
    bias = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    weight.grad = torch.tensor([[1.0, 3.0], [2.0, 8.0]])
    bias.grad = torch.tensor([2.0, -1.0])
    base = torch.optim.AdamW([weight, bias], lr=1e-3, weight_decay=0.0)
    optimizer = PaperGCSAM([weight, bias], base, rho=0.05)
    original_weight = weight.detach().clone()
    original_bias = bias.detach().clone()
    diagnostics = optimizer.first_step(zero_grad=True)
    assert diagnostics["perturbation_norm"] == pytest.approx(0.05, abs=2e-7)
    optimizer.restore()
    assert torch.equal(weight, original_weight)
    assert torch.equal(bias, original_bias)


def test_second_step_matches_adamw_reference_at_second_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    base = torch.optim.AdamW([parameter], lr=2e-3, weight_decay=1e-5)
    reference_optimizer = torch.optim.AdamW([reference], lr=2e-3, weight_decay=1e-5)
    optimizer = PaperGCSAM([parameter], base, rho=0.05)
    parameter.grad = torch.tensor([[1.0, 4.0], [-2.0, 3.0]])
    optimizer.first_step(zero_grad=True)
    second_gradient = torch.tensor([[0.25, -0.5], [1.5, 2.0]])
    parameter.grad = second_gradient.clone()
    reference.grad = second_gradient.clone()
    reference_optimizer.step()
    optimizer.second_step(zero_grad=True)
    assert torch.equal(parameter, reference)


def test_state_roundtrip_and_active_serialization_guard() -> None:
    model = torch.nn.Linear(3, 2)
    base = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = PaperGCSAM(model.parameters(), base, rho=0.05)
    inputs = torch.ones(2, 3)
    model(inputs).sum().backward()
    optimizer.first_step(zero_grad=True)
    with pytest.raises(RuntimeError, match="active perturbation"):
        optimizer.state_dict()
    optimizer.restore()
    state = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(state)
    assert state["rho"] == 0.05


def test_nonfinite_and_zero_gradient_fail_safely() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="positive"):
        centralized_global_grad_norm([parameter])
    parameter.grad = torch.full_like(parameter, float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        centralized_global_grad_norm([parameter])


def test_nested_first_step_is_rejected() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    base = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer = PaperGCSAM(model.parameters(), base, rho=0.05)
    model(torch.tensor([[1.0, 2.0]])).sum().backward()
    optimizer.first_step(zero_grad=True)
    with pytest.raises(RuntimeError, match="already active"):
        optimizer.first_step()
    optimizer.restore()


def test_rho_zero_matches_plain_adamw_update() -> None:
    torch.manual_seed(7)
    control = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GELU(), torch.nn.Linear(3, 1))
    candidate = copy.deepcopy(control)
    inputs = torch.randn(5, 4)
    targets = torch.randn(5, 1)
    control_optimizer = torch.optim.AdamW(control.parameters(), lr=2e-3, weight_decay=1e-5)
    candidate_base = torch.optim.AdamW(candidate.parameters(), lr=2e-3, weight_decay=1e-5)
    candidate_optimizer = PaperGCSAM(candidate.parameters(), candidate_base, rho=0.0)

    control_optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(control(inputs), targets).backward()
    control_optimizer.step()

    candidate_optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(candidate(inputs), targets).backward()
    diagnostics = candidate_optimizer.first_step(zero_grad=True)
    assert diagnostics["perturbation_norm"] == 0.0
    torch.nn.functional.mse_loss(candidate(inputs), targets).backward()
    candidate_optimizer.second_step(zero_grad=True)
    for control_parameter, candidate_parameter in zip(control.parameters(), candidate.parameters()):
        assert torch.equal(control_parameter, candidate_parameter)
