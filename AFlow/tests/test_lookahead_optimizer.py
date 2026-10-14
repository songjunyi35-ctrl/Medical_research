from __future__ import annotations

import pytest
import torch

from scripts.lookahead_optimizer import LookaheadOptimizer


def _optimizer(value: float = 0.0, total_steps: int = 6) -> tuple[torch.nn.Parameter, LookaheadOptimizer]:
    parameter = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
    inner = torch.optim.SGD([parameter], lr=1.0)
    return parameter, LookaheadOptimizer(inner, total_steps=total_steps)


def test_lookahead_default_fifth_step_interpolates_and_syncs() -> None:
    parameter, optimizer = _optimizer()
    for _ in range(4):
        parameter.grad = torch.tensor([-1.0])
        optimizer.step()
    assert torch.equal(parameter.detach(), torch.tensor([4.0]))
    parameter.grad = torch.tensor([-1.0])
    optimizer.step()
    assert torch.equal(parameter.detach(), torch.tensor([2.5]))
    assert optimizer.audit()["slow_updates"] == 1


def test_lookahead_final_step_loads_latest_slow_without_extra_update() -> None:
    parameter, optimizer = _optimizer(total_steps=6)
    for _ in range(6):
        parameter.grad = torch.tensor([-1.0])
        optimizer.step()
    assert torch.equal(parameter.detach(), torch.tensor([2.5]))
    assert optimizer.audit() == {
        "k": 5,
        "alpha": 0.5,
        "total_steps": 6,
        "fast_steps": 6,
        "slow_updates": 1,
        "expected_slow_updates": 1,
        "trailing_fast_steps": 1,
        "final_slow_loaded": True,
        "all_counts_exact": True,
    }


def test_lookahead_does_not_consume_torch_rng() -> None:
    parameter, optimizer = _optimizer(total_steps=1)
    state = torch.random.get_rng_state().clone()
    parameter.grad = torch.tensor([-1.0])
    optimizer.step()
    assert torch.equal(state, torch.random.get_rng_state())


def test_lookahead_rejects_nonfrozen_hyperparameters_and_extra_steps() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    inner = torch.optim.SGD([parameter], lr=1.0)
    with pytest.raises(ValueError, match="k=5"):
        LookaheadOptimizer(inner, k=4, total_steps=1)
    _, optimizer = _optimizer(total_steps=1)
    optimizer.step()
    with pytest.raises(RuntimeError, match="more steps"):
        optimizer.step()
