"""Frozen Lookahead(k=5, alpha=0.5) wrapper for the AFlow stability study."""
from __future__ import annotations

from typing import Any

import torch


class LookaheadOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        k: int = 5,
        alpha: float = 0.5,
        total_steps: int,
    ) -> None:
        if k != 5 or alpha != 0.5:
            raise ValueError("frozen Lookahead protocol requires k=5 and alpha=0.5")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.total_steps = total_steps
        self.param_groups = optimizer.param_groups
        self.parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if not self.parameters or not all(parameter.is_floating_point() for parameter in self.parameters):
            raise ValueError("Lookahead expects floating trainable parameters")
        self.slow_weights = [parameter.detach().clone() for parameter in self.parameters]
        self.fast_steps = 0
        self.slow_updates = 0
        self.final_slow_loaded = False

    @torch.no_grad()
    def _sync_slow(self) -> None:
        for slow, fast in zip(self.slow_weights, self.parameters, strict=True):
            slow.add_(fast.detach() - slow, alpha=self.alpha)
            fast.copy_(slow)
        self.slow_updates += 1

    @torch.no_grad()
    def load_slow_for_evaluation(self) -> None:
        for slow, fast in zip(self.slow_weights, self.parameters, strict=True):
            fast.copy_(slow)
        self.final_slow_loaded = True

    def step(self, closure: Any = None) -> Any:
        if self.fast_steps >= self.total_steps:
            raise RuntimeError("Lookahead received more steps than frozen total")
        result = self.optimizer.step(closure=closure) if closure is not None else self.optimizer.step()
        self.fast_steps += 1
        if self.fast_steps % self.k == 0:
            self._sync_slow()
        if self.fast_steps == self.total_steps:
            self.load_slow_for_evaluation()
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def audit(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "alpha": self.alpha,
            "total_steps": self.total_steps,
            "fast_steps": self.fast_steps,
            "slow_updates": self.slow_updates,
            "expected_slow_updates": self.total_steps // self.k,
            "trailing_fast_steps": self.total_steps % self.k,
            "final_slow_loaded": self.final_slow_loaded,
            "all_counts_exact": (
                self.fast_steps == self.total_steps
                and self.slow_updates == self.total_steps // self.k
                and self.final_slow_loaded
            ),
        }
