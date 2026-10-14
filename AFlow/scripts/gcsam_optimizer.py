#!/usr/bin/env python3
"""Paper-form Gradient-Centralized Sharpness-Aware Minimization helpers.

This is the preregistered 3D adaptation for the 2026-08-31 Full972 experiment.
It intentionally uses the norm of the centralized gradient, matching Eq. (9) of
Hassan et al. (2025), rather than the ordering in their public example script.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def centralize_gradient_(gradient: torch.Tensor) -> torch.Tensor:
    """Subtract each output unit's mean for tensors with more than one dim."""
    if gradient.ndim > 1:
        gradient.sub_(gradient.mean(dim=tuple(range(1, gradient.ndim)), keepdim=True))
    return gradient


def _materialize(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return list(parameters)


def centralized_global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Centralize gradients in-place and return their finite global L2 norm."""
    params = _materialize(parameters)
    gradients: list[torch.Tensor] = []
    for parameter in params:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("non-finite gradient before GCSAM perturbation")
        centralize_gradient_(parameter.grad)
        gradients.append(parameter.grad)
    if not gradients:
        raise RuntimeError("GCSAM requires at least one gradient")
    device = gradients[0].device
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(gradient.detach()).to(device) for gradient in gradients
    ]))
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise RuntimeError("GCSAM centralized gradient norm must be finite and positive")
    return norm


class PaperGCSAM:
    """GCSAM perturbation wrapper around an ordinary PyTorch base optimizer.

    The wrapper stores exact pre-perturbation parameter copies so restoration is
    bit-exact before the base optimizer consumes the second-pass gradients.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        base_optimizer: torch.optim.Optimizer,
        *,
        rho: float = 0.05,
    ) -> None:
        if rho < 0.0:
            raise ValueError("GCSAM requires rho >= 0")
        self.parameters = _materialize(parameters)
        if not self.parameters:
            raise ValueError("GCSAM received no parameters")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self._originals: dict[int, torch.Tensor] | None = None
        self._perturbations: dict[int, torch.Tensor] | None = None
        self.last_first_step: dict[str, float | int] | None = None

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.base_optimizer.param_groups

    @property
    def perturbation_active(self) -> bool:
        return self._originals is not None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self, *, zero_grad: bool = True) -> dict[str, float | int]:
        if self._originals is not None:
            raise RuntimeError("GCSAM perturbation is already active")
        norm = centralized_global_grad_norm(self.parameters)
        scale = self.rho / norm
        originals: dict[int, torch.Tensor] = {}
        perturbations: dict[int, torch.Tensor] = {}
        perturbed = 0
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            original = parameter.detach().clone(memory_format=torch.preserve_format)
            perturbation = parameter.grad.detach() * scale.to(parameter.device)
            if not bool(torch.isfinite(perturbation).all()):
                raise FloatingPointError("non-finite GCSAM perturbation")
            originals[id(parameter)] = original
            perturbations[id(parameter)] = perturbation
            parameter.add_(perturbation)
            perturbed += 1
        if perturbed == 0:
            raise RuntimeError("GCSAM perturbed no parameters")
        actual_norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(value) for value in perturbations.values()
        ]))
        diagnostics: dict[str, float | int] = {
            "centralized_grad_norm": float(norm),
            "perturbation_norm": float(actual_norm),
            "perturbed_parameter_tensors": perturbed,
        }
        self._originals = originals
        self._perturbations = perturbations
        self.last_first_step = diagnostics
        if zero_grad:
            self.zero_grad(set_to_none=True)
        return diagnostics

    @torch.no_grad()
    def restore(self) -> None:
        if self._originals is None:
            raise RuntimeError("no active GCSAM perturbation to restore")
        for parameter in self.parameters:
            original = self._originals.get(id(parameter))
            if original is not None:
                parameter.copy_(original)
        self._originals = None
        self._perturbations = None

    @torch.no_grad()
    def second_step(self, *, zero_grad: bool = True) -> None:
        if self._originals is None:
            raise RuntimeError("GCSAM second_step called without first_step")
        for parameter in self.parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                self.restore()
                raise FloatingPointError("non-finite second-pass GCSAM gradient")
        self.restore()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        if self._originals is not None:
            raise RuntimeError("cannot serialize optimizer during active perturbation")
        return {
            "schema": "aflow.paper_gcsam_optimizer",
            "schema_version": 1,
            "rho": self.rho,
            "base_optimizer": self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if float(state_dict["rho"]) != self.rho:
            raise ValueError("GCSAM rho mismatch while loading optimizer")
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
