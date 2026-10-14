"""Dedicated-RNG retained-statistics Gamma augmentation for 3D T1ce."""
from __future__ import annotations

from typing import Any

import torch


class RetainedStatsGamma:
    def __init__(self, device: torch.device, *, seed: int, probability: float = 0.3) -> None:
        if probability != 0.3:
            raise ValueError("frozen Gamma probability must be 0.3")
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.seed = seed
        self.probability = probability
        self.total_samples = 0
        self.applied_samples = 0
        self.gamma_min = float("inf")
        self.gamma_max = float("-inf")
        self.max_mean_abs_error = 0.0
        self.max_std_abs_error = 0.0

    def _sample_gamma(self, count: int) -> torch.Tensor:
        side = torch.rand(count, device=self.device, generator=self.generator)
        unit = torch.rand(count, device=self.device, generator=self.generator)
        eps = torch.finfo(torch.float32).eps
        unit = unit.clamp(min=eps, max=1.0 - eps)
        below = 0.7 + 0.3 * unit
        above = 1.0 + 0.5 * unit
        return torch.where(side < 0.5, below, above)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 5 or image.shape[1] != 1 or label.shape != image.shape:
            raise ValueError("expected matching Bx1xDxHxW image/label")
        if image.device != self.device:
            raise ValueError("Gamma generator and image device differ")
        batch = image.shape[0]
        apply_mask = torch.rand(batch, device=self.device, generator=self.generator) < self.probability
        selected = apply_mask.nonzero(as_tuple=False).flatten()
        result = image.clone()
        gammas = self._sample_gamma(int(selected.numel()))
        for slot, gamma in zip(selected.tolist(), gammas, strict=True):
            value = image[slot]
            low = value.amin()
            high = value.amax()
            span = high - low
            if not bool(torch.isfinite(span)) or float(span) <= 1e-6:
                raise ValueError("Gamma requires finite nonconstant image")
            old_mean = value.mean()
            old_std = value.std(unbiased=False)
            normalized = ((value - low) / span).clamp(0.0, 1.0)
            transformed = normalized.pow(gamma) * span + low
            new_mean = transformed.mean()
            new_std = transformed.std(unbiased=False)
            if not bool(torch.isfinite(new_std)) or float(new_std) <= 1e-6:
                raise ValueError("Gamma produced degenerate statistics")
            transformed = (transformed - new_mean) / new_std * old_std + old_mean
            result[slot] = transformed
            mean_error = abs(float(transformed.mean() - old_mean))
            std_error = abs(float(transformed.std(unbiased=False) - old_std))
            self.max_mean_abs_error = max(self.max_mean_abs_error, mean_error)
            self.max_std_abs_error = max(self.max_std_abs_error, std_error)
        self.total_samples += batch
        self.applied_samples += int(selected.numel())
        if gammas.numel():
            self.gamma_min = min(self.gamma_min, float(gammas.min()))
            self.gamma_max = max(self.gamma_max, float(gammas.max()))
        return result, label

    def audit(self) -> dict[str, Any]:
        fraction = self.applied_samples / self.total_samples if self.total_samples else 0.0
        return {
            "seed": self.seed,
            "probability": self.probability,
            "total_samples": self.total_samples,
            "applied_samples": self.applied_samples,
            "applied_fraction": fraction,
            "gamma_min": None if self.applied_samples == 0 else self.gamma_min,
            "gamma_max": None if self.applied_samples == 0 else self.gamma_max,
            "max_mean_abs_error": self.max_mean_abs_error,
            "max_population_std_abs_error": self.max_std_abs_error,
        }
