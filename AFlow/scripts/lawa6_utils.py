"""Deterministic latest-weight averaging helpers for the frozen LAWA6 study."""
from __future__ import annotations

from collections.abc import Mapping

import torch


FORMAL_CAPTURE_EPOCHS = (163, 164, 165, 166, 167, 168)
SMOKE_CAPTURE_EPOCHS = (1, 2, 3, 4, 5, 6)


class LawaAccumulator:
    """Accumulate floating tensors in CPU float64 and audit discrete buffers."""

    def __init__(self, expected_count: int = 6) -> None:
        if expected_count != 6:
            raise ValueError("the frozen protocol requires exactly six states")
        self.expected_count = expected_count
        self.count = 0
        self._keys: tuple[str, ...] | None = None
        self._sums: dict[str, torch.Tensor] = {}
        self._discrete: dict[str, torch.Tensor] = {}
        self._dtypes: dict[str, torch.dtype] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

    def update(self, state: Mapping[str, torch.Tensor]) -> None:
        if self.count >= self.expected_count:
            raise RuntimeError("received more than six states")
        keys = tuple(state.keys())
        if self._keys is None:
            self._keys = keys
        elif keys != self._keys:
            raise ValueError("state keys or ordering changed across captures")

        for key, value in state.items():
            tensor = value.detach().cpu()
            if self.count == 0:
                self._dtypes[key] = tensor.dtype
                self._shapes[key] = tuple(tensor.shape)
            elif tuple(tensor.shape) != self._shapes[key] or tensor.dtype != self._dtypes[key]:
                raise ValueError(f"state tensor metadata changed: {key}")
            if tensor.is_floating_point() or tensor.is_complex():
                promoted = tensor.to(torch.complex128 if tensor.is_complex() else torch.float64)
                if self.count == 0:
                    self._sums[key] = promoted.clone()
                else:
                    self._sums[key].add_(promoted)
            elif self.count == 0:
                self._discrete[key] = tensor.clone()
            elif not torch.equal(self._discrete[key], tensor):
                raise ValueError(f"non-floating tensor changed across captures: {key}")
        self.count += 1

    def average(self) -> dict[str, torch.Tensor]:
        if self.count != self.expected_count or self._keys is None:
            raise RuntimeError("average requested before all six captures")
        result: dict[str, torch.Tensor] = {}
        for key in self._keys:
            if key in self._sums:
                result[key] = (self._sums[key] / self.expected_count).to(self._dtypes[key])
            else:
                result[key] = self._discrete[key].clone()
        return result


def cpu_clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def capture_epochs(smoke: bool) -> tuple[int, ...]:
    return SMOKE_CAPTURE_EPOCHS if smoke else FORMAL_CAPTURE_EPOCHS
