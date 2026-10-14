from __future__ import annotations

import pytest
import torch

from scripts.lawa6_utils import LawaAccumulator, capture_epochs


def test_lawa6_uniform_float64_accumulation_and_dtype_restore() -> None:
    accumulator = LawaAccumulator()
    for value in range(6):
        accumulator.update({
            "weight": torch.tensor([value, value + 2], dtype=torch.float32),
            "counter": torch.tensor(7, dtype=torch.int64),
        })
    averaged = accumulator.average()
    assert averaged["weight"].dtype == torch.float32
    assert torch.equal(averaged["weight"], torch.tensor([2.5, 4.5]))
    assert torch.equal(averaged["counter"], torch.tensor(7))


def test_lawa6_rejects_nonfloating_buffer_change() -> None:
    accumulator = LawaAccumulator()
    accumulator.update({"counter": torch.tensor(1)})
    with pytest.raises(ValueError, match="non-floating"):
        accumulator.update({"counter": torch.tensor(2)})


def test_lawa6_rejects_key_or_metadata_change() -> None:
    accumulator = LawaAccumulator()
    accumulator.update({"weight": torch.ones(2)})
    with pytest.raises(ValueError, match="keys"):
        accumulator.update({"other": torch.ones(2)})

    accumulator = LawaAccumulator()
    accumulator.update({"weight": torch.ones(2)})
    with pytest.raises(ValueError, match="metadata"):
        accumulator.update({"weight": torch.ones(3)})


def test_lawa6_requires_exactly_six_captures() -> None:
    accumulator = LawaAccumulator()
    for _ in range(5):
        accumulator.update({"weight": torch.ones(1)})
    with pytest.raises(RuntimeError, match="six"):
        accumulator.average()
    accumulator.update({"weight": torch.ones(1)})
    accumulator.average()
    with pytest.raises(RuntimeError, match="more than six"):
        accumulator.update({"weight": torch.ones(1)})


def test_lawa6_capture_epochs_are_frozen() -> None:
    assert capture_epochs(False) == (163, 164, 165, 166, 167, 168)
    assert capture_epochs(True) == (1, 2, 3, 4, 5, 6)
