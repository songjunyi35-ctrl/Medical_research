from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate_t1ce_wt_residual_refiner_postprocess import (
    binary_dice,
    filter_components,
)


def test_component_filters() -> None:
    prediction = np.zeros((8, 8, 8), dtype=bool)
    prediction[1:4, 1:4, 1:4] = True
    prediction[6, 6, 6] = True
    assert np.array_equal(filter_components(prediction, "raw"), prediction)
    assert int(filter_components(prediction, "largest_component").sum()) == 27
    assert int(filter_components(prediction, "remove_lt_10").sum()) == 27
    assert int(filter_components(prediction, "remove_lt_100").sum()) == 0


def test_binary_dice_and_invalid_mode() -> None:
    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    assert binary_dice(target, target) == 1.0
    assert binary_dice(np.zeros_like(target), target) == 0.0
    with pytest.raises(ValueError, match="unknown"):
        filter_components(target, "bad")
