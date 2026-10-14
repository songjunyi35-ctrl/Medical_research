from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_remind_quantized_geometry_probe import ds_interval, gap_interval, intersects


def test_ds_interval_uses_written_decimal_precision():
    low, high, half, text = ds_interval("135.104")
    assert text == "135.104" and half == Decimal("0.0005")
    assert low == Decimal("135.1035") and high == Decimal("135.1045")
    assert ds_interval("-.993743")[2] == Decimal("0.0000005")
    with pytest.raises(ValueError):
        ds_interval("1e-3")


def test_interval_intersection_and_gap_containment():
    assert intersects("1.000", "1.0004")
    assert not intersects("1.000", "1.002")
    low, high, nominal, bound = gap_interval(["0.000", "0.000", "0.000"], ["0.000", "0.000", "1.000"])
    assert low < 1 < high and nominal == 1 and bound > 0


def test_observed_rounded_components_are_compatible_with_one_mm():
    low, high, _, _ = gap_interval(["135.104", "-268.342", "-14.9747"],
                                   ["135.099", "-267.832", "-15.8352"])
    assert low <= 1.0 <= high and low > 0
