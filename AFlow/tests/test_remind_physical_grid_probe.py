from decimal import Decimal

import numpy as np
import SimpleITK as sitk

from scripts.audit_remind_physical_grid_probe import ds_interval, gap_interval, make_image, positive_bbox_inside


def test_scientific_decimal_quantum_uses_decimal_exponent():
    assert ds_interval("1.000000e+00")[2] == Decimal("0.0000005")
    assert ds_interval("4.000000e-01")[2] == Decimal("0.00000005")
    assert ds_interval("-.993743")[2] == Decimal("0.0000005")


def test_gap_interval_contains_declared_spacing_at_written_precision():
    low, high, nominal, bound = gap_interval(["0.000", "0.000", "0.000"], ["0.000", "0.000", "1.000"])
    assert low <= 1 <= high and nominal == 1 and bound > 0


def test_physical_image_axis_contract_and_bbox():
    array = np.zeros((3, 4, 5), dtype=np.uint8)
    array[1, 1:3, 1:4] = 1
    image = make_image(array, ["0", "0", "0"], ["1", "0", "0", "0", "1", "0"], ["2", "3"], "4")
    assert image.GetSize() == (5, 4, 3)
    assert image.GetSpacing() == (3.0, 2.0, 4.0)
    inside, metadata = positive_bbox_inside(image, array, image)
    assert inside and metadata["positive"] == 6
