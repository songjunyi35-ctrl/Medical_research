import numpy as np
import pytest

from scripts.audit_utsw_primary_fets_no_n4_input_bridge import validate_source_arrays


def test_all_positive_utsw_labels_become_wt():
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[1:, 1:, 1:] = 2.0
    label = np.zeros_like(image)
    label[1, 1, 1] = 1
    label[1, 1, 2] = 2
    label[1, 2, 1] = 4
    binary, row = validate_source_arrays(image, label)
    assert binary.sum() == 3
    assert binary[1, 2, 1] == 1
    assert row["label_values"] == [0.0, 1.0, 2.0, 4.0]


def test_negative_primary_image_fails_closed():
    image = np.array([[[0.0, -1.0], [2.0, 0.0]]], dtype=np.float32)
    label = np.array([[[0, 0], [1, 0]]], dtype=np.uint8)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_source_arrays(image, label)


def test_fractional_or_empty_label_fails_closed():
    image = np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32)
    with pytest.raises(ValueError, match="integer-exact"):
        validate_source_arrays(image, np.array([[[0.0, 0.5], [0.0, 0.0]]]))
    with pytest.raises(ValueError, match="background and tumor"):
        validate_source_arrays(image, np.zeros_like(image))
