import numpy as np

from scripts.audit_utsw_label_variant_bridge import array_validity, geometry_row


class Dummy:
    def __init__(self, shape, affine):
        self.shape = shape
        self.affine = affine


def test_geometry_gate_requires_shape_and_affine():
    identity = np.eye(4)
    assert geometry_row(Dummy((2, 3, 4), identity), Dummy((2, 3, 4), identity))["passed"]
    shifted = identity.copy()
    shifted[0, 3] = 2e-6
    assert not geometry_row(Dummy((2, 3, 4), identity), Dummy((2, 3, 4), shifted))["passed"]
    assert not geometry_row(Dummy((2, 3, 4), identity), Dummy((2, 3, 5), identity))["passed"]


def test_label_validity_is_exact_integer_nonempty_with_background():
    row = array_validity(np.array([0.0, 1.0, 2.0, 4.0]), is_label=True)
    assert row["finite"] and row["integer_exact"] and row["includes_zero"] and row["includes_positive"]
    assert not array_validity(np.array([0.0, 1.5]), is_label=True)["integer_exact"]
    assert not array_validity(np.array([0.0, np.nan]), is_label=True)["finite"]


def test_image_validity_reports_support_without_label_assumptions():
    row = array_validity(np.array([0.0, -1.0, 2.0]), is_label=False)
    assert row == {"finite": True, "shape": [3], "nonzero_voxels": 2}
