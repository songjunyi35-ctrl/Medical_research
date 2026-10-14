import numpy as np

from scripts.analyze_flip8_external_overlap_synthesis import ci, vals


def test_empty_prediction_precision_uses_project_convention_one():
    rows = [
        {
            "predicted_wt_voxels": 0,
            "target_wt_voxels": 10,
            "intersection_voxels": 0,
        },
        {
            "predicted_wt_voxels": 8,
            "target_wt_voxels": 10,
            "intersection_voxels": 4,
        },
    ]
    out = vals(rows)
    np.testing.assert_allclose(out["precision"], [1.0, 0.5])
    np.testing.assert_allclose(out["recall"], [0.0, 0.4])
    np.testing.assert_allclose(out["predicted"], [0.0, 8.0])
    np.testing.assert_allclose(out["ratio"], [0.0, 0.8])


def test_bootstrap_ci_is_deterministic_and_contains_constant():
    assert ci(np.full(5, 0.25), seed=123) == [0.25, 0.25]
    assert ci(np.arange(5, dtype=float), seed=456) == ci(
        np.arange(5, dtype=float), seed=456
    )
