from copy import deepcopy
from pathlib import Path

import pytest

import scripts.evaluate_t1ce_wt_flip8_tta_test270 as evaluate
import scripts.freeze_t1ce_wt_flip8_tta_test270_preaccess as freeze


ROOT = Path(__file__).resolve().parents[1]


def _summary(scan: float, subject: float, hits: int) -> dict:
    return {
        "locator_dsc_mean": scan,
        "locator_subject_dsc_mean": subject,
        "per_case": [
            {
                "case_id": f"C{index}",
                "subject_id": f"S{index}",
                "locator_dsc": scan,
                "automatic_point": [0, 0, 0],
                "automatic_point_hits_gt": index < hits,
            }
            for index in range(270)
        ],
    }


def test_success_gates_use_frozen_directions() -> None:
    identity = _summary(freeze.EXPECTED_IDENTITY["scan_dice"], freeze.EXPECTED_IDENTITY["subject_dice"], 262)
    flip8 = _summary(freeze.CURRENT_SCAN_LEADER + 1e-6, freeze.EXPECTED_IDENTITY["subject_dice"], 261)
    assert all(evaluate.success_gates(identity, flip8).values())
    flip8["locator_dsc_mean"] = freeze.CURRENT_SCAN_LEADER
    assert evaluate.success_gates(identity, flip8)["flip8_scan_strictly_above_current_leader"] is False
    flip8["locator_dsc_mean"] = freeze.CURRENT_SCAN_LEADER + 1e-6
    flip8["locator_subject_dsc_mean"] = freeze.EXPECTED_IDENTITY["subject_dice"] - 1e-12
    assert evaluate.success_gates(identity, flip8)["flip8_subject_at_least_identity"] is False
    flip8["locator_subject_dsc_mean"] = freeze.EXPECTED_IDENTITY["subject_dice"]
    flip8["per_case"][260]["automatic_point_hits_gt"] = False
    assert evaluate.success_gates(identity, flip8)["flip8_point_loses_at_most_one"] is False


def test_identity_parity_is_exact() -> None:
    value = _summary(0.75, 0.74, 262)
    evaluate.assert_identity_parity(value, deepcopy(value))
    changed = deepcopy(value)
    changed["per_case"][3]["locator_dsc"] += 1e-12
    with pytest.raises(RuntimeError, match="identity parity failed"):
        evaluate.assert_identity_parity(value, changed)


def test_frozen_formula_and_no_tuning_controls() -> None:
    source = (ROOT / "scripts/evaluate_t1ce_wt_flip8_tta_test270.py").read_text()
    assert 'batch_size=8' in source
    assert 'threshold": 0.0' in (ROOT / "scripts/freeze_t1ce_wt_flip8_tta_test270_preaccess.py").read_text()
    for control in ("--threshold", "--batch-size", "--reflection", "--weight", "--checkpoint-epoch"):
        assert control not in source
    assert "refusing repeated test270 evaluation" in source
    assert "assert_identity_parity" in source
    assert "result_must_not_trigger_view_weight_threshold_checkpoint_or_postprocessing_tuning" in source


def test_constants_match_preregistration() -> None:
    assert freeze.CHECKPOINT_SHA256 == "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
    assert freeze.IDENTITY_REPORT_SHA256 == "9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd"
    assert freeze.CURRENT_SCAN_LEADER == pytest.approx(0.752003176)
    assert freeze.EXPECTED_IDENTITY["point_hits"] == 262
    assert freeze.BOOTSTRAP_SAMPLES == 10_000
