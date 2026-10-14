from copy import deepcopy

from scripts.evaluate_utsw_no_n4_paired_external import bootstrap, summarize, support_gates


def rows(offset=0.0, point=True):
    return [
        {"case_id": f"BT{i:04d}", "subject_id": f"BT{i:04d}", "dice": 0.5 + offset,
         "point_hit": point, "predicted_voxels": 10, "target_voxels": 12}
        for i in range(1, 363)
    ]


def test_summary_enforces_one_scan_per_subject():
    value = summarize(rows())
    assert value["cases"] == value["subjects"] == 362
    assert value["scan_dice"] == value["subject_dice"] == 0.5
    assert value["point_hits"] == 362


def test_bootstrap_is_deterministic_and_paired():
    first = bootstrap(rows(), rows(0.01))
    second = bootstrap(rows(), rows(0.01))
    assert first == second
    low, high = first["scan_delta_no_n4_minus_canonical_95ci"]
    assert abs(low - 0.01) < 1e-12 and abs(high - 0.01) < 1e-12


def test_support_gate_uses_frozen_delta_ci_and_three_point_margin():
    canonical = summarize(rows())
    candidate_rows = rows(0.01)
    for item in candidate_rows[:3]:
        item["point_hit"] = False
    candidate = summarize(candidate_rows)
    interval = bootstrap(rows(), candidate_rows)
    assert all(support_gates(canonical, candidate, interval).values())
    candidate_rows[3]["point_hit"] = False
    assert not support_gates(canonical, summarize(candidate_rows), interval)["flip8_point_deficit_at_most_3"]
