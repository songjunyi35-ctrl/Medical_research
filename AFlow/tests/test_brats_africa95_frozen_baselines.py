from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import brats_africa95_protocol as protocol
import scripts.download_brats_africa95_source as download
import scripts.evaluate_brats_africa95_frozen_baselines as evaluate


def test_frozen_cohort_has_exact_95_unique_pairs() -> None:
    rows = protocol.cohort_rows()
    assert len(rows) == len({row["case_id"] for row in rows}) == 95
    assert len({row["subject_id"] for row in rows}) == 95
    assert download.expected(rows) and sum(download.expected(rows).values()) == 277_630_584
    assert all("/95_Glioma/" in row[kind]["path"] for row in rows for kind in ("t1ce", "seg"))


def test_reflection_group_is_exact_and_invertible() -> None:
    assert len(protocol.AXES) == len(set(protocol.AXES)) == 8 and protocol.AXES[0] == ()
    x = torch.arange(2 * 3 * 4).reshape(1, 1, 2, 3, 4)
    for axes in protocol.AXES:
        assert torch.equal(torch.flip(torch.flip(x, axes), axes), x)


def _summary(values):
    rows = [{"case_id": f"c{i}", "subject_id": f"s{i}", "locator_dsc": value, "automatic_point_hits_gt": hit} for i, (value, hit) in enumerate(values)]
    return {"per_case": rows, "automatic_point_hits": sum(hit for _, hit in values)}


def test_paired_comparison_direction_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "BOOTSTRAP_SAMPLES", 100)
    left = _summary([(0.4, False), (0.8, True)])
    right = _summary([(0.6, True), (0.7, True)])
    result = evaluate.compare(left, right, 3, "right_minus_left")
    assert result["right_minus_left"]["scan_dice"] == pytest.approx(0.05)
    assert result["right_minus_left"]["point_hits"] == 1
    assert (result["cases_improved"], result["cases_worse"], result["cases_tied"]) == (1, 1, 0)


def test_preoperative_target_is_all_positive_and_not_rc_exclusion() -> None:
    source = Path("scripts/build_brats_africa95_cache.py").read_text()
    assert "rounded > 0" in source
    assert "{0, 1, 2, 3, 4}" in source
    prereg = json.loads((protocol.RECORD / "PREREGISTRATION.json").read_text())
    assert prereg["target"]["rule"] == "all positive segmentation values"


def test_no_performance_gate_and_exact_forward_budget() -> None:
    source = Path(evaluate.__file__).read_text()
    assert '"performance_promotion_gate": None' in source
    assert "forwards != 855" in source
    assert '"result_must_not_trigger_tuning_or_repeat": True' in source
