#!/usr/bin/env python3
"""Correct superseded leaderboard status labels after the Flip8 insertion."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "AFLOW_BRATS_LEADERBOARD.json"


def main() -> None:
    value = json.loads(PATH.read_text(encoding="utf-8"))
    rows = value["test270_direct_locator_rank_by_scan"]
    by_method = {row["method"]: row for row in rows}
    flip8 = by_method["full972_random_init_voxel_kd_epoch168_flip8_tta"]
    if flip8.get("rank") != 1 or flip8.get("scan_dice") != 0.7572696616942132:
        raise RuntimeError("unexpected Flip8 leaderboard row")
    if "absolute_ci_seed" in flip8 or "bootstrap_correction_path" in flip8:
        raise RuntimeError("Flip8 provenance correction already applied")
    gcsam = by_method["gcsam_inner_fit864_epoch184"]
    identity = by_method["full972_random_init_voxel_kd_epoch168"]
    if gcsam.get("status") != "highest_reused_internal_test_scan_point_estimate_only; subject_and_point_not_best; paired_dice_ci_crosses_zero":
        raise RuntimeError("unexpected pre-correction GCSAM status")
    if identity.get("status") != "highest_reused_internal_test_subject_and_point_point_estimates_failed_gate_exploratory; rank2_by_scan":
        raise RuntimeError("unexpected pre-correction identity status")
    flip8["absolute_ci_seed"] = 20260817
    flip8["bootstrap_correction_path"] = "archive/research_records/2026-09-04_direct_locator_flip8_tta_stability/BOOTSTRAP_CORRECTION.json"
    gcsam["status"] = "highest_1x_reused_internal_test_scan_point_estimate; rank2_overall_by_scan; subject_and_point_not_best; paired_dice_ci_crosses_zero"
    identity["status"] = "highest_1x_reused_internal_test_subject_and_point_point_estimates_failed_gate_exploratory; rank3_overall_by_scan"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=PATH.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(PATH)
    print(json.dumps({"status": "corrected", "fields": 4}, indent=2))


if __name__ == "__main__":
    main()
