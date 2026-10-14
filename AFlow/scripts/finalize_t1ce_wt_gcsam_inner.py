#!/usr/bin/env python3
"""Apply the preregistered paired gates to completed GCSAM train-inner arms."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_gcsam_inner as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_gcsam_inner_decision"


def load_report(path: Path, arm: str, runner_sha: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError("unexpected inner report schema/status")
    if value.get("arm") != arm or value.get("runner_sha256") != runner_sha:
        raise ValueError("inner report arm/runner mismatch")
    if value.get("all_gates_passed") is not True or len(value.get("history", [])) != 200:
        raise ValueError("inner engineering gates/history failed")
    if value.get("access") != {
        "train_only_inner": True,
        "fixed_validation108": False,
        "outer": False,
        "confirmation": False,
        "test270": False,
    }:
        raise ValueError("inner access seal mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing repeated inner decision")
    runner_sha = sha256_file(ROOT / "scripts/train_t1ce_wt_gcsam_inner.py")
    control = load_report(args.control, "adamw_control", runner_sha)
    candidate = load_report(args.candidate, "gcsam_candidate", runner_sha)
    if control["initial_state_sha256"] != candidate["initial_state_sha256"]:
        raise ValueError("paired initial states differ")
    if control["sampling_weight_sha256"] != candidate["sampling_weight_sha256"]:
        raise ValueError("paired sampling weights differ")
    control_sampler = [row["sampler_index_sha256"] for row in control["history"]]
    candidate_sampler = [row["sampler_index_sha256"] for row in candidate["history"]]
    if control_sampler != candidate_sampler:
        raise ValueError("paired sampler index sequences differ")
    if control["history"][0]["first_batch_sha256"] != candidate["history"][0]["first_batch_sha256"]:
        raise ValueError("paired first augmented batch differs")

    selected_epoch = int(candidate["selections"]["post100"]["epoch"])
    candidate_row = next(row for row in candidate["history"] if int(row["epoch"]) == selected_epoch)
    control_row = next(row for row in control["history"] if int(row["epoch"]) == selected_epoch)
    comparison = compare(control_row["inner"], candidate_row["inner"], seed=20261004, samples=10_000)
    delta = comparison["ema_minus_raw"]
    gates = {
        "paired_initial_state_exact": True,
        "paired_sampling_weights_exact": True,
        "paired_all_epoch_indices_exact": True,
        "paired_first_augmented_batch_exact": True,
        "candidate_scan_strictly_higher": float(delta["scan"]) > 0.0,
        "candidate_subject_strictly_higher": float(delta["subject"]) > 0.0,
        "point_drop_at_most_one_of_108": float(delta["point_hit"]) >= -(1.0 / 108.0) - 1e-12,
        "scan_ci_lower_at_least_minus_0.003": float(delta["scan_95ci"][0]) >= -0.003,
        "subject_ci_lower_at_least_minus_0.003": float(delta["subject_95ci"][0]) >= -0.003,
        "both_engineering_reports_passed": True,
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "selected_epoch": selected_epoch,
        "selection_source": "gcsam candidate post100 max Scan, Subject tie, earlier epoch",
        "control": control_row["inner"],
        "candidate": candidate_row["inner"],
        "comparison": comparison,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "final_retraining_authorized": all(gates.values()),
        "inputs": {
            "control_report_sha256": sha256_file(args.control),
            "candidate_report_sha256": sha256_file(args.candidate),
            "runner_sha256": runner_sha,
        },
        "access": {"train_only_inner": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False},
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

