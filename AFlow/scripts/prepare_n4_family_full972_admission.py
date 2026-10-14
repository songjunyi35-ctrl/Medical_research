"""Fail-closed admission check for the preregistered conditional Full972 stage."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import no_n4_clean_protocol as shared

DEADLINE = datetime.fromisoformat("2026-09-06T14:02:39+00:00")


def performance_passes(report):
    comparisons = report["comparisons"]
    valid = True
    for fold in ("e", "f"):
        for style in ("identity", "flip8"):
            row = comparisons[fold][style]
            delta = row["delta"]
            values = [delta["scan_dice"], delta["subject_dice"], delta["point_hits"],
                      *delta["scan_dice_95ci"], *delta["subject_dice_95ci"]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite promotion evidence")
            gates = {
                "scan_positive": delta["scan_dice"] > 0,
                "subject_positive": delta["subject_dice"] > 0,
                "point_within_one": delta["point_hits"] >= -1,
                "scan_ci_safe": delta["scan_dice_95ci"][0] >= -0.003,
                "subject_ci_safe": delta["subject_dice_95ci"][0] >= -0.003,
            }
            if row["gates"] != gates or row["all_gates_passed"] != all(gates.values()):
                raise ValueError("recorded fold gates contradict frozen inequalities")
            valid = valid and all(gates.values())
    for style in ("identity", "flip8"):
        scan = sum(comparisons[f][style]["delta"]["scan_dice"] for f in ("e", "f")) / 2
        subject = sum(comparisons[f][style]["delta"]["subject_dice"] for f in ("e", "f")) / 2
        valid = valid and scan > 0.001 and subject > 0
    if report["aggregate"]["all_gates_passed"] != valid or report["stage_b_authorized"] != valid:
        raise ValueError("recorded Stage B authorization contradicts raw gates")
    return valid


def time_gate(now):
    if now.tzinfo is None:
        raise ValueError("timezone-aware time required")
    return (DEADLINE - now).total_seconds() >= 8 * 3600


def main():
    shared.docs()
    family_done = shared.RECORD / "FAMILY_READOUT_DONE.json"
    if not family_done.is_file():
        raise RuntimeError("all family readouts and archival must finish first")
    records = {
        "no_n4": shared.RECORD,
        "otsu": ROOT / "archive/research_records/2026-09-05_n4_otsu_clean_replication",
    }
    done = json.loads(family_done.read_text())
    decisions = {}
    for method, directory in records.items():
        raw_path = directory / "RAW_INNER.json"
        shared.checked(raw_path, done[method + "_raw_sha256"])
        raw = json.loads(raw_path.read_text())
        decisions[method] = {"performance_passed": performance_passes(raw), "raw_sha256": shared.sha(raw_path)}
    now = datetime.now(timezone.utc)
    eligible = [method for method in ("no_n4", "otsu") if decisions[method]["performance_passed"]]
    report = {
        "created_utc": now.isoformat(), "decisions": decisions,
        "eligible_methods_in_fixed_priority": eligible,
        "eight_hour_time_gate": time_gate(now),
        "remaining_hours": max(0, (DEADLINE - now).total_seconds() / 3600),
        "automatic_training_launched": False,
        "next_step": "prepare and verify complete Full972 execution only for eligible methods with sufficient time",
        "seeds": {"history_teacher": 20261341, "ema_teacher": 20261342, "paired_students": 20261343},
        "epochs": 168, "source_scans": 972, "source_subjects": 444,
        "old_cohort_evaluation_authorized": False,
        "source_sha256": shared.SOURCE_SHA,
        "runner_sha256": shared.sha(Path(__file__)),
    }
    shared.write_new(shared.RECORD / "FULL972_ADMISSION_CHECK.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
