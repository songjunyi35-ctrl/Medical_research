"""Freeze UTSW checkpoint/source access after every conditional gate passes."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from scripts import utsw_no_n4_paired_external_protocol as p


def main() -> None:
    from scripts.freeze_utsw_no_n4_paired_external import check_implementation

    check_implementation()
    if not p.four_hour_gate():
        raise RuntimeError("less than four hours remain before UTSW source access")
    if shutil.disk_usage(p.SOURCE.parent).free < 16 * 1024**3:
        raise RuntimeError("less than 16GiB /tmp free before UTSW source access")
    rows = p.cohort()
    precedence = p.verify_precedence()
    reports, evidence = p.eligible_models()
    models = {
        role: {
            "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "report_sha256": item["report_sha256"],
        }
        for role, item in reports.items()
    }
    value = {
        "schema": "aflow.utsw_no_n4_paired_external.preaccess",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_utsw_model_and_source_access",
        "cohort": {
            "cases": 362, "subjects": 362, "all_manual_yes_retained": True,
            "manual_ids_sha256": p.MANUAL_IDS_SHA,
            "source_files": p.SOURCE_FILES, "source_bytes_per_copy": p.SOURCE_BYTES,
        },
        "models": models,
        "eligibility_evidence": evidence,
        "rhuh_precedence_sha256": p.sha(p.PRECEDENCE),
        "rhuh_precedence_status": precedence["status"],
        "protocol": {
            "arms": {"canonical": "positive-mask N4", "no_n4": "no N4"},
            "target": "UTSW WT = every published label > 0",
            "grid": "primary FeTS image-support crop resized to 128^3; metrics on raster",
            "views": ["identity", "Flip8 raw-logit mean"],
            "threshold": 0.0,
            "sam": False,
            "postprocessing": False,
            "bootstrap": {"unit": "subject", "samples": p.BOOTSTRAP_SAMPLES, "seed": p.BOOTSTRAP_SEED},
            "support_gates": {"flip8_scan_delta_gt": 0.0, "delta_ci_lower_ge": -0.003, "point_deficit_max": 3},
        },
        "time_gate": {"deadline_utc": p.DEADLINE.isoformat(), "minimum_remaining_hours": 4, "passed": True},
        "disk_gate": {"minimum_free_gib": 16, "passed": True},
        "access": {
            "existing_utsw_subjects": ["BT0001", "BT0002", "BT0003"],
            "new_utsw_subjects": False,
            "model_checkpoint_loaded": False,
            "model_forwards": False,
            "external_metrics": False,
            "fixed_validation": False, "outer": False, "confirmation": False,
            "test270": False, "ucsd75": False,
        },
        "cohort_rows_sha256": __import__("hashlib").sha256(
            ("\n".join(row["subject_id"] for row in rows) + "\n").encode()
        ).hexdigest(),
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.PREACCESS, value)
    p.verify_preaccess()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
