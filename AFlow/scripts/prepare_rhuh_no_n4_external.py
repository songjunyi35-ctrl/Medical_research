"""Create the dynamic checkpoint-bound RHUH preaccess receipt after all gates."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import rhuh_no_n4_external_protocol as p


def main() -> None:
    from scripts.freeze_rhuh_no_n4_external import check_implementation

    check_implementation()
    if not p.two_hour_gate():
        raise RuntimeError("less than two hours remain; do not download RHUH labels")
    checkpoint, report, evidence = p.eligible_checkpoint()
    inventory = json.loads((p.REVIEW / "RHUH_LABEL_INVENTORY_READINESS.json").read_text())
    if inventory.get("all_gates_passed") is not True or inventory.get("labels_downloaded") is not False:
        raise ValueError("RHUH label inventory metadata is not sealed")
    value = {
        "schema": "aflow.rhuh_no_n4_external.preaccess",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_rhuh_label_download",
        "cohort": {"cases": 80, "subjects": 40, "timepoints": [1, 2], "all_cases_retained": True},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": report["checkpoint"]["sha256"],
            "epoch": 168,
            "role": "no_n4",
            "seed": 20261343,
        },
        "eligibility_evidence": evidence,
        "input_qc_sha256": p.INPUT_QC_SHA,
        "label_inventory_sha256": p.LABEL_INVENTORY_SHA,
        "official_sums_sha256": p.SUMS_SHA,
        "protocol": {
            "target": "WT = union of every positive published RHUH tumor label",
            "grid": "image-only crop resized to 128^3; metrics on this raster",
            "views": "fixed Flip8 raw-logit mean",
            "threshold": 0.0,
            "point": "argmax of averaged raw logits",
            "sam": False,
            "postprocessing": False,
            "bootstrap": {"unit": "patient_cluster_all_two_timepoints", "samples": p.BOOTSTRAP_SAMPLES, "seed": p.BOOTSTRAP_SEED},
            "absolute_gates": {"scan_dice": 0.70, "subject_dice": 0.70, "point_hits": 72},
        },
        "time_gate": {"deadline_utc": p.DEADLINE.isoformat(), "minimum_remaining_hours": 2, "passed": True},
        "access": {
            "rhuh_images": True,
            "rhuh_labels_downloaded": False,
            "rhuh_label_voxels": False,
            "model_checkpoint_loaded": False,
            "model_forwards": False,
            "new_metrics": False,
            "fixed_validation": False,
            "outer": False,
            "confirmation": False,
            "test270": False,
            "ucsd75": False,
        },
        "automatic_repeat_or_tuning": False,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.PREACCESS, value)
    p.verify_preaccess()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
