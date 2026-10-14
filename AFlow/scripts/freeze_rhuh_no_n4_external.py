"""Freeze/verify conditional RHUH no-N4 external execution code."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from scripts import rhuh_no_n4_external_protocol as p

FILES = (
    "scripts/rhuh_no_n4_external_protocol.py",
    "scripts/prepare_rhuh_no_n4_external.py",
    "scripts/download_rhuh_no_n4_external_labels.py",
    "scripts/build_rhuh_no_n4_external_cache.py",
    "scripts/evaluate_rhuh_no_n4_external.py",
    "scripts/smoke_rhuh_no_n4_external.py",
    "scripts/queue_rhuh_no_n4_external.py",
    "scripts/freeze_rhuh_no_n4_external.py",
    "scripts/audit_rhuh_no_n4_input_bridge.py",
    "sammed3d/background_zero_decode.py",
    "scripts/n4_family_full972_protocol.py",
    "tests/test_rhuh_no_n4_external_contract.py",
)
RECEIPT = p.RECORD / "EXTERNAL_IMPLEMENTATION.json"


def current() -> dict[str, str]:
    return {name: p.sha(p.ROOT / name) for name in FILES}


def check_implementation() -> dict:
    p.docs()
    value = json.loads(RECEIPT.read_text())
    if value.get("files") != current():
        raise ValueError("RHUH external implementation hash mismatch")
    return value


def main() -> None:
    p.docs()
    value = {
        "schema": "aflow.rhuh_no_n4_external.implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_checkpoint_eligibility_and_label_download",
        "files": current(),
        "cohort": {"cases": 80, "subjects": 40, "timepoints": [1, 2]},
        "views": 8,
        "threshold": 0.0,
        "bootstrap": {"samples": p.BOOTSTRAP_SAMPLES, "seed": p.BOOTSTRAP_SEED},
        "automatic_old_cohort_access": False,
    }
    p.write_new(RECEIPT, value)
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
