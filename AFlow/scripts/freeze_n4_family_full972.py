"""Freeze and verify the isolated conditional Full972 implementation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import n4_family_full972_protocol as p

FILES = (
    "scripts/n4_family_full972_protocol.py",
    "scripts/freeze_n4_family_full972.py",
    "scripts/train_n4_family_full972.py",
    "scripts/run_n4_family_full972.py",
    "scripts/queue_n4_family_full972.py",
    "scripts/prepare_n4_family_full972_admission.py",
    "scripts/no_n4_clean_protocol.py",
    "scripts/n4_otsu_protocol.py",
    "scripts/train_t1ce_wt_gcsam_inner.py",
    "tests/test_n4_family_full972_contract.py",
    "tests/test_n4_full972_admission.py",
)
PATH = p.RECORD / "FULL972_IMPLEMENTATION.json"


def current() -> dict[str, str]:
    return {name: p.sha(p.ROOT / name) for name in FILES}


def check_implementation() -> dict:
    p.docs()
    if not PATH.is_file():
        raise RuntimeError("Full972 implementation has not been frozen")
    value = json.loads(PATH.read_text())
    if value.get("files") != current():
        raise ValueError("Full972 implementation hash mismatch")
    return value


def main() -> None:
    p.docs()
    value = {
        "schema": "aflow.n4_family_full972.implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_full972_execution",
        "files": current(),
        "seeds": p.SEEDS,
        "epochs": p.EPOCHS,
        "cases": 972,
        "subjects": 444,
        "automatic_old_cohort_evaluation": False,
    }
    p.write_new(PATH, value)
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
