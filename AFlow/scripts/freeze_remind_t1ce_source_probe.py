"""Freeze/verify ReMIND source-probe implementation before pixel download."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from scripts.download_remind_t1ce_source_probe import RECORD, ROOT, sha

FILES = (
    "scripts/prepare_remind_t1ce_source_probe.py",
    "scripts/download_remind_t1ce_source_probe.py",
    "scripts/audit_remind_t1ce_source_probe.py",
    "scripts/freeze_remind_t1ce_source_probe.py",
    "tests/test_remind_t1ce_source_probe.py",
    "tests/test_remind_t1ce_source_probe_qc.py",
)
RECEIPT = RECORD / "IMPLEMENTATION.json"


def current() -> dict[str, str]:
    return {name: sha(ROOT / name) for name in FILES}


def check_implementation() -> dict:
    value = json.loads(RECEIPT.read_text())
    if value.get("files") != current():
        raise ValueError("ReMIND source-probe implementation changed")
    return value


def main() -> None:
    value = {
        "schema": "aflow.remind_t1ce_source_probe.implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_probe_pixels",
        "files": current(),
        "probe": "ReMIND-002",
        "candidate_pixels": 1,
        "nrrd_transfers": 2,
        "model_inference": False,
    }
    with RECEIPT.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
