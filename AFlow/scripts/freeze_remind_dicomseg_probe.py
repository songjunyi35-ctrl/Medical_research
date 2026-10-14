"""Freeze/verify ReMIND DICOM SEG source-probe implementation."""
from datetime import datetime, timezone
import json

from scripts import remind_dicomseg_probe_protocol as p

FILES = (
    "scripts/remind_dicomseg_probe_protocol.py",
    "scripts/prepare_remind_dicomseg_probe.py",
    "scripts/download_remind_dicomseg_probe.py",
    "scripts/audit_remind_dicomseg_probe.py",
    "scripts/freeze_remind_dicomseg_probe.py",
    "tests/test_remind_dicomseg_probe.py",
)
RECEIPT = p.RECORD / "IMPLEMENTATION.json"


def current() -> dict[str, str]:
    return {name: p.sha(p.ROOT / name) for name in FILES}


def check_implementation() -> dict:
    p.docs()
    value = json.loads(RECEIPT.read_text())
    if value.get("files") != current():
        raise ValueError("ReMIND DICOM SEG implementation changed")
    return value


def main() -> None:
    p.docs()
    value = {
        "schema": "aflow.remind_dicomseg_source_probe.implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_dicomseg_download",
        "files": current(),
        "case_id": "ReMIND-002",
        "assets": p.ASSETS,
        "dependency_manifest_sha256": p.DEPENDENCY_MANIFEST_SHA,
        "model_inference": False,
    }
    p.write_new(RECEIPT, value)
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
