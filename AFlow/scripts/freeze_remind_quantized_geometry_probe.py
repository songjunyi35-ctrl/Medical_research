"""Freeze/verify quantization-aware ReMIND geometry probe."""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import remind_dicomseg_probe_protocol as base

ROOT = base.ROOT
RECORD = ROOT / "archive/research_records/2026-09-05_remind_quantized_geometry_probe"
FILES = (
    "archive/research_records/2026-09-05_remind_quantized_geometry_probe/IDEA.md",
    "archive/research_records/2026-09-05_remind_quantized_geometry_probe/PREREG.md",
    "scripts/audit_remind_quantized_geometry_probe.py",
    "scripts/freeze_remind_quantized_geometry_probe.py",
    "tests/test_remind_quantized_geometry_probe.py",
    "scripts/remind_dicomseg_probe_protocol.py",
)
RECEIPT = RECORD / "IMPLEMENTATION.json"


def current() -> dict[str, str]:
    return {name: base.sha(ROOT / name) for name in FILES}


def check_implementation() -> dict:
    value = json.loads(RECEIPT.read_text())
    if value.get("files") != current():
        raise ValueError("quantized ReMIND geometry implementation changed")
    return value


def main() -> None:
    value = {"schema": "aflow.remind_quantized_geometry.implementation",
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "status": "frozen_before_any_pixel_array", "files": current(),
             "network_access": False, "model_inference": False,
             "download_receipt_sha256": "270ab28554f76f11142f6956efbfcb8cd54c659dbe5318e29c082c335880c4f9"}
    base.write_new(RECEIPT, value)
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
