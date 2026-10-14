"""Freeze/verify RHUH Otsu input-only implementation."""
from __future__ import annotations

from datetime import datetime, timezone
import json

from scripts.audit_rhuh_otsu_input_bridge import RECORD, ROOT, sha

FILES = (
    "scripts/audit_rhuh_otsu_input_bridge.py",
    "scripts/freeze_rhuh_otsu_input_bridge.py",
    "sammed3d/background_zero_decode.py",
    "sammed3d/n4_otsu_input.py",
    "tests/test_rhuh_otsu_input_bridge.py",
)
RECEIPT = RECORD / "IMPLEMENTATION.json"


def current() -> dict[str, str]:
    return {name: sha(ROOT / name) for name in FILES}


def check_implementation() -> dict:
    value = json.loads(RECEIPT.read_text())
    if value.get("files") != current():
        raise ValueError("RHUH Otsu bridge implementation hash mismatch")
    return value


def main() -> None:
    value = {
        "schema": "aflow.rhuh_otsu_input_bridge.implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_input_qc",
        "files": current(),
        "label_access": False,
        "model_inference": False,
    }
    with RECEIPT.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    check_implementation()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
