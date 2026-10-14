#!/usr/bin/env python3
"""Validate two independent KAS-arm reports and freeze their pairing receipt."""
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

from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_iad_kas_pairing_receipt"
TRAINING_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_continuation"
PAIRING_FIELDS = (
    "initial_student_state_sha256",
    "first_batch_case_ids",
    "first_batch_case_ids_sha256",
    "first_batch_raw_image_sha256",
    "first_batch_raw_label_sha256",
    "first_batch_augmented_image_sha256",
    "first_batch_augmented_label_sha256",
    "epoch_case_order_sha256",
    "epoch_cuda_rng_before_sha256",
    "epoch_cuda_rng_after_sha256",
)


def _read(path: Path, arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != TRAINING_SCHEMA or value.get("arm") != arm:
        raise ValueError(f"unexpected {arm} TRAIN_DONE contract")
    if value.get("status") != "complete":
        raise ValueError(f"{arm} training is not complete")
    for field in (
        "reload_max_abs_difference",
        "stripped_deploy_max_abs_difference",
    ):
        if float(value.get(field)) != 0.0:
            raise ValueError(f"{arm} {field} is not zero")
    for field in ("validation_access", "outer_selector_access", "confirmation_access", "test_access"):
        if value.get(field) is not False:
            raise ValueError(f"{arm} {field} must be false")
    pairing = value.get("pairing")
    if not isinstance(pairing, dict):
        raise ValueError(f"{arm} lacks pairing evidence")
    for field in PAIRING_FIELDS:
        if field not in pairing:
            raise ValueError(f"{arm} pairing lacks {field}")
    expected_epochs = {str(epoch) for epoch in range(201, 233)}
    for field in PAIRING_FIELDS[-3:]:
        if set(pairing[field]) != expected_epochs:
            raise ValueError(f"{arm} {field} does not cover global201-232")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--kas-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite pairing receipt")
    control = _read(args.control_report, "control")
    kas = _read(args.kas_report, "kas")
    mismatched = [
        field for field in PAIRING_FIELDS
        if control["pairing"][field] != kas["pairing"][field]
    ]
    if mismatched:
        raise RuntimeError(f"paired arms differ: {mismatched}")
    arms = {}
    for arm, report, path in (
        ("control", control, args.control_report),
        ("kas", kas, args.kas_report),
    ):
        arms[arm] = {
            "status": report["status"],
            "training_report": str(path.resolve()),
            "training_report_sha256": sha256_file(path),
            "reload_max_abs_difference": report["reload_max_abs_difference"],
            "stripped_deploy_max_abs_difference": report[
                "stripped_deploy_max_abs_difference"
            ],
            "pairing": report["pairing"],
            "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
            "elapsed_seconds": report["elapsed_seconds"],
        }
    receipt = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_before_new_validation_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        "mismatched_fields": mismatched,
        "all_gates_passed": not mismatched,
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "all_gates_passed": receipt["all_gates_passed"],
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
