#!/usr/bin/env python3
"""Validate the two formal VM-Dice arms and freeze their pairing receipt."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_vm_dice_pairing_receipt"
TRAINING_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_vm_dice_continuation"
ARMS = ("paper_dice_gamma0", "vm_dice_gamma1")
EXPECTED_GAMMAS = {"paper_dice_gamma0": 0.0, "vm_dice_gamma1": 1.0}
EPOCHS = (208, 216, 224, 232)
TRAINING_GLOBAL_EPOCHS = tuple(range(201, 233))
GPU_MEMORY_STOP_BYTES = 24 * 1024**3
ACCESS_FLAGS = (
    "validation_access",
    "outer_selector_access",
    "confirmation_access",
    "test_access",
)
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


def _expected_formula_contract(arm: str) -> dict[str, Any]:
    gamma = EXPECTED_GAMMAS[arm]
    return {
        "arm": arm,
        "gamma": gamma,
        "gamma_foreground": gamma,
        "gamma_background": gamma,
        "epsilon": 1e-6,
        "class_order": ["background", "foreground"],
        "present_gt_classes_only": True,
        "modulation_detached": True,
        "denominator": "sum(m*(target^2+probability^2))",
        "bce_with_logits_weight": 0.1,
        "teacher_forwards": 0,
        "deploy": "plain student final raw foreground logit only",
    }


def _require_exact_zero(value: Any, *, field: str, source: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} lacks numeric {field}") from exc
    if not math.isfinite(number) or number != 0.0:
        raise ValueError(f"{source} {field} must be exactly zero")


def _read(path: Path, arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    source = f"{arm} TRAIN_DONE"
    if value.get("schema") != TRAINING_SCHEMA or value.get("schema_version") != 1:
        raise ValueError(f"unexpected {source} schema")
    if value.get("arm") != arm or value.get("gamma") != EXPECTED_GAMMAS[arm]:
        raise ValueError(f"{source} arm/gamma mismatch")
    if value.get("status") != "complete":
        raise ValueError(f"{source} is not complete")
    if value.get("all_self_gates_passed") is not True:
        raise ValueError(f"{source} self gates did not pass")
    for flag in ACCESS_FLAGS:
        if value.get(flag) is not False:
            raise ValueError(f"{source} {flag} must be false")
    for field in (
        "reload_max_abs_difference",
        "stripped_deploy_max_abs_difference",
    ):
        _require_exact_zero(value.get(field), field=field, source=source)
    peak = int(value.get("peak_gpu_memory_bytes", -1))
    if peak < 0 or peak > GPU_MEMORY_STOP_BYTES:
        raise ValueError(f"{source} violates the formal 24-GiB memory gate")
    history = value.get("history")
    if not isinstance(history, list) or len(history) != len(TRAINING_GLOBAL_EPOCHS):
        raise ValueError(f"{source} history is not 32 epochs")
    if [item.get("global_epoch") for item in history] != list(TRAINING_GLOBAL_EPOCHS):
        raise ValueError(f"{source} global epoch sequence differs")
    if any(float(item.get("learning_rate")) != 1e-5 for item in history):
        raise ValueError(f"{source} learning rate differs")
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {
        str(epoch) for epoch in EPOCHS
    }:
        raise ValueError(f"{source} checkpoint grid differs")
    for epoch in EPOCHS:
        item = checkpoints[str(epoch)]
        if not isinstance(item, dict):
            raise ValueError(f"{source} epoch{epoch} checkpoint entry is invalid")
        checkpoint_path = Path(str(item.get("path", "")))
        if not checkpoint_path.is_file():
            raise ValueError(f"{source} epoch{epoch} checkpoint file is absent")
        if sha256_file(checkpoint_path) != item.get("sha256"):
            raise ValueError(f"{source} epoch{epoch} checkpoint SHA mismatch")
        _require_exact_zero(
            item.get("checkpoint_reload_max_abs_difference"),
            field="checkpoint_reload_max_abs_difference",
            source=f"{source} epoch{epoch}",
        )
        _require_exact_zero(
            item.get("stripped_deploy_max_abs_difference"),
            field="stripped_deploy_max_abs_difference",
            source=f"{source} epoch{epoch}",
        )
    pairing = value.get("pairing")
    if not isinstance(pairing, dict):
        raise ValueError(f"{source} lacks pairing evidence")
    for field in PAIRING_FIELDS:
        item = pairing.get(field)
        if item is None or item == "" or item == [] or item == {}:
            raise ValueError(f"{source} pairing lacks {field}")
    expected_epoch_keys = {str(epoch) for epoch in TRAINING_GLOBAL_EPOCHS}
    for field in PAIRING_FIELDS[-3:]:
        if not isinstance(pairing[field], dict) or set(pairing[field]) != expected_epoch_keys:
            raise ValueError(f"{source} {field} does not cover global201-232")
    contract = value.get("vm_dice")
    if contract != _expected_formula_contract(arm):
        raise ValueError(f"{source} VM Dice formula contract mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma0-report", type=Path, required=True)
    parser.add_argument("--gamma1-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite VM Dice pairing receipt")
    paths = {
        "paper_dice_gamma0": args.gamma0_report,
        "vm_dice_gamma1": args.gamma1_report,
    }
    reports = {arm: _read(paths[arm], arm) for arm in ARMS}
    mismatched = [
        field
        for field in PAIRING_FIELDS
        if reports[ARMS[0]]["pairing"][field]
        != reports[ARMS[1]]["pairing"][field]
    ]
    if mismatched:
        raise RuntimeError(f"VM Dice paired arms differ: {mismatched}")
    arms: dict[str, Any] = {}
    for arm in ARMS:
        report = reports[arm]
        path = paths[arm]
        arms[arm] = {
            "status": report["status"],
            "gamma": report["gamma"],
            "training_report": str(path.resolve()),
            "training_report_sha256": sha256_file(path),
            "reload_max_abs_difference": report["reload_max_abs_difference"],
            "stripped_deploy_max_abs_difference": report[
                "stripped_deploy_max_abs_difference"
            ],
            "checkpoints": report["checkpoints"],
            "pairing": report["pairing"],
            "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
            "elapsed_seconds": report["elapsed_seconds"],
            "all_self_gates_passed": report["all_self_gates_passed"],
        }
    receipt = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete_before_new_validation_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        "mismatched_fields": mismatched,
        "formal_peak_memory_at_most_24_gib_each": all(
            int(arms[arm]["peak_gpu_memory_bytes"]) <= GPU_MEMORY_STOP_BYTES
            for arm in ARMS
        ),
        "all_gates_passed": not mismatched
        and all(bool(arms[arm]["all_self_gates_passed"]) for arm in ARMS),
        "start_checkpoint_historical_validation_access": True,
        "start_checkpoint_historical_test_access_after_selection": True,
        "validation_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    if receipt["all_gates_passed"] is not True:
        raise RuntimeError("VM Dice final pairing gates failed")
    write_json_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "all_gates_passed": receipt["all_gates_passed"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
