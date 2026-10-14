#!/usr/bin/env python3
"""Freeze a completed FV-LoRA run, then evaluate the authorized 270-case test once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import torch


SCHEMA = "aflow.sammed3d.t1ce_wt.fv_lora_auto_test_watcher"
SCHEMA_VERSION = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def process_command(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return None


def verify_training_process(pid: int, output_dir: Path) -> bool:
    command = process_command(pid)
    if command is None:
        return False
    expected_output_name = output_dir.name
    if (
        "train_sammed3d_t1ce_wt.py" not in command
        or expected_output_name not in command
    ):
        raise RuntimeError(f"PID {pid} was reused or is not the frozen trainer: {command}")
    return True


def require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{field} mismatch: {actual!r} != {expected!r}")


def verify_completed_training(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.training_output.resolve()
    prereg = load_json(args.prereg.resolve())
    require_equal(prereg.get("status"), "frozen_before_full_training", "prereg.status")
    require_equal(prereg.get("output_dir"), args.training_output.as_posix(), "prereg.output_dir")

    source = prereg["source"]
    for relative_key, hash_key in (
        ("trainer", "trainer_sha256"),
        ("preprocessing_module", "preprocessing_module_sha256"),
    ):
        source_path = (args.root / source[relative_key]).resolve()
        require_equal(sha256_file(source_path), source[hash_key], hash_key)

    paths = {
        "protocol": output_dir / "t1ce_wt_protocol.json",
        "train_config": output_dir / "train_config.json",
        "best": output_dir / "best.json",
        "latest": output_dir / "sam_model_latest.pth",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"completed training lacks {name}: {path}")
    if args.training_log.is_file() and "TRAIN_DONE" not in args.training_log.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise RuntimeError("training process exited without TRAIN_DONE")

    protocol = load_json(paths["protocol"])
    train_config = load_json(paths["train_config"])
    best = load_json(paths["best"])
    require_equal(
        protocol["preprocessing"]["geometry"]["name"],
        "whole_resize",
        "protocol.preprocessing.geometry",
    )
    require_equal(
        protocol["preprocessing"]["augmentation"]["name"],
        "affine_elastic_noise",
        "protocol.preprocessing.augmentation",
    )
    require_equal(protocol["test_access_during_training"], False, "test access")
    require_equal(train_config["train_rows"], 972, "train rows")
    require_equal(train_config["val_rows"], 108, "validation rows")
    require_equal(train_config["trainable_parameters"], 7731760, "trainable parameters")
    config_args = train_config["args"]
    expected_args = {
        "epochs": 50,
        "strategy": "decoder_only",
        "encoder_qv_lora_rank": 4,
        "encoder_qv_lora_alpha": 8.0,
        "num_clicks": 3,
        "click_method": "random",
        "max_train_cases": 0,
        "max_val_cases": 0,
        "seed": 20260725,
        "val_seed": 20260723,
        "selection_metric": "val_dice",
    }
    for key, expected in expected_args.items():
        require_equal(config_args.get(key), expected, f"train_config.args.{key}")

    latest = torch.load(paths["latest"], map_location="cpu", weights_only=False)
    require_equal(latest.get("epoch"), 50, "latest.epoch")
    require_equal(len(latest.get("history", [])), 50, "latest.history length")
    require_equal(latest.get("best"), best, "latest.best")
    adapter = latest.get("adapter_config", {}).get("encoder_qv_lora", {})
    require_equal(adapter.get("enabled"), True, "latest adapter enabled")
    require_equal(adapter.get("rank"), 4, "latest adapter rank")
    require_equal(adapter.get("alpha"), 8.0, "latest adapter alpha")

    selection = best.get("selection", {})
    require_equal(selection.get("metric"), "val_dice", "best selection metric")
    checkpoint_name = selection.get("checkpoint")
    if checkpoint_name != "sam_model_val_dice_best.pth":
        raise RuntimeError(f"unexpected selected checkpoint: {checkpoint_name}")
    checkpoint = output_dir / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_equal(selected.get("epoch"), selection.get("epoch"), "selected epoch")
    metadata = selected.get("checkpoint_metadata", {})
    require_equal(metadata.get("kind"), "selected_best", "selected kind")
    require_equal(metadata.get("metric"), "val_dice", "selected metric")
    require_equal(metadata.get("metric_epoch"), selection.get("epoch"), "selected metric epoch")
    require_equal(selected.get("adapter_config"), latest.get("adapter_config"), "adapter config")

    return {
        "prereg": prereg,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "hashes": {key: sha256_file(value) for key, value in paths.items()},
        "best": best,
        "checkpoint": checkpoint.resolve(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "adapter_config": selected["adapter_config"],
    }


def run(args: argparse.Namespace) -> None:
    state = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "waiting_for_training",
        "created_at_utc": now_utc(),
        "training_pid": args.training_pid,
        "training_output": str(args.training_output.resolve()),
        "test_manifest_path_frozen_but_not_opened": str(args.test_manifest.resolve()),
        "expected_test_manifest_sha256": args.expected_test_manifest_sha256,
    }
    if any(path.exists() for path in (args.freeze_output, args.test_output, args.final_output)):
        raise FileExistsError("freeze, test report, or final binding already exists")
    atomic_json(args.state_output, state)
    while verify_training_process(args.training_pid, args.training_output.resolve()):
        state["last_training_alive_at_utc"] = now_utc()
        atomic_json(args.state_output, state)
        time.sleep(args.poll_seconds)

    state["training_exited_at_utc"] = now_utc()
    state["status"] = "verifying_completed_training"
    atomic_json(args.state_output, state)
    verified = verify_completed_training(args)

    evaluator = args.evaluator.resolve()
    evaluator_sha256 = sha256_file(evaluator)
    freeze = {
        "schema": "aflow.sammed3d.t1ce_wt.fv_lora_test_freeze",
        "schema_version": 1,
        "status": "frozen_before_test_manifest_open",
        "created_at_utc": now_utc(),
        "training": {
            "output_dir": str(args.training_output.resolve()),
            "artifacts": verified["paths"],
            "artifact_sha256": verified["hashes"],
            "best": verified["best"],
        },
        "model": {
            "checkpoint": str(verified["checkpoint"]),
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "adapter_config": verified["adapter_config"],
        },
        "test": {
            "manifest": str(args.test_manifest.resolve()),
            "expected_manifest_sha256": args.expected_test_manifest_sha256,
            "expected_split": "test",
            "cases": 270,
            "seed": 20260723,
            "spatial_preprocessing": "whole_resize",
            "threshold": "raw logit > 0",
            "num_clicks": 3,
            "output": str(args.test_output.resolve()),
        },
        "evaluator": {
            "path": str(evaluator),
            "sha256": evaluator_sha256,
        },
        "selection_used_test": False,
    }
    atomic_json(args.freeze_output, freeze)
    freeze_sha256 = sha256_file(args.freeze_output)
    state.update(
        status="evaluating_test",
        freeze_output=str(args.freeze_output.resolve()),
        freeze_sha256=freeze_sha256,
        evaluation_started_at_utc=now_utc(),
    )
    atomic_json(args.state_output, state)

    command = [
        sys.executable,
        "-u",
        str(evaluator),
        "--manifest",
        str(args.test_manifest.resolve()),
        "--expected-manifest-sha256",
        args.expected_test_manifest_sha256,
        "--expected-split",
        "test",
        "--checkpoint",
        str(verified["checkpoint"]),
        "--expected-checkpoint-sha256",
        verified["checkpoint_sha256"],
        "--spatial-preprocessing",
        "whole_resize",
        "--device",
        args.device,
        "--seed",
        "20260723",
        "--output",
        str(args.test_output.resolve()),
    ]
    subprocess.run(command, cwd=args.root, check=True)
    report = load_json(args.test_output)
    require_equal(report["data"]["selected_cases"], 270, "test selected cases")
    require_equal(report["data"]["full_manifest_cases"], 270, "test manifest cases")
    require_equal(report["protocol"]["spatial_preprocessing"], "whole_resize", "test geometry")
    require_equal(report["model"]["checkpoint_sha256"], verified["checkpoint_sha256"], "test checkpoint")
    final = {
        "schema": "aflow.sammed3d.t1ce_wt.fv_lora_auto_test_result",
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": now_utc(),
        "freeze": str(args.freeze_output.resolve()),
        "freeze_sha256": freeze_sha256,
        "test_report": str(args.test_output.resolve()),
        "test_report_sha256": sha256_file(args.test_output),
        "checkpoint": str(verified["checkpoint"]),
        "checkpoint_sha256": verified["checkpoint_sha256"],
        "summary": report["summary"],
    }
    atomic_json(args.final_output, final)
    state.update(
        status="complete",
        completed_at_utc=now_utc(),
        final_output=str(args.final_output.resolve()),
        test_report_sha256=final["test_report_sha256"],
    )
    atomic_json(args.state_output, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--training-output", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--expected-test-manifest-sha256", required=True)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--freeze-output", required=True, type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--final-output", required=True, type=Path)
    parser.add_argument("--state-output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    try:
        run(args)
    except BaseException as exc:
        failure = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failed_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(args.state_output, failure)
        raise


if __name__ == "__main__":
    main()
