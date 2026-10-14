#!/usr/bin/env python3
"""One-time fixed-threshold evaluation of a newly trained plain locator."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import (
    BrainCropDataset,
    CHECKPOINT_SCHEMA,
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_locator_confirmation"
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_SAMPLES = 10_000
SUCCESS_THRESHOLDS = {
    "scan_dsc_mean": 0.68,
    "subject_dsc_mean": 0.67,
    "automatic_point_hit_rate": 0.85,
}


def bootstrap_interval(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    means = array[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def intervals(result: dict[str, Any]) -> dict[str, Any]:
    cases = list(result["per_case"])
    scan_values = [float(case["locator_dsc"]) for case in cases]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_subject[str(case["subject_id"])].append(float(case["locator_dsc"]))
    subject_values = [statistics.fmean(values) for values in by_subject.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {
        "method": "percentile bootstrap",
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "scan_dsc_mean_95ci": bootstrap_interval(scan_values, rng),
        "subject_dsc_mean_95ci": bootstrap_interval(subject_values, rng),
    }


def verified_role_cases(metadata: dict[str, Any] | None, *, role: str) -> list[str]:
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint lacks frozen {role} role metadata")
    path = Path(str(metadata["path"]))
    if sha256_file(path) != str(metadata["sha256"]):
        raise RuntimeError(f"checkpoint-bound {role} role manifest SHA-256 mismatch")
    values = read_case_ids(path)
    assert values is not None
    return values


def load_plain_locator(
    checkpoint_path: Path, device: torch.device
) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unexpected locator checkpoint schema")
    if payload.get("test_access") is not False:
        raise ValueError("test-accessible locator checkpoint is forbidden")
    if payload.get("arm") != "plain":
        raise ValueError("confirmation accepts only the frozen plain locator arm")
    architecture = payload["architecture"]
    if bool(architecture["center_aware"]):
        raise ValueError("plain locator checkpoint unexpectedly has a centre head")
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--confirmation-case-ids", type=Path, required=True)
    parser.add_argument("--locator-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-confirmation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation report: {output}")
    checkpoint = args.locator_checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError("locator checkpoint SHA-256 mismatch")
    confirmation_manifest = args.confirmation_case_ids.resolve()
    confirmation_sha256 = sha256_file(confirmation_manifest)
    if confirmation_sha256 != args.expected_confirmation_sha256:
        raise RuntimeError("confirmation role manifest SHA-256 mismatch")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("confirmation requires CUDA")
    torch.cuda.set_device(device)
    model, payload = load_plain_locator(checkpoint, device)

    role_metadata = payload.get("data_roles", {})
    train_cases = verified_role_cases(role_metadata.get("train_case_ids"), role="train")
    selector_cases = verified_role_cases(
        role_metadata.get("validation_case_ids"), role="selector"
    )
    confirmation_cases = read_case_ids(confirmation_manifest)
    assert confirmation_cases is not None
    known_cases = set(train_cases) | set(selector_cases)
    if known_cases & set(confirmation_cases):
        raise RuntimeError("confirmation cases overlap checkpoint train/selector roles")
    known_subjects = {subject_id(case_id) for case_id in known_cases}
    confirmation_subjects = {subject_id(case_id) for case_id in confirmation_cases}
    if known_subjects & confirmation_subjects:
        raise RuntimeError("confirmation subjects overlap checkpoint train/selector roles")

    started_path = output.with_suffix(output.suffix + ".started.json")
    write_json_atomic(
        started_path,
        {
            "schema": SCHEMA + ".started",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint_sha256,
            "confirmation_manifest_sha256": confirmation_sha256,
            "fixed_logit_threshold": 0.0,
        },
    )
    data = BrainCropDataset(
        args.cache,
        expected_split="train",
        case_ids=confirmation_cases,
    )
    loader = DataLoader(
        data,
        batch_size=min(args.batch_size, len(data)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    result = validate(model, loader, device)
    observed = {
        "scan_dsc_mean": float(result["locator_dsc_mean"]),
        "subject_dsc_mean": float(result["locator_subject_dsc_mean"]),
        "automatic_point_hit_rate": float(result["automatic_point_hit_rate"]),
    }
    gates = {
        key: {
            "threshold": threshold,
            "observed": observed[key],
            "passed": observed[key] >= threshold,
        }
        for key, threshold in SUCCESS_THRESHOLDS.items()
    }
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": (
            "one-time internal subject-clean protocol replication on cases excluded "
            "from this checkpoint's train and selector roles"
        ),
        "not_external_confirmation": True,
        "test_access": False,
        "historical_108_validation_access": False,
        "270_case_test_access": False,
        "protocol": {
            "model": "plain 4.70M-parameter MONAI SegResNet locator",
            "input": "T1ce only",
            "geometry": "image-derived nonzero brain crop resized to 128^3",
            "threshold": "raw segmentation logit > 0, fixed before confirmation",
            "checkpoint_selection": "highest selector scan-macro Dice",
            "confirmation_use": "metrics only; no tuning or checkpoint selection",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "selected_epoch": int(payload["epoch"]),
        },
        "data": {
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "confirmation_manifest": str(confirmation_manifest),
            "confirmation_manifest_sha256": confirmation_sha256,
            "cases": len(confirmation_cases),
            "subjects": len(confirmation_subjects),
            "case_overlap_with_train_selector": 0,
            "subject_overlap_with_train_selector": 0,
        },
        "summary": result,
        "bootstrap": intervals(result),
        "preregistered_gates": gates,
        "all_gates_passed": all(value["passed"] for value in gates.values()),
    }
    write_json_atomic(output, report)
    print(json.dumps({**observed, "all_gates_passed": report["all_gates_passed"]}, indent=2))


if __name__ == "__main__":
    main()
