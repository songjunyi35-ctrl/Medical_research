#!/usr/bin/env python3
"""Evaluate the fully frozen epoch-168 plain locator once on internal test270."""
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

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_locator_test270 import (  # noqa: E402
    TEST_CASES,
    TEST_MANIFEST_SHA256,
    TEST_SUBJECTS,
    TestBrainCropDataset,
)
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    read_case_ids,
    sha256_file,
    subject_id,
    validate,
    write_json_atomic,
)
from scripts.train_t1ce_wt_plain_fixed_epoch import CHECKPOINT_SCHEMA  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_fixed_epoch168_test270"
AUTHORIZATION = "USER_AUTHORIZED_PLAIN_FIXED_EPOCH168_TEST270_2026-08-18"
CHECKPOINT_SHA256 = "f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1"
TRAIN_MANIFEST_SHA256 = "a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16"
SELECTOR_MANIFEST_SHA256 = "3710d63842acaccb1e4fe81c5fade578b6f1a306363d969ea2d799edd6c32b07"
INNER_DECISION_SHA256 = "6308c5e3807e513cc0eb3aebf0c4718e29c7e04e5748b314da6f3b28a7b227b4"
OUTER_DECISION_SHA256 = "f8f876e6e04b31300c578c9bd2f235cd1b74948f0139f9810a5c4d67b4711ca7"
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_SAMPLES = 10_000


def bootstrap_interval(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return [float(x) for x in np.quantile(array[draws].mean(axis=1), [0.025, 0.975])]


def intervals(result: dict[str, Any]) -> dict[str, Any]:
    scan_values = [float(case["locator_dsc"]) for case in result["per_case"]]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case in result["per_case"]:
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


def verified_ids(path: Path, expected_sha256: str, expected_count: int) -> list[str]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"role manifest SHA-256 mismatch: {path}")
    values = read_case_ids(path)
    assert values is not None
    if len(values) != expected_count:
        raise RuntimeError(f"role manifest count mismatch: {path}")
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization != AUTHORIZATION:
        raise RuntimeError("explicit epoch168 test270 authorization is required")
    output = args.output.resolve()
    started_path = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started_path.exists():
        raise FileExistsError("refusing a repeated epoch168 test270 evaluation")
    if sha256_file(args.test_manifest) != TEST_MANIFEST_SHA256:
        raise RuntimeError("test manifest SHA-256 mismatch")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("epoch168 checkpoint SHA-256 mismatch")
    if sha256_file(args.outer_decision) != OUTER_DECISION_SHA256:
        raise RuntimeError("outer decision SHA-256 mismatch")
    outer_decision = json.loads(args.outer_decision.read_text())
    if outer_decision.get("all_outer_gates_passed") is not True:
        raise RuntimeError("outer selector did not authorize the development candidate")
    if outer_decision.get("candidate", {}).get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("outer decision selected a different checkpoint")
    if outer_decision.get("test_access") is not False:
        raise RuntimeError("outer decision already recorded test access")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or int(checkpoint.get("epoch", -1)) != 168:
        raise RuntimeError("unexpected fixed-epoch checkpoint")
    if checkpoint.get("inner_decision_sha256") != INNER_DECISION_SHA256:
        raise RuntimeError("checkpoint inner-decision binding mismatch")
    if checkpoint.get("data_role", {}).get("train", {}).get("sha256") != TRAIN_MANIFEST_SHA256:
        raise RuntimeError("checkpoint train-role binding mismatch")
    if any(checkpoint.get(key) is not False for key in (
        "validation_access", "outer_selector_access", "confirmation_access", "test_access"
    )):
        raise RuntimeError("checkpoint access flags are not sealed")

    train_ids = verified_ids(args.train_manifest, TRAIN_MANIFEST_SHA256, 756)
    selector_ids = verified_ids(args.selector_manifest, SELECTOR_MANIFEST_SHA256, 108)
    rows = [json.loads(line) for line in args.test_manifest.read_text().splitlines() if line.strip()]
    case_ids = [str(row["case_id"]) for row in rows]
    row_subjects = [str(row["subject_id"]) for row in rows]
    if len(rows) != TEST_CASES or len(set(case_ids)) != TEST_CASES:
        raise RuntimeError("test manifest case count or uniqueness mismatch")
    if len(set(row_subjects)) != TEST_SUBJECTS:
        raise RuntimeError("test manifest subject count mismatch")
    for row in rows:
        if row.get("split") != "test" or set(row.get("modalities", {})) != {"t1c"}:
            raise RuntimeError("test manifest row violates the frozen T1ce projection")
        if not row.get("seg_path"):
            raise RuntimeError("test manifest row lacks ground truth")
    development_ids = set(train_ids) | set(selector_ids)
    development_subjects = {subject_id(case_id) for case_id in development_ids}
    if development_ids & set(case_ids) or development_subjects & set(row_subjects):
        raise RuntimeError("test overlaps development roles")

    data = TestBrainCropDataset(args.cache, case_ids)
    if data.index.get("manifest_sha256") != TEST_MANIFEST_SHA256:
        raise RuntimeError("test cache is bound to the wrong manifest")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(args.cache / filename) != str(data.index[field]):
            raise RuntimeError(f"test cache {filename} SHA-256 mismatch")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal test270 evaluation requires CUDA")
    torch.cuda.set_device(device)
    architecture = checkpoint["architecture"]
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 4_700_897:
        raise RuntimeError("unexpected locator parameter count")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    write_json_atomic(started_path, {
        "schema": SCHEMA + ".started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_epoch": 168,
        "outer_decision_sha256": OUTER_DECISION_SHA256,
        "test_manifest_sha256": TEST_MANIFEST_SHA256,
        "fixed_logit_threshold": 0.0,
        "checkpoint_selection_uses_test": False,
        "test_result_will_not_trigger_tuning": True,
    })
    loader = DataLoader(
        data,
        batch_size=min(args.batch_size, len(data)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    result = validate(model, loader, device)
    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "fixed final evaluation on a previously revealed internal test set",
        "not_pristine_blind_test": True,
        "not_external_confirmation": True,
        "authorization": AUTHORIZATION,
        "protocol": {
            "model": "plain 4.70M-parameter MONAI SegResNet locator",
            "input": "T1ce only",
            "geometry": "image-derived nonzero brain crop resized to 128^3",
            "threshold": "raw segmentation logit > 0, fixed before test",
            "checkpoint_selection": "epoch168 selected by fresh inner; outer gates passed before test",
            "test_use": "final metrics only; no tuning, threshold choice, or checkpoint selection",
            "tta": False,
            "postprocessing": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": CHECKPOINT_SHA256,
            "selected_epoch": 168,
        },
        "selection_receipt": {
            "outer_decision": str(args.outer_decision.resolve()),
            "outer_decision_sha256": OUTER_DECISION_SHA256,
        },
        "data": {
            "manifest": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": sha256_file(args.cache / "index.json"),
            "cache_images_sha256": data.index["images_sha256"],
            "cache_labels_sha256": data.index["labels_sha256"],
            "cases": TEST_CASES,
            "subjects": TEST_SUBJECTS,
            "case_overlap_with_train_selector": 0,
            "subject_overlap_with_train_selector": 0,
        },
        "summary": result,
        "bootstrap": intervals(result),
        "outer_selector_access_during_test": False,
        "confirmation_access": False,
        "test_access": True,
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "scan_dsc_mean": result["locator_dsc_mean"],
        "subject_dsc_mean": result["locator_subject_dsc_mean"],
        "automatic_point_hit_rate": result["automatic_point_hit_rate"],
        "cases": result["cases"],
        "subjects": result["subjects"],
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outer-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
