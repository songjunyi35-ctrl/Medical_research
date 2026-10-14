#!/usr/bin/env python3
"""Freeze the one-run Flip8 test270 contract after both fresh-inner gates pass."""
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

import scripts.evaluate_t1ce_wt_flip8_tta_replication_inner as inner  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402


SCHEMA = "aflow.brats.flip8_tta.test270_preaccess"
CHECKPOINT_SHA256 = "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
IDENTITY_REPORT_SHA256 = "9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd"
TEST_MANIFEST_SHA256 = "a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9"
TEST_CACHE_INDEX_SHA256 = "41bd6cf44553b8d156de49f2c57bb424cab60cfa26c06881bdfb96ff91de39d3"
EXPECTED_IDENTITY = {
    "scan_dice": 0.7517595372128266,
    "subject_dice": 0.7373271071260759,
    "point_hits": 262,
    "cases": 270,
    "subjects": 119,
}
CURRENT_SCAN_LEADER = 0.752003176
PAIRED_BOOTSTRAP_SEED = 20261207
BOOTSTRAP_SAMPLES = 10_000


def validate_stage_a(value: dict[str, Any]) -> None:
    if value.get("schema") != inner.SCHEMA or value.get("status") != "complete":
        raise ValueError("Stage A schema/status mismatch")
    if value.get("test270_authorized") is not True:
        raise ValueError("Stage A did not authorize test270")
    expected_access = {
        "fit": True,
        "inner_metrics": True,
        "fixed_validation108": False,
        "outer": False,
        "confirmation": False,
        "test270": False,
        "ucsd_external75": False,
    }
    if value.get("access") != expected_access:
        raise ValueError("Stage A access boundary mismatch")
    if value.get("evaluator_sha256") != sha256_file(Path(inner.__file__).resolve()):
        raise ValueError("Stage A evaluator SHA mismatch")
    replications = value.get("replications", {})
    if set(replications) != {"a", "b"}:
        raise ValueError("Stage A replication set mismatch")
    if not all(replications[key].get("gates", {}).get("all_passed") is True for key in ("a", "b")):
        raise ValueError("a per-replication Stage A gate failed")
    expected_cross = {
        "mean_scan_delta_exceeds_current_leader_margin": True,
        "mean_subject_delta_positive": True,
    }
    if value.get("cross_replication", {}).get("gates") != expected_cross:
        raise ValueError("a cross-replication Stage A gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-report", type=Path, required=True)
    parser.add_argument("--inner-preaccess", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    test_output = args.test_output.resolve()
    started = test_output.with_suffix(test_output.suffix + ".started.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pretest receipt: {output}")
    if test_output.exists() or started.exists():
        raise FileExistsError("test270 was already started or completed")

    fixed_hashes = {
        args.checkpoint.resolve(): CHECKPOINT_SHA256,
        args.identity_report.resolve(): IDENTITY_REPORT_SHA256,
        args.test_manifest.resolve(): TEST_MANIFEST_SHA256,
        (args.test_cache.resolve() / "index.json"): TEST_CACHE_INDEX_SHA256,
    }
    for path, expected in fixed_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen SHA-256 mismatch: {path}")

    stage_a_path = args.stage_a_report.resolve()
    stage_a = json.loads(stage_a_path.read_text())
    validate_stage_a(stage_a)
    preaccess_path = args.inner_preaccess.resolve()
    preaccess = json.loads(preaccess_path.read_text())
    if (
        preaccess.get("schema") != "aflow.brats.flip8_tta_replication.inner_preaccess"
        or preaccess.get("status") != "frozen_after_two_trainings_before_inner_access"
        or preaccess.get("access", {}).get("test270") is not False
    ):
        raise ValueError("invalid Stage A inner preaccess receipt")

    evaluator = args.evaluator.resolve()
    receipt = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_stage_a_pass_before_test270_access",
        "authorization": "user-authorized leaderboard objective after all preregistered Stage A gates passed",
        "evidence_label": "post-test development; reused internal test; not pristine blind; not external",
        "one_run_limit": True,
        "stage_a": {
            "report_path": str(stage_a_path),
            "report_sha256": sha256_file(stage_a_path),
            "inner_preaccess_path": str(preaccess_path),
            "inner_preaccess_sha256": sha256_file(preaccess_path),
            "test270_authorized": True,
        },
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": CHECKPOINT_SHA256, "epoch": 168},
        "archived_identity_report": {
            "path": str(args.identity_report.resolve()),
            "sha256": IDENTITY_REPORT_SHA256,
            "expected": EXPECTED_IDENTITY,
            "required_parity": "exact full summary and per-case equality before candidate publication",
        },
        "test270": {
            "manifest_path": str(args.test_manifest.resolve()),
            "manifest_sha256": TEST_MANIFEST_SHA256,
            "cache_path": str(args.test_cache.resolve()),
            "cache_index_sha256": TEST_CACHE_INDEX_SHA256,
            "cases": 270,
            "subjects": 119,
        },
        "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        "test_output": str(test_output),
        "tta": {
            "axes": [list(axes) for axes in inner.REFLECTION_AXES],
            "aggregation": "ordered float32 mean of inverse-reflected raw logits",
            "forwards_per_case": 8,
            "identity_is_first_atom": True,
            "batch_size": 8,
            "threshold": 0.0,
            "point": "argmax mean raw logit",
            "sam": False,
            "postprocessing": False,
        },
        "bootstrap": {
            "method": "percentile bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "absolute_seed": 20260817,
            "paired_seed": PAIRED_BOOTSTRAP_SEED,
        },
        "success_gates": {
            "flip8_scan_strictly_above_current_leader": CURRENT_SCAN_LEADER,
            "flip8_scan_strictly_above_identity": EXPECTED_IDENTITY["scan_dice"],
            "flip8_subject_at_least_identity": EXPECTED_IDENTITY["subject_dice"],
            "flip8_point_hits_at_least": EXPECTED_IDENTITY["point_hits"] - 1,
        },
        "result_must_not_trigger_view_weight_threshold_checkpoint_or_postprocessing_tuning": True,
        "access": {
            "fit": True,
            "inner_metrics": True,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "test270": False,
            "ucsd_external75": False,
        },
    }
    write_json_atomic(output, receipt)
    print(json.dumps({"status": receipt["status"], "test_access": False, "test_output": str(test_output)}, indent=2))


if __name__ == "__main__":
    main()
