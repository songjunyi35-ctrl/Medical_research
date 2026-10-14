#!/usr/bin/env python3
"""Independently recompute and archive the OpenNeuro-GBM334 raw result."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import scripts.evaluate_brats_africa95_frozen_baselines as metrics
from scripts import openneuro_gbm334_protocol as p


metrics.p = p


def main() -> None:
    p.docs()
    raw_path = p.RECORD / "RAW_OPENNEURO_GBM334.json"
    started_path = p.RECORD / "RAW_OPENNEURO_GBM334_STARTED.json"
    progress_path = p.RECORD / "RAW_OPENNEURO_GBM334_PROGRESS.json"
    preaccess_path = p.RECORD / "MODEL_PREACCESS.json"
    preflight_path = p.RECORD / "INPUT_PREFLIGHT.json"
    result_path = p.RECORD / "RESULT.json"
    verification_path = p.RECORD / "VERIFICATION_RECEIPT.json"
    if result_path.exists() or verification_path.exists():
        raise FileExistsError("final outputs already exist")
    for path in (raw_path, started_path, progress_path, preaccess_path, preflight_path):
        if not path.exists():
            raise FileNotFoundError(path)
    raw = json.loads(raw_path.read_text())
    progress = json.loads(progress_path.read_text())
    started = json.loads(started_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    identity = metrics.summarize(raw["full972_kd168"]["identity"]["per_case"])
    flip8 = metrics.summarize(raw["full972_kd168"]["flip8"]["per_case"])
    gcsam = metrics.summarize(raw["gcsam184"]["identity"]["per_case"])
    recomputed = {
        "full_identity_absolute": metrics.absolute(identity, p.BOOTSTRAP_SEED),
        "full_flip8_absolute": metrics.absolute(flip8, p.BOOTSTRAP_SEED + 10),
        "gcsam_identity_absolute": metrics.absolute(gcsam, p.BOOTSTRAP_SEED + 20),
        "full_flip8_minus_identity": metrics.compare(
            identity, flip8, p.BOOTSTRAP_SEED + 100, "flip8_minus_identity"
        ),
        "gcsam_identity_minus_full_identity": metrics.compare(
            identity, gcsam, p.BOOTSTRAP_SEED + 200, "gcsam_minus_full"
        ),
    }
    gates = {
        "raw_schema": raw["schema"] == "aflow.openneuro_gbm334.frozen_checkpoint_panel",
        "raw_case_counts": all(
            summary["cases"] == p.CASES
            for summary in (identity, flip8, gcsam)
        ),
        "raw_unique_case_order": [x["case_id"] for x in identity["per_case"]]
        == [x["case_id"] for x in flip8["per_case"]]
        == [x["case_id"] for x in gcsam["per_case"]],
        "forward_count": raw["forwards"] == p.CASES * 9 == 3006,
        "progress_terminal": progress["completed_cases"] == p.CASES
        and progress["model_forwards"] == 3006
        and progress["started_sha256"] == p.sha256(started_path),
        "preflight_passed": preflight["status"] == "passed"
        and all(preflight["gates"].values()),
        "checkpoints_exact": p.sha256(p.FULL) == p.FULL_SHA
        and p.sha256(p.GCSAM) == p.GCSAM_SHA,
        "absolute_bootstrap_exact": recomputed["full_identity_absolute"]
        == raw["full972_kd168"]["identity_bootstrap"]
        and recomputed["full_flip8_absolute"]
        == raw["full972_kd168"]["flip8_bootstrap"]
        and recomputed["gcsam_identity_absolute"]
        == raw["gcsam184"]["identity_bootstrap"],
        "paired_bootstrap_exact": recomputed["full_flip8_minus_identity"]
        == raw["comparisons"]["full_flip8_minus_identity"]
        and recomputed["gcsam_identity_minus_full_identity"]
        == raw["comparisons"]["gcsam_identity_minus_full_identity"],
        "no_other_cohort_access": all(
            not value for key, value in raw["access"].items() if key != "openneuro_gbm334"
        ),
        "streaming_no_persistent_cache": raw["data"]["persistent_source_or_cache_created"]
        is False,
    }
    if not all(gates.values()):
        raise RuntimeError(gates)
    result = {
        "schema": "aflow.openneuro_gbm334.result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_once_and_closed",
        "scope": raw["evidence_scope"],
        "cases": p.CASES,
        "subjects": p.CASES,
        "metrics": {
            "full_identity": {
                "scan_dice": identity["locator_dsc_mean"],
                "subject_dice": identity["locator_subject_dsc_mean"],
                "point_hits": identity["automatic_point_hits"],
                "dice_95ci": recomputed["full_identity_absolute"]["scan_dsc_mean_95ci"],
                "forwards": 1,
            },
            "full_flip8": {
                "scan_dice": flip8["locator_dsc_mean"],
                "subject_dice": flip8["locator_subject_dsc_mean"],
                "point_hits": flip8["automatic_point_hits"],
                "dice_95ci": recomputed["full_flip8_absolute"]["scan_dsc_mean_95ci"],
                "forwards": 8,
            },
            "gcsam_identity": {
                "scan_dice": gcsam["locator_dsc_mean"],
                "subject_dice": gcsam["locator_subject_dsc_mean"],
                "point_hits": gcsam["automatic_point_hits"],
                "dice_95ci": recomputed["gcsam_identity_absolute"]["scan_dsc_mean_95ci"],
                "forwards": 1,
            },
        },
        "comparisons": raw["comparisons"],
        "interpretation_gates": raw["interpretation_gates"],
        "cost": {
            "training": 0,
            "formal_forwards": raw["forwards"],
            "formal_elapsed_seconds": raw["elapsed_seconds"],
            "peak_gpu_memory_bytes": raw["peak_gpu_memory_bytes"],
            "source_bytes_streamed_preflight": preflight["source_bytes_streamed"],
            "source_bytes_streamed_formal": raw["data"]["source_bytes_restreamed"],
            "persistent_cache_bytes": 0,
        },
        "access": raw["access"],
        "hashes": {
            "raw_sha256": p.sha256(raw_path),
            "started_sha256": p.sha256(started_path),
            "progress_sha256": p.sha256(progress_path),
            "preaccess_sha256": p.sha256(preaccess_path),
            "input_preflight_sha256": p.sha256(preflight_path),
            "source_inventory_sha256": p.INVENTORY_SHA,
            "full_checkpoint_sha256": p.FULL_SHA,
            "gcsam_checkpoint_sha256": p.GCSAM_SHA,
            "evaluator_sha256": raw["evaluator_sha256"],
            "finalizer_sha256": p.sha256(Path(__file__)),
        },
        "decision": "close cohort; no tuning, selection, exclusions, or repeat",
    }
    p.write_new(result_path, result)
    verification = {
        "schema": "aflow.openneuro_gbm334.verification",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "gates": gates,
        "result_sha256": p.sha256(result_path),
        "raw_sha256": p.sha256(raw_path),
        "input_preflight_sha256": p.sha256(preflight_path),
        "source_inventory_sha256": p.INVENTORY_SHA,
        "full_checkpoint_sha256": p.FULL_SHA,
        "gcsam_checkpoint_sha256": p.GCSAM_SHA,
        "finalizer_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(verification_path, verification)
    print(json.dumps({"status": "passed", "metrics": result["metrics"], "comparisons": result["comparisons"], "hashes": result["hashes"]}, indent=2))


if __name__ == "__main__":
    main()
