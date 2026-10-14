#!/usr/bin/env python3
"""Freeze the model-performance preaccess receipt after all GBM334 input gates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import openneuro_gbm334_protocol as p


def main() -> None:
    output = p.RECORD / "MODEL_PREACCESS.json"
    if output.exists():
        raise FileExistsError(output)
    p.docs()
    first = p.RECORD / "FIRST_CASE_PROBE.json"
    preflight = p.RECORD / "INPUT_PREFLIGHT.json"
    smoke = p.RECORD / "SMOKE_RECEIPT.json"
    evaluator = p.ROOT / "scripts/evaluate_openneuro_gbm334_panel.py"
    for path in (first, preflight, smoke, evaluator, p.FULL, p.GCSAM):
        if not path.exists():
            raise FileNotFoundError(path)
    first_data = json.loads(first.read_text())
    preflight_data = json.loads(preflight.read_text())
    smoke_data = json.loads(smoke.read_text())
    if (
        first_data["status"] != "passed"
        or not all(first_data["gates"].values())
        or preflight_data["status"] != "passed"
        or not all(preflight_data["gates"].values())
        or smoke_data["status"] != "passed"
        or not smoke_data["synthetic_only"]
        or smoke_data["openneuro_payload_access"]
        or p.sha256(p.FULL) != p.FULL_SHA
        or p.sha256(p.GCSAM) != p.GCSAM_SHA
    ):
        raise ValueError("preaccess prerequisite")
    receipt = {
        "schema": "aflow.openneuro_gbm334.model_preaccess",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_all_input_gates_before_external_model",
        "snapshot": "ds007045:1.0.1",
        "cases": p.CASES,
        "expected_forwards": p.CASES * 9,
        "idea_sha256": p.IDEA_SHA,
        "prereg_sha256": p.PREREG_SHA,
        "source_inventory_sha256": p.INVENTORY_SHA,
        "manifest_sha256": p.MANIFEST_SHA,
        "first_case_probe_sha256": p.sha256(first),
        "input_preflight_sha256": p.sha256(preflight),
        "smoke_receipt_sha256": p.sha256(smoke),
        "evaluator_sha256": p.sha256(evaluator),
        "checkpoint_sha256": {"full972_kd168": p.FULL_SHA, "gcsam184": p.GCSAM_SHA},
        "views": {
            "full972_kd168": ["identity", "fixed_flip8_equal_raw_logit_mean"],
            "gcsam184": ["identity"],
        },
        "target": "WT=labels {1,2,3}",
        "threshold": 0.0,
        "bootstrap_samples": p.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": p.BOOTSTRAP_SEED,
        "streaming": {
            "persistent_source_or_cache": False,
            "each_case_redownloaded_from_same_versionId_and_sha": True,
            "performance_boundary_progress_receipt_each_case": True,
        },
        "access": {
            "openneuro_source_all_334": True,
            "external_performance": False,
            "checkpoint_loaded_for_synthetic_smoke_only": True,
            "internal_test270": False,
            "fixed_validation108": False,
            "outer": False,
            "confirmation": False,
            "other_external_cohort": False,
        },
        "runner_sha256": p.sha256(Path(__file__)),
    }
    p.write_new(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
