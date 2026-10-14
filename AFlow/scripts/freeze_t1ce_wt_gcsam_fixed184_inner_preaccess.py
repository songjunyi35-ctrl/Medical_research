#!/usr/bin/env python3
"""Freeze all four replication artifacts before either train-only inner is opened."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_t1ce_wt_gcsam_fixed184_replication_inner as evaluate
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic


def freeze(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preaccess receipt: {output}")
    preflight = evaluate.preflight_training_artifacts(args.work_dir.resolve())
    artifacts = {}
    for key, report in preflight["reports"].items():
        report_path = args.work_dir.resolve() / f"formal_{key}/TRAIN_DONE.json"
        artifacts[key] = {
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "checkpoint_path": report["checkpoint"]["path"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "model_state_sha256": report["checkpoint"]["model_state_sha256"],
            "initial_state_sha256": report["initial_state_sha256"],
            "sampling_weight_sha256": report["sampling_weight_sha256"],
        }
    receipt = {
        "schema": "aflow.brats.gcsam_fixed184_replication.inner_preaccess",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_four_trainings_before_inner_access",
        "fixed_epoch": 184,
        "training_artifacts": artifacts,
        "pairing": preflight["pairing"],
        "inner_manifests": {replicate: {"path": str(value["path"]), "sha256": value["sha256"], "cases": 108, "subjects": value["subjects"]} for replicate, value in evaluate.INNER.items()},
        "evaluator": str(Path(evaluate.__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(evaluate.__file__).resolve()),
        "bootstrap_samples": evaluate.BOOTSTRAP_SAMPLES,
        "gates": "per IDEA/PREREG; no post-training change",
        "access": {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
    }
    write_json_atomic(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args)
    print(json.dumps({"status": receipt["status"], "training_artifacts": sorted(receipt["training_artifacts"]), "pairing": receipt["pairing"], "inner_metrics_access": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
