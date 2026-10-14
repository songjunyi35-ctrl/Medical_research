#!/usr/bin/env python3
"""Freeze four paired checkpoints before either C/D inner is opened."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_t1ce_wt_flip_equivariance_consistency_inner as evaluate  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402


SCHEMA = "aflow.brats.flip_equivariance_consistency.inner_preaccess"


def freeze(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preaccess receipt: {output}")
    split = evaluate.load_split(args.split_manifest.resolve())
    reports = evaluate.preflight_training_artifacts(args.work_dir.resolve())
    if sha256_file(args.cache / "index.json") != evaluate.prior_cache_sha():
        raise ValueError("train cache index mismatch")
    artifacts = {}
    pairing = {}
    for replicate, pair in reports.items():
        pairing[replicate] = evaluate.pairing_gates(pair)
        if not pairing[replicate]["all_passed"]:
            raise ValueError(f"replicate {replicate} pairing gate failed")
        artifacts[replicate] = {}
        for arm, report in pair.items():
            report_path = args.work_dir.resolve() / f"formal_{replicate}_{arm}/TRAIN_DONE.json"
            artifacts[replicate][arm] = {
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "checkpoint_path": report["checkpoint"]["path"],
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "model_state_sha256": report["checkpoint"]["model_state_sha256"],
                "initial_state_sha256": report["initial_state_sha256"],
                "sampling_weight_sha256": report["sampling_weight_sha256"],
                "consistency_weight": report["consistency_weight"],
            }
    receipt = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_four_trainings_before_inner_access",
        "fixed_epoch": evaluate.train.FORMAL_EPOCHS,
        "training_artifacts": artifacts,
        "pairing_gates": pairing,
        "train_cache": {"path": str(args.cache.resolve()), "index_sha256": evaluate.prior_cache_sha()},
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": evaluate.train.SPLIT_MANIFEST_SHA256,
        "inner_manifests": {replicate: split["replications"][replicate]["inner"] for replicate in evaluate.train.REPLICATES},
        "evaluator": str(Path(evaluate.__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(evaluate.__file__).resolve()),
        "bootstrap_samples": evaluate.BOOTSTRAP_SAMPLES,
        "gates": "per IDEA/PREREG/AMENDMENT_01; no post-training change",
        "documents": {"idea_sha256": evaluate.train.IDEA_SHA256, "prereg_sha256": evaluate.train.PREREG_SHA256, "amendment_sha256": evaluate.train.AMENDMENT_SHA256},
        "access": {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
    }
    write_json_atomic(output, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    receipt = freeze(parser.parse_args())
    print(json.dumps({"status": receipt["status"], "replicates": sorted(receipt["training_artifacts"]), "inner_metrics_access": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
