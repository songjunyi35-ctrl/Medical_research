#!/usr/bin/env python3
"""Freeze both base checkpoints before either fresh Flip8 inner is opened."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_t1ce_wt_flip8_tta_replication_inner as evaluate  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402


def freeze(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preaccess receipt: {output}")
    split = evaluate.load_split(args.split_manifest.resolve())
    reports = evaluate.preflight_training_artifacts(args.work_dir.resolve())
    if sha256_file(args.cache / "index.json") != evaluate.train.CONTROL_CACHE_INDEX_SHA256:
        raise ValueError("train cache index mismatch")
    artifacts = {}
    for replicate, report in reports.items():
        report_path = args.work_dir.resolve() / f"formal_{replicate}_base/TRAIN_DONE.json"
        artifacts[replicate] = {
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "checkpoint_path": report["checkpoint"]["path"],
            "checkpoint_sha256": report["checkpoint"]["sha256"],
            "model_state_sha256": report["checkpoint"]["model_state_sha256"],
            "initial_state_sha256": report["initial_state_sha256"],
            "sampling_weight_sha256": report["sampling_weight_sha256"],
        }
    receipt = {
        "schema": "aflow.brats.flip8_tta_replication.inner_preaccess",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_two_trainings_before_inner_access",
        "fixed_epoch": evaluate.train.FORMAL_EPOCHS,
        "training_artifacts": artifacts,
        "train_cache": {"path": str(args.cache.resolve()), "index_sha256": evaluate.train.CONTROL_CACHE_INDEX_SHA256},
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "inner_manifests": {replicate: split["replications"][replicate]["inner"] for replicate in ("a", "b")},
        "evaluator": str(Path(evaluate.__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(evaluate.__file__).resolve()),
        "reflection_axes": [list(axes) for axes in evaluate.REFLECTION_AXES],
        "bootstrap_samples": evaluate.BOOTSTRAP_SAMPLES,
        "leader_margin": evaluate.LEADER_MARGIN,
        "gates": "per IDEA/PREREG; no post-training change",
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
    print(json.dumps({"status": receipt["status"], "training_artifacts": sorted(receipt["training_artifacts"]), "inner_metrics_access": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
