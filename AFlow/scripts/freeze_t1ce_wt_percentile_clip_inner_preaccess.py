#!/usr/bin/env python3
"""Freeze all four percentile-clipping artifacts before either inner is opened."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.evaluate_t1ce_wt_percentile_clip_replication_inner as evaluate  # noqa: E402
from scripts.train_t1ce_wt_capg3d import sha256_file, write_json_atomic  # noqa: E402


def freeze(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preaccess receipt: {output}")
    split = evaluate.load_split(args.split_manifest.resolve())
    preflight = evaluate.preflight_training_artifacts(args.work_dir.resolve())
    caches = {"control": args.control_cache.resolve(), "percentile_clip": args.candidate_cache.resolve()}
    cache_artifacts = {}
    for arm, cache in caches.items():
        index = json.loads((cache / "index.json").read_text())
        cache_artifacts[arm] = {
            "path": str(cache),
            "index_sha256": sha256_file(cache / "index.json"),
            "images_sha256": index["images_sha256"],
            "labels_sha256": index["labels_sha256"],
        }
    if not (
        cache_artifacts["control"]["labels_sha256"]
        == cache_artifacts["percentile_clip"]["labels_sha256"]
        == evaluate.train.CONTROL_LABELS_SHA256
    ):
        raise ValueError("control/candidate labels are not byte-exact")
    if cache_artifacts["control"]["images_sha256"] == cache_artifacts["percentile_clip"]["images_sha256"]:
        raise ValueError("candidate images unexpectedly equal control")
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
        "schema": "aflow.brats.percentile_clip_replication.inner_preaccess",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_four_trainings_before_inner_access",
        "fixed_epoch": evaluate.train.FORMAL_EPOCHS,
        "training_artifacts": artifacts,
        "pairing": preflight["pairing"],
        "caches": cache_artifacts,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "inner_manifests": {replicate: split["replications"][replicate]["inner"] for replicate in ("a", "b")},
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
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    receipt = freeze(parser.parse_args())
    print(json.dumps({
        "status": receipt["status"],
        "training_artifacts": sorted(receipt["training_artifacts"]),
        "pairing": receipt["pairing"],
        "inner_metrics_access": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
