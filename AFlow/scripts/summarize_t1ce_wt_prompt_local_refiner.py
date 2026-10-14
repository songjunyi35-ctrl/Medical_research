#!/usr/bin/env python3
"""Write the development-only result ledger for the prompt-local refiner."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_cached import MODEL_SHAPE  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_id(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def subject_macro(case_ids: list[str], values: np.ndarray) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for case_id, value in zip(case_ids, values, strict=True):
        groups[subject_id(case_id)].append(float(value))
    return float(np.mean([np.mean(group) for group in groups.values()]))


def volume_rows(volumes: np.ndarray, coarse: np.ndarray, refined: np.ndarray) -> list[dict[str, Any]]:
    definitions = (
        ("<8000", volumes < 8000),
        ("8000-19999", (volumes >= 8000) & (volumes < 20000)),
        ("20000-39999", (volumes >= 20000) & (volumes < 40000)),
        (">=40000", volumes >= 40000),
    )
    rows = []
    for name, selected in definitions:
        rows.append(
            {
                "bin": name,
                "cases": int(selected.sum()),
                "global_dsc": float(coarse[selected].mean()),
                "prompt_local_dsc": float(refined[selected].mean()),
                "gain": float((refined[selected] - coarse[selected]).mean()),
                "improved_cases": int(np.sum(refined[selected] > coarse[selected])),
            }
        )
    return rows


def bootstrap_ci(delta: np.ndarray, seed: int, samples: int = 10000) -> list[float]:
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = delta[generator.integers(0, len(delta), size=len(delta))].mean()
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_done_path = args.run_dir / "TRAIN_DONE.json"
    train_done = json.loads(train_done_path.read_text())
    if train_done.get("test_access") is not False:
        raise RuntimeError("result summarizer accepts development-only runs")
    checkpoint_path = Path(train_done["best_checkpoint"]["path"])
    if sha256_file(checkpoint_path) != train_done["best_checkpoint"]["sha256"]:
        raise RuntimeError("selected checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validation = checkpoint["validation"]
    if int(checkpoint["epoch"]) != int(train_done["selection"]["epoch"]):
        raise RuntimeError("selected epoch mismatch")
    if abs(float(validation["local_scan_macro_dsc"]) - float(train_done["selection"]["value"])) > 1e-12:
        raise RuntimeError("selected validation metric mismatch")

    per_case = validation["per_case"]
    case_ids = [str(row["case_id"]) for row in per_case]
    coarse = np.asarray([row["global_dsc"] for row in per_case], dtype=np.float64)
    refined = np.asarray([row["local_refined_dsc"] for row in per_case], dtype=np.float64)
    cache_index = json.loads((args.validation_cache / "index.json").read_text())
    if cache_index.get("test_access") is not False or cache_index["case_ids"] != case_ids:
        raise RuntimeError("validation cache ordering or test guard mismatch")
    source = Path(cache_index["source_cache"]["path"])
    labels = np.memmap(
        source / "labels.u8", dtype=np.uint8, mode="r",
        shape=(len(case_ids), *MODEL_SHAPE),
    )
    volumes = np.asarray([np.count_nonzero(labels[index]) for index in range(len(case_ids))])
    delta = refined - coarse
    prereg_sha = sha256_file(args.prereg)
    result = {
        "schema": "aflow.sammed3d.t1ce_wt_prompt_local_refiner_result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "mainline": {
            "modality": "T1ce only",
            "target": "WT={1,2,3}; raw label 4 excluded",
            "split": "972 train / 108 validation, subject-clean",
            "prompt_protocol": "3 accumulated corrective clicks with recurrent low-resolution mask",
            "decision_rule": "raw logit > 0",
            "sealed_test": "not accessed",
        },
        "experiment": {
            "name": "fixed 64^3 first-positive-click-centred local residual over frozen global refiner",
            "run_directory": str(args.run_dir),
            "preregistration": str(args.prereg),
            "preregistration_sha256": prereg_sha,
            "train_done_sha256": sha256_file(train_done_path),
            "runner_sha256": train_done["runner_sha256"],
            "world_size": train_done["world_size"],
        },
        "selected_result": {
            "epoch": int(checkpoint["epoch"]),
            "global_validation_scan_macro_dsc": float(coarse.mean()),
            "prompt_local_validation_scan_macro_dsc": float(refined.mean()),
            "absolute_gain": float(delta.mean()),
            "paired_bootstrap_95_ci": bootstrap_ci(delta, 20260812),
            "improved_cases": int(np.sum(delta > 0)),
            "worsened_cases": int(np.sum(delta < 0)),
            "unchanged_cases": int(np.sum(delta == 0)),
            "validation_cases": len(case_ids),
            "validation_subjects": len({subject_id(case_id) for case_id in case_ids}),
            "global_subject_macro_dsc": subject_macro(case_ids, coarse),
            "prompt_local_subject_macro_dsc": subject_macro(case_ids, refined),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "volume_stratified_validation": volume_rows(volumes, coarse, refined),
        "gates": {
            "development_success_threshold": 0.68,
            "target_reproduction_threshold": 0.70,
            "development_success": bool(refined.mean() >= 0.68),
            "target_reproduced": bool(refined.mean() >= 0.70),
            "automatic_cascade_gate_open": bool(refined.mean() >= 0.68),
            "sealed_test_access_authorized": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["selected_result"], indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
