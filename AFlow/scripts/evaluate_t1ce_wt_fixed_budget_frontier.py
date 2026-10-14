#!/usr/bin/env python3
"""Evaluate the preregistered fixed 1/3/4/8/12/24 forward frontier."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    VALIDATION_CACHE_INDEX_SHA256,
    RobustAtomCache,
    WorkflowEvaluator,
    WorkflowSpec,
    selected_atom_indices,
    sha256_file,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    write_json_atomic,
)


DEFAULT_ATOMS = Path("workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32")
DEFAULT_VALIDATION = Path(
    "workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/validation"
)
DEFAULT_PREREG = Path("FIXED_BUDGET_FRONTIER_PREREG_20260814.md")
DEFAULT_OUTPUT = Path(
    "workspace/BraTS/sammed3d_t1ce_wt/fixed_budget_frontier_20260814/"
    "FINAL_FIXED_BUDGET_FRONTIER.json"
)
REPLICATES = 10_000
SEED = 20260814


WORKFLOWS = {
    "fixed1_seed15_identity": WorkflowSpec(
        model_mask=4,
        view_policy="identity",
        fusion="logit_mean",
        threshold_q=-15,
        uncertainty_beta_q=0,
        min_component_voxels=500,
    ),
    "fixed3_models_identity": WorkflowSpec(
        model_mask=7,
        view_policy="identity",
        fusion="logit_mean",
        threshold_q=-20,
        uncertainty_beta_q=0,
        min_component_voxels=500,
    ),
    "fixed4_seed15_single_flips": WorkflowSpec(
        model_mask=4,
        view_policy="single_flips",
        fusion="logit_mean",
        threshold_q=-5,
        uncertainty_beta_q=-5,
        min_component_voxels=250,
    ),
    "fixed8_seed15_all_views": WorkflowSpec(
        model_mask=4,
        view_policy="all8",
        fusion="logit_mean",
        threshold_q=-5,
        uncertainty_beta_q=-5,
        min_component_voxels=250,
    ),
    "fixed12_models_single_flips": WorkflowSpec(
        model_mask=7,
        view_policy="single_flips",
        fusion="logit_median",
        threshold_q=-5,
        uncertainty_beta_q=-5,
        min_component_voxels=500,
    ),
    "fixed24_locked": WorkflowSpec.locked_reference(),
}


def bootstrap_delta(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    seed: int,
) -> dict[str, Any]:
    subjects = sorted(candidate)
    if subjects != sorted(reference):
        raise RuntimeError("paired subject sets differ")
    delta = np.asarray([candidate[key] - reference[key] for key in subjects])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(REPLICATES, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "subjects": len(subjects),
        "mean": float(np.mean(delta)),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "improved_subjects": int(np.count_nonzero(delta > 0)),
        "worsened_subjects": int(np.count_nonzero(delta < 0)),
        "unchanged_subjects": int(np.count_nonzero(delta == 0)),
        "replicates": REPLICATES,
        "seed": seed,
    }


def pareto_names(results: dict[str, dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for name, value in results.items():
        accuracy = float(value["evaluation"]["subject_macro_dsc"])
        forwards = int(value["forwards"])
        dominated = any(
            other_name != name
            and float(other["evaluation"]["subject_macro_dsc"]) >= accuracy
            and int(other["forwards"]) <= forwards
            and (
                float(other["evaluation"]["subject_macro_dsc"]) > accuracy
                or int(other["forwards"]) < forwards
            )
            for other_name, other in results.items()
        )
        if not dominated:
            names.append(name)
    return sorted(names, key=lambda name: results[name]["forwards"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.atom_cache, args.validation_cache, args.prereg, args.output):
        lowered = str(path.resolve()).lower()
        if "sealed_test" in lowered or "test270" in lowered:
            raise ValueError("test paths are forbidden")
    atoms = RobustAtomCache(args.atom_cache)
    validation = RefinerDataset(args.validation_cache)
    if atoms.case_ids != validation.case_ids or len(atoms.case_ids) != 108:
        raise RuntimeError("atom/label validation ordering mismatch")
    validation_index = args.validation_cache / "index.json"
    if sha256_file(validation_index) != VALIDATION_CACHE_INDEX_SHA256:
        raise RuntimeError("validation cache binding mismatch")
    validation._open()
    assert validation._labels is not None
    evaluator = WorkflowEvaluator(
        atoms.atoms,
        validation._labels,
        atoms.case_ids,
        memory_cache_bytes=int(args.memory_cache_gib * 1024**3),
    )
    evaluations: dict[str, dict[str, Any]] = {}
    for name, spec in WORKFLOWS.items():
        value = evaluator.evaluate(spec, range(108))
        evaluations[name] = {
            "workflow_id": spec.workflow_id,
            "spec": spec.canonical_dict(),
            "forwards": len(selected_atom_indices(spec.model_mask, spec.view_policy)),
            "evaluation": value,
        }
        print(
            f"{name}: subject={value['subject_macro_dsc']:.9f} "
            f"scan={value['scan_macro_dsc']:.9f} forwards={value['mean_forwards']:.1f}",
            flush=True,
        )
    locked = evaluations["fixed24_locked"]["evaluation"]
    for index, (name, value) in enumerate(evaluations.items()):
        value["paired_vs_locked"] = bootstrap_delta(
            value["evaluation"]["subject_values"],
            locked["subject_values"],
            seed=SEED + index,
        )
    report = {
        "schema": "aflow.sammed3d.t1ce_wt_fixed_budget_frontier",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "evidence_status": "preregistered ablation on previously used development validation",
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "scope": "refiner member-view forwards; not end-to-end latency",
        "workflows": evaluations,
        "subject_dice_pareto_workflows": pareto_names(evaluations),
        "protocol": {
            "scans": 108,
            "subjects": 50,
            "bootstrap_replicates": REPLICATES,
            "bootstrap_seed_base": SEED,
        },
        "bindings": {
            "prereg": str(args.prereg),
            "prereg_sha256": sha256_file(args.prereg),
            "atom_index_sha256": sha256_file(atoms.index_path),
            "validation_index_sha256": sha256_file(validation_index),
            "mcts_core_sha256": sha256_file(ROOT / "sammed3d/t1ce_wt_mcts.py"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "pareto": report["subject_dice_pareto_workflows"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-cache", type=Path, default=DEFAULT_ATOMS)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-cache-gib", type=float, default=2.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
