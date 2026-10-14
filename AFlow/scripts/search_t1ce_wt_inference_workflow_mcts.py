#!/usr/bin/env python3
"""Run genuine MCTS over frozen robust-refiner validation workflows.

Search and selection use subject-disjoint nested roles inside the already-used
108-case validation set.  Each outer role is scored exactly once after its
winner has been selected.  This controls new workflow-selection leakage but
does not turn the result into independent external evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    MCTSSearch,
    RobustAtomCache,
    SearchSpace,
    VALIDATION_CACHE_INDEX_SHA256,
    WorkflowEvaluator,
    WorkflowSpec,
    aggregate_scores,
    nested_roles,
    sha256_file,
    stable_id,
    subject_id,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    write_json_atomic,
)


DEFAULT_VALIDATION_CACHE = Path(
    "workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/validation"
)
HANDOFF = ROOT / "AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md"


def _fail_if_test_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "sealed_test" in lowered or "test270" in lowered:
        raise ValueError("test paths are forbidden for MCTS workflow search")


def _spec(value: dict[str, Any]) -> WorkflowSpec:
    fields = dict(value)
    fields.pop("schema_version", None)
    return WorkflowSpec(**fields)


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    temporary.replace(path)


def _bootstrap_subject_delta(
    candidate_by_subject: dict[str, float],
    reference_by_subject: dict[str, float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    subjects = sorted(set(candidate_by_subject) & set(reference_by_subject))
    delta = np.asarray(
        [candidate_by_subject[key] - reference_by_subject[key] for key in subjects],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(replicates, len(delta)))
    means = np.mean(delta[draws], axis=1)
    return {
        "subjects": len(subjects),
        "mean": float(np.mean(delta)),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "seed": seed,
        "replicates": replicates,
    }


def _consensus_workflow(winners: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the observed winner with minimum field-wise distance to peers."""
    specs = [_spec(item["winner"]["spec"]) for item in winners]
    dicts = [spec.canonical_dict() for spec in specs]
    fields = [key for key in dicts[0] if key != "schema_version"]
    ranked: list[tuple[int, float, str, WorkflowSpec]] = []
    for index, spec in enumerate(specs):
        distance = sum(
            dicts[index][field] != other[field]
            for other in dicts
            for field in fields
        )
        mean_forwards = float(
            winners[index]["winner"]["selector_evaluation"]["mean_forwards"]
        )
        ranked.append((distance, mean_forwards, spec.workflow_id, spec))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen = ranked[0][3]
    return {
        "rule": "observed outer-fold winner minimizing field-wise distance to the other winners; ties prefer lower compute then hash",
        "spec": chosen.canonical_dict(),
        "workflow_id": chosen.workflow_id,
        "status": "deployment candidate only; not evaluated on the full validation set by this runner",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    atom_root = args.atom_cache.resolve()
    validation_cache = args.validation_cache.resolve()
    output_dir = args.output_dir.resolve()
    for path in (atom_root, validation_cache, output_dir):
        _fail_if_test_path(path)
    output_existed = output_dir.exists()
    if output_existed and not args.resume:
        raise FileExistsError("output directory exists; use a new run id or --resume")
    if args.resume and not output_existed:
        raise FileNotFoundError("--resume requires an existing run directory")
    if args.unique_workflows < 1 or args.max_simulations < args.unique_workflows:
        raise ValueError("require max_simulations >= unique_workflows >= 1")
    if not 1 <= args.selector_top_k <= args.unique_workflows:
        raise ValueError("selector_top_k must be in [1, unique_workflows]")
    if args.checkpoint_every < 1 or args.bootstrap_replicates < 1:
        raise ValueError("checkpoint_every and bootstrap_replicates must be positive")
    if not np.isfinite(args.memory_cache_gib) or args.memory_cache_gib < 0:
        raise ValueError("memory_cache_gib must be finite and nonnegative")
    if not np.isfinite(args.uct_exploration) or args.uct_exploration < 0:
        raise ValueError("uct_exploration must be finite and nonnegative")
    if not np.isfinite(args.widening_k) or args.widening_k <= 0:
        raise ValueError("widening_k must be finite and positive")
    if not np.isfinite(args.widening_alpha) or not 0 < args.widening_alpha <= 1:
        raise ValueError("widening_alpha must be finite and in (0, 1]")
    output_dir.mkdir(parents=True, exist_ok=True)

    atom_cache = RobustAtomCache(atom_root)
    if sha256_file(validation_cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise RuntimeError("validation target cache hash mismatch")
    validation_data = RefinerDataset(validation_cache)
    if validation_data.index.get("split") != "validation" or validation_data.index.get("test_access") is not False:
        raise ValueError("MCTS scorer requires development-only validation targets")
    if validation_data.case_ids != atom_cache.case_ids:
        raise RuntimeError("atom and target cache case ordering mismatch")
    actual_labels_sha = sha256_file(validation_data.source / "labels.u8")
    if actual_labels_sha != atom_cache.index["source_labels_sha256"]:
        raise RuntimeError("validation label blob hash does not match the exact atom cache binding")
    validation_data._open()
    assert validation_data._labels is not None
    evaluator = WorkflowEvaluator(
        atom_cache.atoms,
        validation_data._labels,
        atom_cache.case_ids,
        memory_cache_bytes=int(args.memory_cache_gib * 1024**3),
    )
    space = SearchSpace()
    folds_to_run = range(3) if args.outer_fold == "all" else (int(args.outer_fold),)
    manifest = {
        "schema": "aflow.sammed3d.t1ce_wt_inference_workflow_mcts.run_manifest",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "evidence_status": "validation-only nested development analysis",
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "uses_gray_loss": False,
        "authoritative_handoff": {"path": str(HANDOFF), "sha256": sha256_file(HANDOFF)},
        "atom_cache": {
            "path": str(atom_root),
            "index_sha256": sha256_file(atom_cache.index_path),
            "atoms_sha256": atom_cache.index["atoms_sha256"],
            "dtype": atom_cache.index["dtype"],
            "evidence_level": atom_cache.index["evidence_level"],
            "core_sha256": atom_cache.index["core_sha256"],
            "parity": atom_cache.index["parity"],
            "autocast": atom_cache.index["autocast"],
        },
        "validation_cache": {
            "path": str(validation_cache),
            "index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "labels_sha256": actual_labels_sha,
        },
        "search_source_sha256": sha256_file(Path(__file__).resolve()),
        "mcts_source_sha256": sha256_file(ROOT / "sammed3d/t1ce_wt_mcts.py"),
        "search_space": space.canonical_dict(),
        "search_space_sha256": stable_id(space.canonical_dict()),
        "mandatory_anchor": {
            "workflow_id": WorkflowSpec.locked_reference().workflow_id,
            "spec": WorkflowSpec.locked_reference().canonical_dict(),
            "budget_semantics": "counts as one unique workflow but creates no synthetic tree visits",
        },
        "config": {
            "outer_fold": args.outer_fold,
            "unique_workflows_per_fold": args.unique_workflows,
            "max_simulations_per_fold": args.max_simulations,
            "selector_top_k": args.selector_top_k,
            "seed": args.seed,
            "uct_exploration": args.uct_exploration,
            "progressive_widening_k": args.widening_k,
            "progressive_widening_alpha": args.widening_alpha,
            "memory_cache_gib": args.memory_cache_gib,
            "checkpoint_every": args.checkpoint_every,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "reward": {
            "primary_accuracy": "subject-macro Dice",
            "paired_se_weight": 0.25,
            "compute_penalty_weight": 0.002,
            "complexity_penalty_weight": 0.0005,
        },
        "forbidden_inputs": ["270-case revealed test", "sealed_test", "test reports"],
    }
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and args.resume:
        previous = json.loads(manifest_path.read_text())
        previous_comparable = dict(previous)
        current_comparable = dict(manifest)
        previous_comparable.pop("created_at_utc", None)
        current_comparable.pop("created_at_utc", None)
        if previous_comparable != current_comparable:
            raise RuntimeError("resume immutable manifest/source/data/config binding mismatch")
    elif args.resume:
        raise FileNotFoundError("--resume run directory has no run_manifest.json")
    else:
        write_json_atomic(manifest_path, manifest)
    final_report_path = output_dir / "final_report.json"
    existing_final_report = (
        json.loads(final_report_path.read_text()) if final_report_path.exists() else None
    )
    manifest_sha = sha256_file(manifest_path)

    fold_reports: list[dict[str, Any]] = []
    oof_case_values: dict[int, float] = {}
    oof_reference_values: dict[int, float] = {}
    for outer_fold in folds_to_run:
        roles = nested_roles(atom_cache.case_ids, outer_fold, seed=args.seed)
        role_receipt = {
            role: {
                "case_indices": indices,
                "case_ids": [atom_cache.case_ids[index] for index in indices],
                "subjects": sorted({subject_id(atom_cache.case_ids[index]) for index in indices}),
            }
            for role, indices in roles.items()
        }
        role_receipt["sha256"] = stable_id(role_receipt)
        fold_dir = output_dir / f"outer_fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        roles_path = fold_dir / "subject_roles.json"
        if roles_path.exists():
            if json.loads(roles_path.read_text()) != role_receipt:
                raise RuntimeError("resume subject-role binding mismatch")
        else:
            write_json_atomic(roles_path, role_receipt)
        context_id = stable_id(
            {
                "manifest_sha256": manifest_sha,
                "roles_sha256": role_receipt["sha256"],
                "outer_fold": outer_fold,
            }
        )
        completed_fold_path = fold_dir / "fold_report.json"
        if completed_fold_path.exists():
            if not args.resume:
                raise FileExistsError("completed fold exists")
            fold_report = json.loads(completed_fold_path.read_text())
            if fold_report.get("context_id") != context_id:
                raise RuntimeError("completed fold context binding mismatch")
            expected_chain = {
                "mcts_state_sha256": sha256_file(fold_dir / "mcts_state.json"),
                "events_sha256": sha256_file(fold_dir / "events.jsonl"),
                "frozen_winner_sha256": sha256_file(fold_dir / "FROZEN_WINNER.json"),
                "outer_started_sha256": sha256_file(
                    fold_dir / "OUTER_EVALUATION_STARTED.json"
                ),
            }
            if fold_report.get("evidence_chain") != expected_chain:
                raise RuntimeError("completed fold evidence-chain hash mismatch")
            if set(fold_report["outer_evaluation"]["case_ids"]) != {
                atom_cache.case_ids[index] for index in roles["outer_holdout"]
            }:
                raise RuntimeError("completed fold outer case ids do not match its holdout role")
            fold_reports.append(fold_report)
            for case_id, dice in zip(
                fold_report["outer_evaluation"]["case_ids"],
                fold_report["outer_evaluation"]["case_dice"],
                strict=True,
            ):
                oof_case_values[atom_cache.case_ids.index(case_id)] = float(dice)
            for case_id, dice in zip(
                fold_report["outer_reference"]["case_ids"],
                fold_report["outer_reference"]["case_dice"],
                strict=True,
            ):
                oof_reference_values[atom_cache.case_ids.index(case_id)] = float(dice)
            continue
        if (fold_dir / "OUTER_EVALUATION_STARTED.json").exists():
            raise RuntimeError(
                "outer evaluation was previously started without a completed receipt; "
                "fail-closed to prevent a silent second holdout read"
            )

        def search_reward(spec: WorkflowSpec) -> dict[str, Any]:
            return evaluator.reward(spec, roles["mcts_search"])

        state_path = fold_dir / "mcts_state.json"
        if state_path.exists() and args.resume:
            search = MCTSSearch.from_state_dict(
                json.loads(state_path.read_text()),
                search_reward,
                expected_context_id=context_id,
            )
        else:
            search = MCTSSearch(
                search_reward,
                space=space,
                seed=args.seed + 100_003 * outer_fold,
                exploration=args.uct_exploration,
                widening_k=args.widening_k,
                widening_alpha=args.widening_alpha,
                context_id=context_id,
            )
        anchor_spec = WorkflowSpec.locked_reference()
        if anchor_spec.workflow_id not in search.evaluations:
            anchor_evaluation = search_reward(anchor_spec)
            if anchor_evaluation.get("workflow_id") != anchor_spec.workflow_id:
                raise RuntimeError("anchor evaluator returned a mismatched workflow id")
            search.evaluations[anchor_spec.workflow_id] = anchor_evaluation
        completed = search.nodes[search.root_id].visits
        if completed > args.max_simulations or len(search.evaluations) > args.unique_workflows:
            raise RuntimeError("resume state exceeds the requested search budget")
        for iteration in range(completed, args.max_simulations):
            if len(search.evaluations) >= args.unique_workflows:
                break
            search.simulate()
            if (iteration + 1) % args.checkpoint_every == 0:
                write_json_atomic(state_path, search.state_dict())
        if len(search.evaluations) < args.unique_workflows:
            raise RuntimeError("MCTS exhausted max_simulations before reaching unique-workflow budget")
        write_json_atomic(state_path, search.state_dict())
        _write_events(fold_dir / "events.jsonl", search.events)

        ranked = search.ranked_evaluations()
        if len(ranked) < args.selector_top_k:
            raise RuntimeError("MCTS produced fewer unique workflows than selector_top_k")
        selector_candidates: list[dict[str, Any]] = []
        for search_evaluation in ranked[: args.selector_top_k]:
            spec = _spec(search_evaluation["spec"])
            selector = evaluator.reward(spec, roles["inner_selector"])
            selector_candidates.append(
                {
                    "workflow_id": spec.workflow_id,
                    "spec": spec.canonical_dict(),
                    "search_evaluation": search_evaluation,
                    "selector_evaluation": selector,
                }
            )
        selector_candidates.sort(
            key=lambda item: (
                -item["selector_evaluation"]["reward"],
                item["selector_evaluation"]["mean_forwards"],
                item["workflow_id"],
            )
        )
        winner = selector_candidates[0]
        winner_spec = _spec(winner["spec"])
        frozen_winner = {
            "context_id": context_id,
            "workflow_id": winner_spec.workflow_id,
            "spec": winner_spec.canonical_dict(),
            "selection_role": "inner_selector",
            "outer_accessed": False,
        }
        write_json_atomic(fold_dir / "FROZEN_WINNER.json", frozen_winner)
        write_json_atomic(
            fold_dir / "OUTER_EVALUATION_STARTED.json",
            {
                "context_id": context_id,
                "workflow_id": winner_spec.workflow_id,
                "policy": "fail closed rather than silently repeat if interrupted",
            },
        )
        outer_evaluation = evaluator.reward(winner_spec, roles["outer_holdout"])
        outer_reference = evaluator.evaluate(
            WorkflowSpec.locked_reference(), roles["outer_holdout"]
        )
        for case_id, dice in zip(
            outer_evaluation["case_ids"], outer_evaluation["case_dice"], strict=True
        ):
            oof_case_values[atom_cache.case_ids.index(case_id)] = float(dice)
        for case_id, dice in zip(
            outer_reference["case_ids"], outer_reference["case_dice"], strict=True
        ):
            oof_reference_values[atom_cache.case_ids.index(case_id)] = float(dice)
        fold_report = {
            "context_id": context_id,
            "outer_fold": outer_fold,
            "roles_sha256": role_receipt["sha256"],
            "simulations": search.nodes[search.root_id].visits,
            "unique_workflows": len(search.evaluations),
            "winner": winner,
            "outer_evaluation": outer_evaluation,
            "outer_reference": {
                key: value for key, value in outer_reference.items() if key != "subject_values"
            },
            "outer_read_policy": "winner fixed on inner_selector before this one outer evaluation",
        }
        fold_report["evidence_chain"] = {
            "mcts_state_sha256": sha256_file(state_path),
            "events_sha256": sha256_file(fold_dir / "events.jsonl"),
            "frozen_winner_sha256": sha256_file(fold_dir / "FROZEN_WINNER.json"),
            "outer_started_sha256": sha256_file(
                fold_dir / "OUTER_EVALUATION_STARTED.json"
            ),
        }
        write_json_atomic(fold_dir / "fold_report.json", fold_report)
        fold_reports.append(fold_report)

    full_oof = len(fold_reports) == 3
    oof_report: dict[str, Any] | None = None
    if full_oof:
        ordered_slots = sorted(oof_case_values)
        if ordered_slots != list(range(len(atom_cache.case_ids))):
            raise RuntimeError("outer folds did not produce exactly one prediction for every validation case")
        selected_ids = [atom_cache.case_ids[index] for index in ordered_slots]
        candidate_values = [oof_case_values[index] for index in ordered_slots]
        reference_values = [oof_reference_values[index] for index in ordered_slots]
        candidate_aggregate = aggregate_scores(selected_ids, candidate_values)
        reference_aggregate = aggregate_scores(selected_ids, reference_values)
        oof_report = {
            "candidate": candidate_aggregate,
            "locked_reference": reference_aggregate,
            "paired_subject_delta": _bootstrap_subject_delta(
                candidate_aggregate["subject_values"],
                reference_aggregate["subject_values"],
                seed=args.seed,
                replicates=args.bootstrap_replicates,
            ),
            "interpretation": (
                "Cross-fitted estimate of the workflow-search procedure inside an already-used "
                "validation set; not an independent test or external confirmation result."
            ),
        }

    if existing_final_report is not None:
        if (
            existing_final_report.get("schema")
            != "aflow.sammed3d.t1ce_wt_inference_workflow_mcts.report"
            or existing_final_report.get("test_access") is not False
            or existing_final_report.get("manifest_sha256") != manifest_sha
            or existing_final_report.get("folds") != fold_reports
        ):
            raise RuntimeError("completed final report binding/content mismatch")
        print(json.dumps({"output": str(output_dir), "status": "already_complete"}, indent=2))
        return existing_final_report

    report = {
        "schema": "aflow.sammed3d.t1ce_wt_inference_workflow_mcts.report",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "uses_gray_loss": False,
        "manifest_sha256": manifest_sha,
        "folds": fold_reports,
        "cross_fitted_oof": oof_report,
        "deployment_candidate": _consensus_workflow(fold_reports) if full_oof else None,
        "evidence_boundary": (
            "The 108 validation cases previously selected checkpoints/postprocessing. Nested roles "
            "limit additional workflow-search leakage but cannot create pristine evidence. The "
            "revealed 270-case test is unreachable from this runner and must not be used for tuning."
        ),
        "next_required_evidence": (
            "Freeze the selected workflow and evaluate once on a new external confirmation set; "
            "separately replace GT-centered oracle localization with an automatic localizer."
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(final_report_path, report)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "fold_winners": [item["winner"]["workflow_id"] for item in fold_reports],
                "cross_fitted_oof": oof_report,
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", choices=("all", "0", "1", "2"), default="all")
    parser.add_argument("--unique-workflows", type=int, default=128)
    parser.add_argument("--max-simulations", type=int, default=512)
    parser.add_argument("--selector-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--uct-exploration", type=float, default=0.02)
    parser.add_argument("--widening-k", type=float, default=2.0)
    parser.add_argument("--widening-alpha", type=float, default=0.5)
    parser.add_argument("--memory-cache-gib", type=float, default=8.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
