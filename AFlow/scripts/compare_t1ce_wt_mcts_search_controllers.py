#!/usr/bin/env python3
"""Equal-budget controller comparison for the frozen inference workflow space.

This is a development analysis on the already-used 108-case validation set.
It does not create independent evidence and cannot access the 270-case test.
The exact historical MCTS pools are replayed from their immutable state files;
uniform random and greedy best-first coordinate-neighbourhood controllers use
the same grammar, subject roles, 128-unique-workflow budget, and selector size.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    RobustAtomCache,
    SearchSpace,
    SearchState,
    VALIDATION_CACHE_INDEX_SHA256,
    WorkflowEvaluator,
    WorkflowSpec,
    nested_roles,
    sha256_file,
    stable_id,
)
from sammed3d.t1ce_wt_mcts_efficiency import (  # noqa: E402
    DEFAULT_ACCURACY_MARGINS,
    mandatory_anchor_pool,
    margin_sweep,
    pareto_front,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    write_json_atomic,
)


DEFAULT_ATOMS = Path("workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32")
DEFAULT_VALIDATION = Path("workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/validation")
DEFAULT_MCTS = Path("workspace/BraTS/sammed3d_t1ce_wt/mcts_workflow_20260813_v2")


def _fail_if_test_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "sealed_test" in lowered or "test270" in lowered:
        raise ValueError("test paths are forbidden for search-controller comparison")


def _rank(evaluations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(value) for value in evaluations),
        key=lambda value: (
            -float(value["reward"]),
            float(value["mean_forwards"]),
            value["workflow_id"],
        ),
    )


def _random_spec(rng: np.random.Generator, space: SearchSpace) -> WorkflowSpec:
    state = SearchState()
    while not state.terminal:
        actions = state.legal_actions(space)
        state = state.apply(actions[int(rng.integers(0, len(actions)))], space)
    return state.to_spec()


def random_search(
    evaluator: WorkflowEvaluator,
    indices: list[int],
    *,
    space: SearchSpace,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    anchor = WorkflowSpec.locked_reference()
    evaluations = {anchor.workflow_id: evaluator.reward(anchor, indices)}
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(evaluations) < budget:
        spec = _random_spec(rng, space)
        attempts += 1
        if spec.workflow_id not in evaluations:
            evaluations[spec.workflow_id] = evaluator.reward(spec, indices)
        if attempts > budget * 1000:
            raise RuntimeError("random controller failed to draw enough unique workflows")
    return list(evaluations.values())


def _acquisition_action(spec: WorkflowSpec) -> str:
    if spec.acquisition == "fixed":
        return f"fixed:{spec.model_mask}:{spec.view_policy}"
    return f"adaptive:{spec.gate3_q}:{spec.gate12_q}"


def _single_atom(spec: WorkflowSpec) -> bool:
    return spec.acquisition == "fixed" and spec.model_mask.bit_count() == 1 and spec.view_policy == "identity"


def coordinate_neighbors(spec: WorkflowSpec, space: SearchSpace) -> list[WorkflowSpec]:
    """All valid workflows one grammar field away, deterministically ordered."""
    candidates: dict[str, WorkflowSpec] = {}

    def add(value: WorkflowSpec) -> None:
        if value.workflow_id != spec.workflow_id:
            candidates[value.workflow_id] = value

    for action in space.acquisition_actions:
        pieces = action.split(":")
        if pieces[0] == "fixed":
            value = replace(
                spec,
                acquisition="fixed",
                model_mask=int(pieces[1]),
                view_policy=pieces[2],
                gate3_q=0,
                gate12_q=0,
            )
            if _single_atom(value):
                value = replace(value, fusion="logit_mean", uncertainty_beta_q=0)
        else:
            value = replace(
                spec,
                acquisition="adaptive",
                model_mask=7,
                view_policy="all8",
                gate3_q=int(pieces[1]),
                gate12_q=int(pieces[2]),
            )
        add(value)
    if not _single_atom(spec):
        for fusion in space.fusion_actions:
            add(replace(spec, fusion=fusion))
    for threshold in space.threshold_actions:
        add(replace(spec, threshold_q=int(threshold)))
    beta_actions = (0,) if _single_atom(spec) else space.beta_actions
    for beta in beta_actions:
        add(replace(spec, uncertainty_beta_q=int(beta)))
    for component in space.component_actions:
        add(replace(spec, min_component_voxels=int(component)))
    return [candidates[key] for key in sorted(candidates)]


def greedy_best_first_search(
    evaluator: WorkflowEvaluator,
    indices: list[int],
    *,
    space: SearchSpace,
    budget: int,
) -> list[dict[str, Any]]:
    """Expand coordinate neighbours of the best unexpanded evaluated node."""
    anchor = WorkflowSpec.locked_reference()
    evaluations = {anchor.workflow_id: evaluator.reward(anchor, indices)}
    specs = {anchor.workflow_id: anchor}
    expanded: set[str] = set()
    while len(evaluations) < budget:
        available = [
            value for key, value in evaluations.items() if key not in expanded
        ]
        if not available:
            raise RuntimeError("greedy controller exhausted its neighbourhood graph")
        parent_evaluation = _rank(available)[0]
        parent_id = parent_evaluation["workflow_id"]
        expanded.add(parent_id)
        added = 0
        for neighbour in coordinate_neighbors(specs[parent_id], space):
            if neighbour.workflow_id in evaluations:
                continue
            specs[neighbour.workflow_id] = neighbour
            evaluations[neighbour.workflow_id] = evaluator.reward(neighbour, indices)
            added += 1
            if len(evaluations) == budget:
                break
        if added == 0 and len(evaluations) < budget:
            continue
    return list(evaluations.values())


def historical_mcts_pool(root: Path, outer_fold: int, budget: int) -> list[dict[str, Any]]:
    state_path = root / f"outer_fold_{outer_fold}" / "mcts_state.json"
    state = json.loads(state_path.read_text())
    evaluations = list(state["evaluations"].values())
    if len(evaluations) != budget:
        raise RuntimeError("historical MCTS pool does not match comparison budget")
    if WorkflowSpec.locked_reference().workflow_id not in state["evaluations"]:
        raise RuntimeError("historical MCTS pool lacks the locked anchor")
    return evaluations


def compact_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "workflow_id",
            "spec",
            "reward",
            "accuracy",
            "paired_subject_delta_mean",
            "paired_subject_delta_se",
            "compute_penalty",
            "complexity_penalty",
            "scan_macro_dsc",
            "subject_macro_dsc",
            "mean_forwards",
            "max_forwards",
            "stop_stage_counts",
        )
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    for path in (args.atom_cache, args.validation_cache, args.historical_mcts, args.output):
        _fail_if_test_path(path)
    if args.unique_workflows != 128 or args.selector_top_k != 8:
        raise ValueError("preregistered comparison requires 128 workflows and selector top-k 8")
    if tuple(args.accuracy_margins) != DEFAULT_ACCURACY_MARGINS:
        raise ValueError("accuracy margins must match the preregistered sweep")

    atoms = RobustAtomCache(args.atom_cache)
    validation = RefinerDataset(args.validation_cache)
    if sha256_file(args.validation_cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise RuntimeError("validation cache binding mismatch")
    if validation.case_ids != atoms.case_ids:
        raise RuntimeError("atom/target case ordering mismatch")
    validation._open()
    assert validation._labels is not None
    evaluator = WorkflowEvaluator(
        atoms.atoms,
        validation._labels,
        atoms.case_ids,
        memory_cache_bytes=int(args.memory_cache_gib * 1024**3),
    )
    roles = nested_roles(atoms.case_ids, args.outer_fold, seed=args.seed)
    space = SearchSpace()
    if args.controller == "mcts":
        search_evaluations = historical_mcts_pool(
            args.historical_mcts,
            args.outer_fold,
            args.unique_workflows,
        )
    elif args.controller == "random":
        search_evaluations = random_search(
            evaluator,
            roles["mcts_search"],
            space=space,
            budget=args.unique_workflows,
            seed=args.seed + 200_003 * (args.outer_fold + 1),
        )
    else:
        search_evaluations = greedy_best_first_search(
            evaluator,
            roles["mcts_search"],
            space=space,
            budget=args.unique_workflows,
        )

    ranked = _rank(search_evaluations)
    anchor_id = WorkflowSpec.locked_reference().workflow_id
    anchor_search = next(value for value in ranked if value["workflow_id"] == anchor_id)
    search_pool = mandatory_anchor_pool(
        ranked,
        anchor_evaluation=anchor_search,
        top_k=args.selector_top_k,
    )
    selector_records: list[dict[str, Any]] = []
    for search_evaluation in search_pool:
        fields = dict(search_evaluation["spec"])
        fields.pop("schema_version", None)
        spec = WorkflowSpec(**fields)
        selector_evaluation = evaluator.reward(spec, roles["inner_selector"])
        selector_records.append(
            {
                **compact_evaluation(selector_evaluation),
                "search_rank": next(
                    index
                    for index, value in enumerate(ranked, start=1)
                    if value["workflow_id"] == spec.workflow_id
                ),
                "search_evaluation": compact_evaluation(search_evaluation),
            }
        )

    sweep = margin_sweep(
        selector_records,
        anchor_workflow_id=anchor_id,
        margins=args.accuracy_margins,
    )
    outer_by_workflow: dict[str, dict[str, Any]] = {}
    for selection in sweep:
        selected = selection["selected"]
        fields = dict(selected["spec"])
        fields.pop("schema_version", None)
        spec = WorkflowSpec(**fields)
        if spec.workflow_id not in outer_by_workflow:
            outer_by_workflow[spec.workflow_id] = evaluator.reward(
                spec,
                roles["outer_holdout"],
            )
        selection["selected"] = {
            "workflow_id": spec.workflow_id,
            "spec": spec.canonical_dict(),
            "search_rank": selected["search_rank"],
            "selector_evaluation": compact_evaluation(selected),
            "outer_evaluation": outer_by_workflow[spec.workflow_id],
        }

    report = {
        "schema": "aflow.sammed3d.t1ce_wt_search_controller_comparison.fold",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "split": "validation",
        "evidence_status": "post-preregistered development comparison on previously used validation",
        "geometry": "GT-centered N4 128^3 oracle geometry; not automatic localization",
        "controller": args.controller,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "unique_workflows": len(search_evaluations),
        "selector_top_k": len(selector_records),
        "mandatory_anchor_present": any(
            value["workflow_id"] == anchor_id for value in selector_records
        ),
        "roles": {key: list(value) for key, value in roles.items()},
        "roles_sha256": stable_id({key: list(value) for key, value in roles.items()}),
        "search_space_sha256": stable_id(space.canonical_dict()),
        "search_front": [compact_evaluation(value) for value in pareto_front(search_evaluations)],
        "selector_candidates": selector_records,
        "selector_front": pareto_front(selector_records),
        "accuracy_margin_sweep": sweep,
        "bindings": {
            "atom_index_sha256": sha256_file(atoms.index_path),
            "validation_index_sha256": sha256_file(args.validation_cache / "index.json"),
            "historical_mcts_final_sha256": sha256_file(
                args.historical_mcts / "final_report.json"
            ),
            "core_sha256": sha256_file(ROOT / "sammed3d/t1ce_wt_mcts.py"),
            "efficiency_sha256": sha256_file(
                ROOT / "sammed3d/t1ce_wt_mcts_efficiency.py"
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "controller": args.controller,
                "outer_fold": args.outer_fold,
                "elapsed_seconds": report["elapsed_seconds"],
                "selections": [
                    {
                        "margin": value["margin"],
                        "workflow_id": value["selected"]["workflow_id"],
                        "selector_forwards": value["selected"]["selector_evaluation"]["mean_forwards"],
                        "outer_dice": value["selected"]["outer_evaluation"]["subject_macro_dsc"],
                    }
                    for value in sweep
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("mcts", "random", "greedy"), required=True)
    parser.add_argument("--outer-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--atom-cache", type=Path, default=DEFAULT_ATOMS)
    parser.add_argument("--validation-cache", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--historical-mcts", type=Path, default=DEFAULT_MCTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unique-workflows", type=int, default=128)
    parser.add_argument("--selector-top-k", type=int, default=8)
    parser.add_argument("--accuracy-margins", type=float, nargs=4, default=DEFAULT_ACCURACY_MARGINS)
    parser.add_argument("--memory-cache-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260813)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
