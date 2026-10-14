from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from sammed3d.t1ce_wt_mcts import (
    ATOM_CACHE_SCHEMA,
    MCTSSearch,
    RobustAtomCache,
    SearchSpace,
    WorkflowEvaluator,
    WorkflowOutput,
    WorkflowSpec,
    case_disagreement,
    build_subject_folds,
    nested_roles,
    run_workflow,
    selected_atom_indices,
    workflow_mask,
)


def test_locked_spec_is_canonical_and_hash_stable() -> None:
    first = WorkflowSpec.locked_reference()
    second = WorkflowSpec(
        acquisition="fixed",
        model_mask=7,
        view_policy="all8",
        fusion="logit_mean",
        threshold_q=-20,
        min_component_voxels=500,
    )
    assert first == second
    assert first.workflow_id == second.workflow_id
    assert first.threshold == -1.0
    with pytest.raises(ValueError, match="adaptive gates"):
        WorkflowSpec(gate3_q=5)
    with pytest.raises(ValueError, match="gate3"):
        WorkflowSpec(acquisition="adaptive", gate3_q=20, gate12_q=10)


def test_atom_sets_and_label_free_execution() -> None:
    assert selected_atom_indices(7, "identity") == (0, 8, 16)
    assert len(selected_atom_indices(7, "single_flips")) == 12
    assert selected_atom_indices(7, "all8")[:6] == (0, 8, 16, 1, 9, 17)
    assert "label" not in inspect.signature(run_workflow).parameters

    atoms = np.stack(
        [np.full((2, 2, 2), float(index), dtype=np.float32) for index in range(24)]
    )
    spec = WorkflowSpec(model_mask=1, view_policy="identity", min_component_voxels=0)
    output = run_workflow(atoms, spec)
    assert np.array_equal(output.fused_logits, atoms[0])
    assert output.forward_count == 1


def test_adaptive_stops_at_three_for_confident_identical_predictions() -> None:
    atoms = np.full((24, 2, 2, 2), 10.0, dtype=np.float32)
    spec = WorkflowSpec(
        acquisition="adaptive",
        gate3_q=5,
        gate12_q=10,
        min_component_voxels=0,
    )
    output = run_workflow(atoms, spec)
    assert output.forward_count == 3
    assert output.stop_stage == 3
    assert output.case_disagreement < 0.05


def test_identical_non_saturated_predictions_have_zero_disagreement() -> None:
    atoms = np.zeros((24, 2, 2, 2), dtype=np.float32)
    assert case_disagreement(atoms, (0, 8, 16)) == pytest.approx(0.0, abs=1e-6)


def test_uncertainty_threshold_and_26_connected_component_boundary() -> None:
    logits = np.full((4, 4, 4), -2.0, dtype=np.float32)
    logits[0, 0, 0] = 2.0
    logits[1, 1, 1] = 2.0  # diagonal adjacency is connected under 26-connectivity
    output = WorkflowOutput(logits, np.zeros_like(logits), 3, 3, 0.0)
    keep_two = WorkflowSpec(
        model_mask=7,
        view_policy="identity",
        threshold_q=0,
        min_component_voxels=2,
    )
    remove_three = WorkflowSpec(
        model_mask=7,
        view_policy="identity",
        threshold_q=0,
        min_component_voxels=3,
    )
    assert int(workflow_mask(output, keep_two).sum()) == 2
    assert int(workflow_mask(output, remove_three).sum()) == 0


def test_evaluator_subject_macro_and_reward_are_separate_from_inference() -> None:
    atoms = np.full((3, 24, 8, 8, 8), 5.0, dtype=np.float32)
    labels = np.ones((3, 8, 8, 8), dtype=np.uint8)
    evaluator = WorkflowEvaluator(
        atoms,
        labels,
        ["BraTS-GLI-00001-100", "BraTS-GLI-00001-101", "BraTS-GLI-00002-100"],
    )
    result = evaluator.evaluate(WorkflowSpec.locked_reference(), range(3))
    assert result["scan_macro_dsc"] == 1.0
    assert result["subject_macro_dsc"] == 1.0
    assert result["subjects"] == 2
    reward = evaluator.reward(WorkflowSpec.locked_reference(), range(3))
    assert reward["paired_subject_delta_mean"] == 0.0
    assert reward["reward"] < 1.0  # explicitly includes compute/complexity penalties


def test_uncertainty_cache_is_independent_of_candidate_evaluation_order() -> None:
    atoms = np.linspace(-2, 2, 3 * 24 * 4 * 4 * 4, dtype=np.float32).reshape(
        3, 24, 4, 4, 4
    )
    labels = np.ones((3, 4, 4, 4), dtype=np.uint8)
    ids = ["subject-a", "subject-b", "subject-c"]
    plain = WorkflowSpec(view_policy="identity", min_component_voxels=0)
    uncertain = WorkflowSpec(
        view_policy="identity", uncertainty_beta_q=10, min_component_voxels=0
    )
    first = WorkflowEvaluator(atoms, labels, ids, memory_cache_bytes=1024**2)
    first.evaluate(plain, range(3))
    result_after_plain = first.evaluate(uncertain, range(3))
    second = WorkflowEvaluator(atoms, labels, ids, memory_cache_bytes=1024**2)
    result_direct = second.evaluate(uncertain, range(3))
    assert result_after_plain["case_dice"] == result_direct["case_dice"]


def test_fixed_and_adaptive_share_logits_without_sharing_stop_metadata() -> None:
    atoms = np.zeros((1, 24, 3, 3, 3), dtype=np.float32)
    atoms[0, 0] = -1.0
    atoms[0, 8] = 0.0
    atoms[0, 16] = 1.0
    labels = np.ones((1, 3, 3, 3), dtype=np.uint8)
    fixed = WorkflowSpec(view_policy="identity", min_component_voxels=0)
    adaptive = WorkflowSpec(
        acquisition="adaptive",
        gate3_q=50,
        gate12_q=50,
        min_component_voxels=0,
    )
    for order in ((fixed, adaptive), (adaptive, fixed)):
        evaluator = WorkflowEvaluator(atoms, labels, ["subject-a"], memory_cache_bytes=1024**2)
        first, second = (evaluator.evaluate(spec, [0]) for spec in order)
        by_id = {order[0].workflow_id: first, order[1].workflow_id: second}
        assert by_id[fixed.workflow_id]["case_disagreement"] == [0.0]
        assert by_id[adaptive.workflow_id]["case_disagreement"][0] > 0.0


def test_subject_folds_and_nested_roles_have_zero_subject_overlap() -> None:
    case_ids = [
        "BraTS-GLI-00001-100",
        "BraTS-GLI-00001-101",
        "BraTS-GLI-00002-100",
        "BraTS-GLI-00003-100",
        "BraTS-GLI-00004-100",
        "BraTS-GLI-00005-100",
        "BraTS-GLI-00006-100",
        "BraTS-GLI-00007-100",
    ]
    folds = build_subject_folds(case_ids, folds=3)
    assert sorted(index for fold in folds for index in fold) == list(range(len(case_ids)))
    assert all(not ({case_ids[i].rsplit('-', 1)[0] for i in left} & {case_ids[i].rsplit('-', 1)[0] for i in right}) for n, left in enumerate(folds) for right in folds[n + 1 :])
    roles = nested_roles(case_ids, 0)
    assert sorted(index for indices in roles.values() for index in indices) == list(range(len(case_ids)))


def test_mcts_has_visits_rollout_and_ancestor_backup_and_resumes() -> None:
    space = SearchSpace(
        acquisition_actions=("fixed:1:identity", "fixed:7:all8"),
        fusion_actions=("logit_mean",),
        threshold_actions=(-20, 0),
        beta_actions=(0,),
        component_actions=(0,),
    )

    def objective(spec: WorkflowSpec) -> dict:
        reward = 1.0 if spec.model_mask == 7 and spec.threshold_q == -20 else 0.0
        return {"workflow_id": spec.workflow_id, "reward": reward, "mean_forwards": 24.0}

    uninterrupted = MCTSSearch(objective, space=space, seed=7, exploration=0.2)
    uninterrupted.run(40)
    partial = MCTSSearch(objective, space=space, seed=7, exploration=0.2)
    partial.run(17)
    resumed = MCTSSearch.from_state_dict(
        json.loads(json.dumps(partial.state_dict())), objective
    )
    resumed.run(23)
    assert resumed.state_dict() == uninterrupted.state_dict()
    assert uninterrupted.nodes[uninterrupted.root_id].visits == 40
    assert max(item["reward"] for item in uninterrupted.ranked_evaluations()) == 1.0
    assert any(node.parent_id is not None and node.visits > 0 for node in uninterrupted.nodes.values())
    with pytest.raises(RuntimeError, match="context"):
        MCTSSearch.from_state_dict(
            json.loads(json.dumps(partial.state_dict())),
            objective,
            expected_context_id="different-run",
        )


def test_atom_cache_rejects_test_before_opening_data(tmp_path) -> None:
    cache = tmp_path / "sealed_test" / "atoms"
    cache.mkdir(parents=True)
    (cache / "index.json").write_text(
        json.dumps(
            {
                "schema": ATOM_CACHE_SCHEMA,
                "split": "test",
                "test_access": True,
            }
        )
    )
    with pytest.raises(ValueError, match="forbidden"):
        RobustAtomCache(cache, strict_repository_bindings=False)


def test_atom_cache_rejects_test_schema_without_test_in_path(tmp_path) -> None:
    cache = tmp_path / "atoms"
    cache.mkdir()
    (cache / "index.json").write_text(
        json.dumps(
            {
                "schema": ATOM_CACHE_SCHEMA,
                "split": "test",
                "test_access": True,
                "complete": True,
                "parity": {"passed": True},
            }
        )
    )
    with pytest.raises(ValueError, match="validation"):
        RobustAtomCache(cache, strict_repository_bindings=False)
