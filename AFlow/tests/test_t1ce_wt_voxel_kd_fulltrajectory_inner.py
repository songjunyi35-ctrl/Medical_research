from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_inner as inner


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972"
SPLIT = ROOT / "data/datasets/brats2024_t1ce_fulltrajectory_common_inner_seed20260930_v1"
ARTIFACT = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling/NATIVE_WT_VOLUMES.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids(path: Path) -> list[str]:
    return [json.loads(line)["case_id"] for line in path.read_text().splitlines() if line]


def test_frozen_document_and_split_hashes() -> None:
    assert sha256(RECORD / "IDEA.md") == inner.IDEA_SHA256
    assert sha256(RECORD / "PREREG.md") == inner.PREREG_SHA256
    assert sha256(SPLIT / "split_manifest.json") == inner.SPLIT_SHA256
    assert sha256(ARTIFACT) == inner.NATIVE_ARTIFACT_SHA256


def test_common_inner_roles_are_exact_and_subject_clean() -> None:
    control = ids(SPLIT / "control_fit_cases.jsonl")
    candidate = ids(SPLIT / "candidate_fit_cases.jsonl")
    tune = ids(SPLIT / "common_inner_tune_cases.jsonl")
    assert len(control) == 648 and len(candidate) == 864 and len(tune) == 108
    assert set(control) < set(candidate)
    assert len(set(candidate) - set(control)) == 216
    assert not set(control) & set(tune)
    assert not set(candidate) & set(tune)
    subject = lambda case: case.rsplit("-", 1)[0]
    assert not {subject(x) for x in control} & {subject(x) for x in tune}
    assert not {subject(x) for x in candidate} & {subject(x) for x in tune}


def test_both_arms_have_same_compute_budget() -> None:
    assert inner.SAMPLER_DRAWS == 972
    assert inner.CONSUMED_DRAWS == 968
    assert inner.STEPS_PER_EPOCH == 121
    assert inner.FIT["inner648_control"][:2] == (648, 297)
    assert inner.FIT["inner864_candidate"][:2] == (864, 395)


def test_native_weights_are_finite_for_both_fit_pools() -> None:
    artifact = json.loads(ARTIFACT.read_text())["native_wt_voxels"]
    for name in ("control_fit_cases.jsonl", "candidate_fit_cases.jsonl"):
        weights = native_sampling_weights(ids(SPLIT / name), artifact)
        assert bool(torch.isfinite(weights).all())
        assert bool((weights > 0).all())


def test_random_initial_state_is_seed_exact() -> None:
    states = []
    for _ in range(2):
        torch.manual_seed(20260931)
        model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1)
        states.append(inner.state_dict_sha256(model.state_dict()))
    assert states[0] == states[1]
    assert sum(p.numel() for p in model.parameters()) == 4_700_897


def test_lr_endpoints() -> None:
    assert inner.learning_rate_for_epoch(
        1, base_lr=2e-4, min_lr=1e-5, constant_epochs=100, total_epochs=200
    ) == 2e-4
    assert inner.learning_rate_for_epoch(
        200, base_lr=2e-4, min_lr=1e-5, constant_epochs=100, total_epochs=200
    ) == 1e-5
