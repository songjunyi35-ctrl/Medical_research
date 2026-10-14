from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
import scripts.train_t1ce_wt_voxel_kd_fulltrajectory_final as final


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-08-24_voxel_kd_fulltrajectory_972"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_documents_and_selection() -> None:
    assert sha256(RECORD / "IDEA.md") == final.IDEA_SHA256
    assert sha256(RECORD / "PREREG.md") == final.PREREG_SHA256
    assert sha256(RECORD / "INNER_SELECTION.json") == final.INNER_SELECTION_SHA256
    selection = json.loads((RECORD / "INNER_SELECTION.json").read_text())
    assert selection["selected_epoch"] == 168
    assert selection["final_retraining_authorized"] is True
    assert selection["fixed_validation108_access"] is False


def test_arm_populations_and_manifests() -> None:
    assert final.ARM["full756_control"][:2] == (756, 346)
    assert final.ARM["full972_candidate"][:2] == (972, 444)
    control = ROOT / "data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl"
    candidate = ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    assert sha256(control) == final.ARM["full756_control"][2]
    assert sha256(candidate) == final.ARM["full972_candidate"][2]


def test_both_final_arms_share_compute() -> None:
    assert final.SAMPLER_DRAWS == 972
    assert final.CONSUMED_DRAWS == 968
    assert final.STEPS_PER_EPOCH == 121


def test_final_random_initialization_seed_is_exact() -> None:
    values = []
    for _ in range(2):
        torch.manual_seed(20260932)
        model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1)
        values.append(final.state_dict_sha256(model.state_dict()))
    assert values[0] == values[1]
    assert sum(p.numel() for p in model.parameters()) == 4_700_897
