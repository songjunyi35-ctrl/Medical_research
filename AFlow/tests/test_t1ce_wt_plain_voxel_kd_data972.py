from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import WeightedRandomSampler

from scripts.train_t1ce_wt_plain_long_schedule import learning_rate_for_epoch
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import binary_voxel_kl
from scripts.train_t1ce_wt_plain_voxel_kd_data972 import (
    AMENDMENT_SHA256,
    ARM_COUNTS,
    ARM_MANIFEST_SHA256,
    CONSUMED_DRAWS,
    FEASIBILITY_SHA256,
    IDEA_SHA256,
    PREREG_SHA256,
    SAMPLER_DRAWS,
    STEPS_PER_EPOCH,
    case_order_sha256,
    optimizer_contract,
    state_dict_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_documents_match_runner_contract() -> None:
    assert sha256(RECORD / "IDEA.md") == IDEA_SHA256
    assert sha256(RECORD / "PREREG.md") == PREREG_SHA256
    assert sha256(RECORD / "AMENDMENT_001_MATCHED_CONTROL.md") == AMENDMENT_SHA256
    assert sha256(RECORD / "FEASIBILITY.json") == FEASIBILITY_SHA256


def test_arm_population_and_manifest_contract() -> None:
    assert ARM_COUNTS == {
        "data756_control": (756, 346),
        "data972_candidate": (972, 444),
    }
    paths = {
        "data756_control": ROOT / "data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl",
        "data972_candidate": ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl",
    }
    for arm, path in paths.items():
        assert sha256(path) == ARM_MANIFEST_SHA256[arm]
        assert len(path.read_text().splitlines()) == ARM_COUNTS[arm][0]


def test_fixed_draw_and_step_contract() -> None:
    assert SAMPLER_DRAWS == 756
    assert CONSUMED_DRAWS == 752
    assert STEPS_PER_EPOCH == 94
    assert (SAMPLER_DRAWS // 8) == STEPS_PER_EPOCH
    assert STEPS_PER_EPOCH * 8 == CONSUMED_DRAWS


@pytest.mark.parametrize("population", [756, 972])
def test_sampler_draws_from_full_population(population: int) -> None:
    weights = torch.linspace(0.5, 4.0, population, dtype=torch.double)
    generator = torch.Generator().manual_seed(20260846)
    indices = list(WeightedRandomSampler(
        weights, num_samples=SAMPLER_DRAWS, replacement=True, generator=generator
    ))
    assert len(indices) == 756
    assert min(indices) >= 0
    assert max(indices) < population
    if population == 972:
        assert any(index >= 756 for index in indices)


def test_voxel_kl_matches_direct_formula_and_has_finite_gradient() -> None:
    student = torch.tensor(
        [[[[[-2.0, -0.3], [0.7, 2.1]]]]], dtype=torch.float64, requires_grad=True
    )
    teacher = torch.tensor(
        [[[[[-1.5, 0.2], [1.1, 1.6]]]]], dtype=torch.float64
    )
    actual = binary_voxel_kl(student, teacher, temperature=2.0)
    student_binary = torch.cat((torch.zeros_like(student), student), dim=1) / 2.0
    teacher_binary = torch.cat((torch.zeros_like(teacher), teacher), dim=1) / 2.0
    expected = F.kl_div(
        F.log_softmax(student_binary, dim=1),
        F.softmax(teacher_binary, dim=1),
        reduction="none",
    ).sum(dim=1).mean() * 4.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert float(student.grad.norm()) > 0


def test_teacher_detach_and_total_loss_gradient() -> None:
    student = torch.randn(2, 1, 3, 4, 5, requires_grad=True)
    teacher_parameter = nn.Parameter(torch.randn(2, 1, 3, 4, 5), requires_grad=False)
    supervised = torch.sigmoid(student).mean()
    kd = binary_voxel_kl(student, teacher_parameter.detach(), temperature=2.0)
    total = supervised + 0.3 * kd
    total.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher_parameter.grad is None


def test_state_dict_hash_is_order_and_value_sensitive() -> None:
    left = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    same = {"a": left["a"].clone(), "b": left["b"].clone()}
    changed = {"a": torch.tensor([1.0, 2.1]), "b": left["b"].clone()}
    assert state_dict_sha256(left) == state_dict_sha256(same)
    assert state_dict_sha256(left) != state_dict_sha256(changed)


def test_case_order_hash_has_no_trailing_newline() -> None:
    values = ["case-a", "case-b", "case-a"]
    expected = hashlib.sha256(b"case-a\ncase-b\ncase-a").hexdigest()
    assert case_order_sha256(values) == expected


def test_optimizer_contract_is_explicit() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=1e-5, foreach=False, fused=False
    )
    contract = optimizer_contract(optimizer)
    assert contract["class"] == "AdamW"
    assert contract["betas"] == [0.9, 0.999]
    assert contract["eps"] == 1e-8
    assert contract["weight_decay"] == 1e-5
    assert contract["foreach"] is False
    assert contract["fused"] is False


def test_late_cosine_grid_matches_archived_protocol() -> None:
    values = {
        epoch: learning_rate_for_epoch(
            epoch, base_lr=2e-4, min_lr=1e-5,
            constant_epochs=100, total_epochs=200,
        )
        for epoch in (169, 176, 184, 192, 200)
    }
    assert values[169] == pytest.approx(5.2377588023741865e-05)
    assert values[200] == pytest.approx(1e-5)
    assert all(values[a] > values[b] for a, b in zip(values, list(values)[1:]))

