from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import scripts.train_t1ce_wt_coverage_small_lesion_replay_kd as method


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-08-24_coverage_small_lesion_replay_kd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights() -> torch.Tensor:
    return torch.linspace(0.5, 4.0, method.POPULATION, dtype=torch.float64)


def candidate(seed: int = 20260884) -> method.CoverageSmallLesionReplaySampler:
    return method.CoverageSmallLesionReplaySampler(
        weights(), num_samples=756, replacement=True, seed=seed
    )


def test_frozen_documents() -> None:
    assert sha256(RECORD / "IDEA.md") == method.IDEA_SHA256
    assert sha256(RECORD / "PREREG.md") == method.PREREG_SHA256
    assert sha256(RECORD / "PREREG.json") == method.PREREG_JSON_SHA256
    assert sha256(RECORD / "SAMPLER_SPEC.md") == method.SAMPLER_SPEC_SHA256
    assert sha256(RECORD / "AMENDMENT_001_PRE_GPU_COMPLEMENT_COVERAGE.md") == method.SAMPLER_AMENDMENT_SHA256
    assert sha256(RECORD / "AMENDMENT_002_PRE_MODEL_SEED_VALIDATOR_COMPATIBILITY.md") == method.SEED_AMENDMENT_SHA256
    assert sha256(RECORD / "AMENDMENT_003_PRE_FORMAL_SELF_CONTAINED_SMOKE_CONTRACT.md") == method.REPORT_AMENDMENT_SHA256
    assert sha256(RECORD / "AMENDMENT_004_PRE_FORMAL_FINAL_WRITER_ORDER.md") == method.WRITER_AMENDMENT_SHA256


def test_output_length_range_and_epoch_allocation() -> None:
    sampler = candidate()
    values = list(sampler)
    assert len(values) == 756
    assert min(values) >= 0 and max(values) < 972
    audit = sampler.epoch_audits[0]
    assert audit["coverage_draws"] == 680
    assert audit["replay_consumed"] == 72
    assert audit["replay_dropped"] == 4
    assert audit["requested_draws"] == 756


def test_same_seed_is_exact_for_four_epochs_and_different_seed_changes() -> None:
    first, second, different = candidate(), candidate(), candidate(20260885)
    first_epochs = [list(first) for _ in range(4)]
    second_epochs = [list(second) for _ in range(4)]
    different_epochs = [list(different) for _ in range(4)]
    assert first_epochs == second_epochs
    assert first_epochs != different_epochs


def test_every_adjacent_two_epoch_coverage_union_contains_population() -> None:
    sampler = candidate()
    for _ in range(5):
        list(sampler)
    for left, right in zip(
        sampler.epoch_coverage_indices, sampler.epoch_coverage_indices[1:]
    ):
        assert len(set(left) | set(right)) == 972


def test_each_coverage_epoch_and_transition_is_exact() -> None:
    sampler = candidate()
    for _ in range(5):
        list(sampler)
    for coverage in sampler.epoch_coverage_indices:
        assert len(coverage) == 680
        assert len(set(coverage)) == 680
    assert sampler.epoch_audits[0]["coverage_retained_from_previous"] is None
    assert sampler.epoch_audits[0]["coverage_complement_from_previous"] is None
    for audit in sampler.epoch_audits[1:]:
        assert audit["coverage_retained_from_previous"] == 388
        assert audit["coverage_complement_from_previous"] == 292


@pytest.mark.parametrize(
    "bad_weights,num_samples,replacement",
    [
        (torch.ones(971), 756, True),
        (torch.ones(972), 755, True),
        (torch.ones(972), 756, False),
        (torch.cat((torch.ones(971), torch.tensor([0.0]))), 756, True),
        (torch.cat((torch.ones(971), torch.tensor([float("nan")]))), 756, True),
    ],
)
def test_candidate_rejects_contract_violations(
    bad_weights: torch.Tensor, num_samples: int, replacement: bool
) -> None:
    with pytest.raises(ValueError):
        method.CoverageSmallLesionReplaySampler(
            bad_weights,
            num_samples=num_samples,
            replacement=replacement,
            seed=20260884,
        )


def test_control_factory_is_historical_weighted_sampler_exact() -> None:
    expected_generator = torch.Generator().manual_seed(17)
    actual_generator = torch.Generator().manual_seed(17)
    expected = list(WeightedRandomSampler(
        weights(), 756, replacement=True, generator=expected_generator
    ))
    actual = list(method.build_sampler(
        "native_replacement_control",
        weights(),
        num_samples=756,
        replacement=True,
        generator=actual_generator,
        seed=20260884,
    ))
    assert actual == expected


class IndexDataset(Dataset[int]):
    def __len__(self) -> int:
        return 972

    def __getitem__(self, index: int) -> int:
        return index


def test_dataloader_drops_only_reserved_replay_tail() -> None:
    sampler = candidate()
    direct = list(sampler)
    replay_tail = direct[-4:]
    sampler = candidate()
    loader = DataLoader(
        IndexDataset(), batch_size=8, sampler=sampler, drop_last=True, num_workers=0
    )
    consumed = [int(value) for batch in loader for value in batch]
    assert len(loader) == 94
    assert len(consumed) == 752
    assert consumed == direct[:752]
    assert sampler.epoch_audits[0]["dropped_index_sha256"] == method.tensor_index_sha256(replay_tail)


def test_candidate_sampler_gates_pass_for_formal_shape() -> None:
    sampler = candidate()
    for _ in range(32):
        list(sampler)
    original = method._ACTIVE_SAMPLER
    try:
        method._ACTIVE_SAMPLER = sampler
        assert all(method.candidate_sampler_gates(32).values())
    finally:
        method._ACTIVE_SAMPLER = original


def test_contract_patch_reports_hybrid_policy() -> None:
    sampler = candidate()
    list(sampler)
    old_arm, old_sampler = method._ACTIVE_ARM, method._ACTIVE_SAMPLER
    try:
        method._ACTIVE_ARM = "coverage_replay_candidate"
        method._ACTIVE_SAMPLER = sampler
        value = method.patch_contract({
            "sampling_contract": {"replacement": True},
            "method_documents": {"old": "value"},
            "data_role": {"role": "legacy"},
        })
        contract = value["sampling_contract"]
        assert contract["coverage_consumed_per_epoch"] == 680
        assert contract["replay_consumed_per_epoch"] == 72
        assert contract["replay_dropped_per_epoch"] == 4
        assert "previous_epoch_complement" in contract["replacement"]
        assert value["method_documents"] == method.method_documents()
        assert value["data_role"]["role"] == "coverage_replay_candidate"
    finally:
        method._ACTIVE_ARM, method._ACTIVE_SAMPLER = old_arm, old_sampler


def test_seed_compatibility_presents_old_seed_then_restores_new_seed() -> None:
    args = SimpleNamespace(seed=20260884)
    observed: list[int] = []
    def inherited(value: SimpleNamespace) -> tuple[list[str], str]:
        observed.append(value.seed)
        return ["case"], "manifest"
    result = method.validate_with_seed_compatibility(args, inherited)
    assert result == (["case"], "manifest")
    assert observed == [20260846]
    assert args.seed == 20260884


def test_seed_compatibility_restores_after_exception_and_rejects_wrong_seed() -> None:
    args = SimpleNamespace(seed=20260884)
    def failing(value: SimpleNamespace) -> tuple[list[str], str]:
        assert value.seed == 20260846
        raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        method.validate_with_seed_compatibility(args, failing)
    assert args.seed == 20260884
    with pytest.raises(ValueError, match="seed20260884"):
        method.validate_with_seed_compatibility(SimpleNamespace(seed=1), failing)


@pytest.mark.parametrize(
    "arm,expected_strategy",
    [
        ("native_replacement_control", "native_volume_weighted_replacement"),
        (
            "coverage_replay_candidate",
            "adjacent_complement_coverage_680_plus_native_weighted_replay_72",
        ),
    ],
)
def test_top_level_report_contract_can_be_attached(
    arm: str, expected_strategy: str
) -> None:
    old_arm, old_sampler = method._ACTIVE_ARM, method._ACTIVE_SAMPLER
    try:
        method._ACTIVE_ARM = arm
        method._ACTIVE_SAMPLER = candidate() if arm == "coverage_replay_candidate" else None
        if method._ACTIVE_SAMPLER is not None:
            list(method._ACTIVE_SAMPLER)
        report = {
            "sampling_contract": method.sampler_contract(),
            "method_documents": method.method_documents(),
        }
        patched = method.patch_contract(report)
        assert patched["sampling_contract"]["strategy"] == expected_strategy
        assert patched["method_documents"] == method.method_documents()
        assert patched["sampling_contract"]["native_volume_artifact_sha256"] == method.native.NATIVE_ARTIFACT_SHA256
        assert patched["method_documents"]["final_writer_amendment_sha256"] == method.WRITER_AMENDMENT_SHA256
    finally:
        method._ACTIVE_ARM, method._ACTIVE_SAMPLER = old_arm, old_sampler
