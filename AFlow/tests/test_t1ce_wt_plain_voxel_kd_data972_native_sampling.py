from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from sammed3d.t1ce_wt_variable_resolution import VariableResolutionBrainCropDataset
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights
from scripts.train_t1ce_wt_plain_voxel_kd_continuation import binary_voxel_kl
import scripts.train_t1ce_wt_plain_voxel_kd_data972_native_sampling as native


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-08-21_voxel_kd_data972_native_sampling"
ARTIFACT = RECORD / "NATIVE_WT_VOLUMES.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_frozen_documents_and_artifact_hashes() -> None:
    assert sha256(RECORD / "IDEA.md") == native.IDEA_SHA256
    assert sha256(RECORD / "PREREG.md") == native.PREREG_SHA256
    assert sha256(RECORD / "AMENDMENT_001_BRAINCROP_NATIVE_VOLUME_DEFINITION.md") == native.AMENDMENT_001_SHA256
    assert sha256(RECORD / "AMENDMENT_002_CANONICAL_RAS_BEFORE_CROP.md") == native.AMENDMENT_002_SHA256
    assert sha256(RECORD / "AMENDMENT_003_FREEZE_NATIVE_ARTIFACT.md") == native.AMENDMENT_003_SHA256
    assert sha256(RECORD / "AMENDMENT_004_HISTORICAL_ADAMW_DEFAULTS.md") == native.AMENDMENT_004_SHA256
    assert sha256(ARTIFACT) == native.NATIVE_ARTIFACT_SHA256
    assert sha256(ROOT / "scripts/build_t1ce_wt_native_volume_artifact.py") == native.NATIVE_BUILDER_SHA256


def test_artifact_all_gates_and_population_counts() -> None:
    value = load_artifact()
    assert value["status"] == "passed"
    assert value["audit"]["all_gates_passed"] is True
    assert value["audit"]["native_volume_mismatch_count"] == 0
    assert value["audit"]["control_weight_values_exact"] is True
    assert value["audit"]["control_weight_max_abs_difference"] == 0.0
    assert len(value["native_wt_voxels"]) == 972
    assert min(value["native_wt_voxels"].values()) > 0


def test_control_native_weights_equal_historical_dataset_exactly() -> None:
    value = load_artifact()
    case_ids = [json.loads(line)["case_id"] for line in (
        ROOT / "data/datasets/brats2024_t1ce_plain_locator_confirm_seed20260817_v1/train_cases.jsonl"
    ).read_text().splitlines() if line]
    generated = native_sampling_weights(case_ids, value["native_wt_voxels"])
    assert native.tensor_sha256(generated) == native.CONTROL_WEIGHT_SHA256
    historical_cache = ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_resolution_20260818/formal_128"
    if historical_cache.is_dir():
        historical = VariableResolutionBrainCropDataset(
            historical_cache,
            expected_resolution=128,
            expected_manifest_sha256=native.ARM_MANIFEST_SHA256["native756_control"],
            case_ids=case_ids,
        ).sampling_weights()
        assert torch.equal(generated, historical)
    else:
        # 2026-09-06: the 756-case formal_128 cache was deleted under the standing
        # user authorization after the 2026-09-02 byte-exact redundancy audit; the
        # 972-case superset and the archived audit must remain in place instead.
        audit = json.loads((
            ROOT / "archive/reviews/2026-09-02_disk_reclaim_audit/REDUNDANT_FORMAL128_AUDIT_V2.json"
        ).read_text(encoding="utf-8"))
        assert audit["all_tensors_byte_exact"] is True
        assert audit["image_byte_exact_cases"] == 756 and audit["label_byte_exact_cases"] == 756
        assert audit["rebuildable_from_raw_and_archived_builder"] is True
        assert (ROOT / "workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json").is_file()


def test_candidate_native_weight_sha_and_range() -> None:
    value = load_artifact()
    case_ids = [json.loads(line)["case_id"] for line in (
        ROOT / "data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl"
    ).read_text().splitlines() if line]
    weights = native_sampling_weights(case_ids, value["native_wt_voxels"])
    assert native.tensor_sha256(weights) == native.CANDIDATE_WEIGHT_SHA256
    assert float(weights.min()) == 0.5
    assert float(weights.max()) == 4.0


def test_crop_native_definition_excludes_four_boundary_voxels_in_audit_case() -> None:
    value = load_artifact()
    case_id = "BraTS-GLI-02660-100"
    assert value["full_fov_native_wt_voxels_audit_only"][case_id] == 115183
    assert value["native_wt_voxels"][case_id] == 115179
    assert value["source_orientations"][case_id] == {
        "stored": ["L", "A", "S"], "canonical": ["R", "A", "S"]
    }


def test_weight_formula_uses_selected_population_median() -> None:
    volumes = {"a": 1, "b": 4, "c": 16}
    values = native_sampling_weights(["a", "b", "c"], volumes)
    expected = torch.tensor([2.0, 1.0, 0.5], dtype=torch.double)
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    subset = native_sampling_weights(["a", "b"], volumes)
    expected_subset = torch.tensor([np.sqrt(2.5), np.sqrt(2.5 / 4.0)], dtype=torch.double)
    torch.testing.assert_close(subset, expected_subset, rtol=0, atol=0)


@pytest.mark.parametrize("arm,population", [("native756_control", 756), ("native972_candidate", 972)])
def test_fixed_sampler_draw_count(arm: str, population: int) -> None:
    weights = torch.ones(population, dtype=torch.double)
    indices = list(WeightedRandomSampler(
        weights, num_samples=756, replacement=True,
        generator=torch.Generator().manual_seed(20260846),
    ))
    assert len(indices) == 756
    assert min(indices) >= 0 and max(indices) < population
    if arm == "native972_candidate":
        assert any(index >= 756 for index in indices)


def test_native_contract_patch_replaces_legacy_weight_source() -> None:
    original_artifact, original_arm = native._ACTIVE_ARTIFACT, native._ACTIVE_ARM
    try:
        native._ACTIVE_ARTIFACT = load_artifact()
        native._ACTIVE_ARM = "native972_candidate"
        value = native.patch_native_contract({
            "sampling_contract": {"weight_source": "legacy resized labels"},
            "method_documents": {"old": "value"},
        })
        assert value["sampling_contract"]["native_weight_sha256"] == native.CANDIDATE_WEIGHT_SHA256
        assert "source-grid WT voxels" in value["sampling_contract"]["weight_source"]
        assert value["method_documents"] == native.method_documents()
    finally:
        native._ACTIVE_ARTIFACT, native._ACTIVE_ARM = original_artifact, original_arm


def test_voxel_kd_formula_and_gradient_remain_unchanged() -> None:
    student = torch.tensor([[[[[-1.0, 0.4], [1.2, 2.0]]]]], requires_grad=True)
    teacher = torch.tensor([[[[[-0.7, 0.8], [0.5, 1.7]]]]])
    actual = binary_voxel_kl(student, teacher, temperature=2.0)
    s = torch.cat((torch.zeros_like(student), student), dim=1) / 2.0
    t = torch.cat((torch.zeros_like(teacher), teacher), dim=1) / 2.0
    expected = F.kl_div(F.log_softmax(s, 1), F.softmax(t, 1), reduction="none").sum(1).mean() * 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert float(student.grad.norm()) > 0


def test_method_document_contract_includes_all_preimplementation_fixes() -> None:
    docs = native.method_documents()
    assert set(docs) == {
        "idea_sha256", "prereg_sha256", "amendment_001_sha256",
        "amendment_002_sha256", "amendment_003_sha256",
        "amendment_004_sha256",
        "native_artifact_sha256", "native_builder_sha256",
    }
