from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sammed3d.t1ce_wt_variable_resolution import (
    CACHE_SCHEMA,
    VariableResolutionBrainCropDataset,
    cubic_shape,
)
from scripts.analyze_t1ce_wt_plain_resolution_pair import analyze
from scripts.build_t1ce_wt_braincrop_cache_variable_resolution import (
    assert_development_output,
    resolve_case_paths,
    stable_sha256,
)
from scripts.build_t1ce_wt_resolution_inner_split import split_rows


def _make_cache(root: Path, resolution: int, source_volumes: dict[str, int]) -> Path:
    root.mkdir(parents=True)
    case_ids = list(source_volumes)
    shape = cubic_shape(resolution)
    images = np.zeros((len(case_ids), *shape), dtype=np.float32)
    labels = np.zeros((len(case_ids), *shape), dtype=np.uint8)
    for index in range(len(case_ids)):
        images[index, 1:-1, 1:-1, 1:-1] = index + 1
        labels[index, 2:4, 2:4, 2:4] = 1
    (root / "images.f32").write_bytes(images.tobytes())
    (root / "labels.u8").write_bytes(labels.tobytes())
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "schema_version": 1,
                "manifest": "data/allowed_train_cases.jsonl",
                "manifest_sha256": "a" * 64,
                "count": len(case_ids),
                "case_ids": case_ids,
                "model_shape": list(shape),
                "split": "historical_train_cohort_only",
                "test_access": False,
                "outer_selector_access": False,
                "confirmation_access": False,
                "preprocessing": {
                    "spatial": "braincrop_resize",
                    "uses_gt_for_localization": False,
                    "output_resolution": resolution,
                },
                "per_case_source_positive_voxels": source_volumes,
                "source_positive_voxels_sha256": stable_sha256(source_volumes),
                "images_sha256": "b" * 64,
                "labels_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_variable_resolution_dataset_and_sampler_are_shape_independent(tmp_path: Path) -> None:
    volumes = {"BraTS-GLI-00001-100": 10, "BraTS-GLI-00002-100": 40}
    cache16 = _make_cache(tmp_path / "r16", 16, volumes)
    cache32 = _make_cache(tmp_path / "r32", 32, volumes)
    left = VariableResolutionBrainCropDataset(
        cache16, expected_resolution=16, expected_manifest_sha256="a" * 64
    )
    right = VariableResolutionBrainCropDataset(
        cache32, expected_resolution=32, expected_manifest_sha256="a" * 64
    )
    assert tuple(left[0]["image"].shape) == (1, 16, 16, 16)
    assert tuple(right[0]["image"].shape) == (1, 32, 32, 32)
    assert torch.equal(left.sampling_weights(), right.sampling_weights())


def test_resolution_must_be_cubic_multiple_of_16() -> None:
    assert cubic_shape(160) == (160, 160, 160)
    with pytest.raises(ValueError):
        cubic_shape(150)


def test_raw_path_resolution_checks_only_one_matching_part(tmp_path: Path) -> None:
    case_id = "BraTS-GLI-00005-100"
    case_root = tmp_path / "part2" / "Part2" / case_id
    case_root.mkdir(parents=True)
    image = case_root / f"{case_id}-t1c.nii"
    label = case_root / f"{case_id}-seg.nii"
    image.write_bytes(b"image")
    label.write_bytes(b"label")
    assert resolve_case_paths(tmp_path, case_id) == (image, label, "Part2")


@pytest.mark.parametrize("name", ["selector/x", "confirmation/x", "sealed/x", "test270/x"])
def test_builder_rejects_forbidden_output_roles(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        assert_development_output(tmp_path / name)


def test_resolution_inner_split_is_deterministic_and_subject_clean() -> None:
    rows = []
    for subject_index, count in enumerate((1, 2, 3, 4), start=1):
        subject = f"BraTS-GLI-{subject_index:05d}"
        for timepoint in range(count):
            rows.append(
                {
                    "case_id": f"{subject}-{100 + timepoint:03d}",
                    "subject_id": subject,
                }
            )
    fit_a, tune_a = split_rows(rows, seed=17, tune_cases=4)
    fit_b, tune_b = split_rows(list(reversed(rows)), seed=17, tune_cases=4)
    assert fit_a == fit_b
    assert tune_a == tune_b
    assert len(fit_a) == 6 and len(tune_a) == 4
    assert {row["subject_id"] for row in fit_a}.isdisjoint(
        {row["subject_id"] for row in tune_a}
    )


def _training_report(resolution: int, shift: float, point_hit: float = 0.9) -> dict:
    arm = "128_control" if resolution == 128 else "160_candidate"
    per_case = [
        {
            "case_id": f"BraTS-GLI-{index:05d}-100",
            "subject_id": f"BraTS-GLI-{index:05d}",
            "locator_dsc": 0.7 + shift,
            "automatic_point_hits_gt": index < round(10 * point_hit),
        }
        for index in range(10)
    ]
    roles = {
        "fit": {"path": "/dev/fit", "sha256": "f" * 64},
        "inner_tune": {"path": "/dev/tune", "sha256": "e" * 64},
        "fit_cases": 648,
        "fit_subjects": 300,
        "inner_tune_cases": 10,
        "inner_tune_subjects": 10,
        "case_overlap": 0,
        "subject_overlap": 0,
    }
    hyperparameters = {
        "seed": 20260829,
        "epochs": 168,
        "schedule_total_epochs": 200,
        "constant_epochs": 100,
        "batch_size": 8,
        "validation_batch_size": 8,
        "num_workers": 4,
        "init_filters": 16,
        "dropout": 0.1,
        "base_lr": 2e-4,
        "min_lr": 1e-5,
        "weight_decay": 1e-5,
        "max_shift": 8,
        "ce_weight": 0.1,
        "expected_parameter_count": 4_700_897,
        "expected_cohort_sha256": "a" * 64,
        "expected_source_positive_voxels_sha256": "d" * 64,
    }
    return {
        "status": "complete",
        "resolution": resolution,
        "arm": arm,
        "fixed_epoch": 168,
        "fixed_logit_threshold": 0.0,
        "inner_tune_first_access": "after_final_optimizer_step",
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
        "reload_max_abs_difference": 0.0,
        "parameter_count": 4_700_897,
        "data_roles": roles,
        "sampling_contract": {
            "basis": "native",
            "weights_sha256": "b" * 64,
            "source_positive_voxels_sha256": "d" * 64,
            "sampler": "weighted",
            "generator_seed": 20260829,
        },
        "schedule": {
            "trained_epochs": 168,
            "definition_total_epochs": 200,
            "constant_epochs": 100,
            "base_lr": 2e-4,
            "min_lr": 1e-5,
            "epoch168_lr": 1.0e-4,
        },
        "hyperparameters": hyperparameters,
        "validation": {
            "locator_dsc_mean": 0.7 + shift,
            "locator_subject_dsc_mean": 0.7 + shift,
            "automatic_point_hit_rate": point_hit,
            "per_case": per_case,
        },
        "checkpoint": {"path": f"r{resolution}.pth", "sha256": str(resolution)},
    }


def test_preregistered_resolution_gates_pass_clear_gain() -> None:
    result = analyze(_training_report(128, 0.0), _training_report(160, 0.01))
    assert result["all_inner_gates_passed"] is True
    assert result["decision"] == "160_positive_on_preregistered_inner_gates"


def test_resolution_gate_stops_on_small_gain() -> None:
    result = analyze(_training_report(128, 0.0), _training_report(160, 0.004))
    assert result["all_inner_gates_passed"] is False
    assert result["decision"] == "160_negative_stop_without_scale_or_threshold_tuning"


def test_resolution_analysis_rejects_sampler_mismatch() -> None:
    candidate = deepcopy(_training_report(160, 0.01))
    candidate["sampling_contract"]["weights_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="sampler"):
        analyze(_training_report(128, 0.0), candidate)
