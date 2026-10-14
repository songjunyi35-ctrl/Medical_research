#!/usr/bin/env python3
"""Promote the completed exact atom cache under the frozen BF16 parity review.

This is a one-purpose validation-only tool.  It preserves the original failed
receipt and cannot alter tolerances from the command line.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_mcts import (  # noqa: E402
    ATOM_CACHE_SCHEMA,
    ENSEMBLE_REPORT_SHA256,
    ENSEMBLE_TTA_REPORT_SHA256,
    FORMAL_AGGREGATE_PARITY_TOLERANCE,
    FORMAL_PER_CASE_PARITY_TOLERANCE,
    LOCKED_SCAN_MACRO_DSC,
    LOCKED_SUBJECT_MACRO_DSC,
    MODEL_SHAPE,
    REFLECTION_AXES,
    ROBUST_CHECKPOINT_SHA256,
    ROBUST_SEEDS,
    VALIDATION_CACHE_INDEX_SHA256,
    WorkflowEvaluator,
    WorkflowSpec,
    sha256_file,
    stable_id,
)
from scripts.cache_t1ce_wt_robust_inference_atoms import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    DEFAULT_VALIDATION_CACHE,
    ENSEMBLE_REPORT,
    ENSEMBLE_TTA_REPORT,
)
from scripts.train_sammed3d_t1ce_wt_residual_refiner import (  # noqa: E402
    RefinerDataset,
    write_json_atomic,
)


CACHE_ROOT = ROOT / "workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32"
PREREG = ROOT / "MCTS_ATOM_PARITY_PREREG_20260813.md"
EXPECTED_FAILED = {
    "actual_scan_macro_dsc": 0.7004216820470339,
    "actual_subject_macro_dsc": 0.6953731986886416,
    "maximum_per_case_absolute_dice_error": 0.00027145610058565417,
}


def _per_case_errors(
    actual: dict,
    expected_by_case: dict[str, dict],
    field: str,
    case_ids: list[str],
) -> list[float]:
    return [
        abs(float(value) - float(expected_by_case[case_id][field]))
        for case_id, value in zip(case_ids, actual["case_dice"], strict=True)
    ]


def main() -> None:
    if not PREREG.exists():
        raise FileNotFoundError("parity-review preregistration is missing")
    if (CACHE_ROOT / "index.json").exists():
        raise FileExistsError("atom cache is already published")
    state_path = CACHE_ROOT / "BUILD_STATE.json"
    failed_path = CACHE_ROOT / "PARITY_FAILED.json"
    atoms_path = CACHE_ROOT / "member_view_logits.f32"
    state = json.loads(state_path.read_text())
    failed = json.loads(failed_path.read_text())
    if any(abs(float(failed[key]) - value) > 1e-15 for key, value in EXPECTED_FAILED.items()):
        raise RuntimeError("original parity failure receipt does not match the preregistered values")
    shape = (108, 24, *MODEL_SHAPE)
    if (
        state.get("completed_cases") != 108
        or state.get("shape") != list(shape)
        or state.get("dtype") != "float32"
        or tuple(state.get("checkpoint_sha256", [])) != ROBUST_CHECKPOINT_SHA256
        or state.get("validation_cache_index_sha256") != VALIDATION_CACHE_INDEX_SHA256
    ):
        raise RuntimeError("completed build-state binding mismatch")

    checkpoints = [(ROOT / path).resolve() for path in DEFAULT_CHECKPOINTS]
    if tuple(sha256_file(path) for path in checkpoints) != ROBUST_CHECKPOINT_SHA256:
        raise RuntimeError("robust checkpoint hash mismatch")
    atom_order = [
        {
            "atom_index": model_slot * 8 + view_slot,
            "model_slot": model_slot,
            "seed": ROBUST_SEEDS[model_slot],
            "checkpoint_sha256": ROBUST_CHECKPOINT_SHA256[model_slot],
            "reflection_axes": list(axes),
            "inverse_reflection_axes": list(axes),
        }
        for model_slot in range(3)
        for view_slot, axes in enumerate(REFLECTION_AXES)
    ]
    if state.get("atom_order_sha256") != stable_id(atom_order):
        raise RuntimeError("atom order binding mismatch")
    expected_bytes = int(np.prod(shape)) * np.dtype("float32").itemsize
    if atoms_path.stat().st_size != expected_bytes:
        raise RuntimeError("atom file byte-size mismatch")

    validation_cache = (ROOT / DEFAULT_VALIDATION_CACHE).resolve()
    if sha256_file(validation_cache / "index.json") != VALIDATION_CACHE_INDEX_SHA256:
        raise RuntimeError("validation cache index hash mismatch")
    data = RefinerDataset(validation_cache)
    if data.index.get("split") != "validation" or data.index.get("test_access") is not False:
        raise ValueError("promotion only accepts the development validation cache")
    actual_coarse_sha = sha256_file(validation_cache / "sam_logits.f16")
    actual_images_sha = sha256_file(data.source / "images.f32")
    actual_labels_sha = sha256_file(data.source / "labels.u8")
    if actual_coarse_sha != data.index["logits"]["sha256"]:
        raise RuntimeError("coarse blob hash mismatch")
    if actual_images_sha != data.index["source_cache"]["images_sha256"]:
        raise RuntimeError("image blob hash mismatch")
    if actual_labels_sha != data.index["source_cache"]["labels_sha256"]:
        raise RuntimeError("label blob hash mismatch")

    tta_path = (ROOT / ENSEMBLE_TTA_REPORT).resolve()
    identity_path = (ROOT / ENSEMBLE_REPORT).resolve()
    if sha256_file(tta_path) != ENSEMBLE_TTA_REPORT_SHA256:
        raise RuntimeError("historical ensemble-TTA report hash mismatch")
    if sha256_file(identity_path) != ENSEMBLE_REPORT_SHA256:
        raise RuntimeError("historical identity-ensemble report hash mismatch")
    historical_tta = json.loads(tta_path.read_text())
    historical_identity = json.loads(identity_path.read_text())
    tta_by_case = {item["case_id"]: item for item in historical_tta["per_case"]}
    identity_by_case = {item["case_id"]: item for item in historical_identity["per_case"]}

    atoms = np.memmap(atoms_path, dtype=np.float32, mode="r", shape=shape)
    data._open()
    assert data._labels is not None
    evaluator = WorkflowEvaluator(atoms, data._labels, data.case_ids)
    locked = evaluator.evaluate(WorkflowSpec.locked_reference(), range(108))
    raw_tta = evaluator.evaluate(
        WorkflowSpec(threshold_q=0, min_component_voxels=0), range(108)
    )
    identity = evaluator.evaluate(
        WorkflowSpec(view_policy="identity", threshold_q=0, min_component_voxels=0),
        range(108),
    )
    errors = _per_case_errors(
        locked, tta_by_case, "ensemble_tta_locked_postprocess_dsc", data.case_ids
    )
    errors += _per_case_errors(
        raw_tta, tta_by_case, "ensemble_tta_threshold0_raw_dsc", data.case_ids
    )
    errors += _per_case_errors(
        identity, identity_by_case, "ensemble_threshold0_raw_dsc", data.case_ids
    )
    identity_seed_metrics: list[float] = []
    for model_slot in range(3):
        member = evaluator.evaluate(
            WorkflowSpec(
                model_mask=1 << model_slot,
                view_policy="identity",
                threshold_q=0,
                min_component_voxels=0,
            ),
            range(108),
        )
        identity_seed_metrics.append(float(member["scan_macro_dsc"]))
        errors += [
            abs(
                float(value)
                - float(identity_by_case[case_id]["individual_threshold0_raw_dsc"][model_slot])
            )
            for case_id, value in zip(data.case_ids, member["case_dice"], strict=True)
        ]

    maximum_per_case_error = max(errors)
    parity = {
        "reference": WorkflowSpec.locked_reference().canonical_dict(),
        "expected_scan_macro_dsc": LOCKED_SCAN_MACRO_DSC,
        "actual_scan_macro_dsc": locked["scan_macro_dsc"],
        "expected_subject_macro_dsc": LOCKED_SUBJECT_MACRO_DSC,
        "actual_subject_macro_dsc": locked["subject_macro_dsc"],
        "aggregate_absolute_tolerance": FORMAL_AGGREGATE_PARITY_TOLERANCE,
        "per_case_absolute_tolerance": FORMAL_PER_CASE_PARITY_TOLERANCE,
        "maximum_per_case_absolute_dice_error": maximum_per_case_error,
        "raw_tta_scan_macro_dsc": raw_tta["scan_macro_dsc"],
        "identity_ensemble_scan_macro_dsc": identity["scan_macro_dsc"],
        "identity_seed_scan_macro_dsc": identity_seed_metrics,
    }
    parity["passed"] = bool(
        abs(locked["scan_macro_dsc"] - LOCKED_SCAN_MACRO_DSC)
        <= FORMAL_AGGREGATE_PARITY_TOLERANCE
        and abs(locked["subject_macro_dsc"] - LOCKED_SUBJECT_MACRO_DSC)
        <= FORMAL_AGGREGATE_PARITY_TOLERANCE
        and maximum_per_case_error <= FORMAL_PER_CASE_PARITY_TOLERANCE
    )
    review = {
        "schema": "aflow.sammed3d.t1ce_wt_robust_inference_atoms.parity_review",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access": False,
        "preregistration": {"path": str(PREREG), "sha256": sha256_file(PREREG)},
        "original_failure": {"path": str(failed_path), "sha256": sha256_file(failed_path)},
        "build_state_sha256": sha256_file(state_path),
        "parity": parity,
        "promotion_runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(CACHE_ROOT / "PARITY_REVIEW.json", review)
    if not parity["passed"]:
        raise RuntimeError("frozen parity-review rules failed; atom cache remains unpublished")

    atoms_sha = sha256_file(atoms_path)
    index = {
        "schema": ATOM_CACHE_SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "split": "validation",
        "test_access": False,
        "geometry": "GT-centered N4 128^3 oracle geometry",
        "target": "BraTS 2024 T1ce WT={1,2,3}; label 4 excluded",
        "uses_gray_loss": False,
        "count": 108,
        "case_ids": data.case_ids,
        "shape": list(shape),
        "dtype": "float32",
        "evidence_level": "exact",
        "atoms_file": atoms_path.name,
        "atoms_sha256": atoms_sha,
        "atom_order": atom_order,
        "checkpoints": [
            {"seed": seed, "path": str(path), "sha256": digest}
            for seed, path, digest in zip(
                ROBUST_SEEDS, checkpoints, ROBUST_CHECKPOINT_SHA256, strict=True
            )
        ],
        "validation_cache": str(validation_cache),
        "validation_cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
        "coarse_logits_sha256": actual_coarse_sha,
        "source_images_sha256": actual_images_sha,
        "source_labels_sha256": actual_labels_sha,
        "aggregation_contract": "historical view-major/model-minor fixed-order float32 fusion",
        "parity": parity,
        "historical_parity_evidence": {
            "ensemble_tta_report_sha256": ENSEMBLE_TTA_REPORT_SHA256,
            "ensemble_report_sha256": ENSEMBLE_REPORT_SHA256,
            "per_case_passed": True,
            "parity_review_sha256": sha256_file(CACHE_ROOT / "PARITY_REVIEW.json"),
        },
        "build_runner_sha256": state["runner_sha256"],
        "build_core_sha256": state["core_sha256"],
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "core_sha256": sha256_file(ROOT / "sammed3d/t1ce_wt_mcts.py"),
        "torch": state["torch"],
        "cuda": state["cuda"],
        "device": state["device_name"],
        "autocast": "bfloat16 model forward; float32 stored atoms and fusion",
    }
    write_json_atomic(CACHE_ROOT / "index.json", index)
    print(json.dumps({"published": str(CACHE_ROOT), "parity": parity}, indent=2))


if __name__ == "__main__":
    main()
