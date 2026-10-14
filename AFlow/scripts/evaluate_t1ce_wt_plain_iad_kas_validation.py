#!/usr/bin/env python3
"""Evaluate preregistered KAS-on-voxel-KD checkpoints on fixed validation108."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    read_case_ids,
    sha256_file,
    validate,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_validation"
CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_checkpoint"
TRAINING_REPORT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_continuation"
IMPLEMENTATION_RECEIPT_SCHEMA = (
    "aflow.sammed3d.t1ce_wt_iad_kas_implementation_receipt"
)
PAIRING_RECEIPT_SCHEMA = "aflow.sammed3d.t1ce_wt_iad_kas_pairing_receipt"
KD_VALIDATION_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_validation"
TRAINER_ENTRYPOINT = "scripts/train_t1ce_wt_plain_iad_kas_continuation.py"

ARMS = ("control", "kas")
EPOCHS = (208, 216, 224, 232)
VALIDATION_CASES = 108
VALIDATION_SUBJECTS = 50
VALIDATION_MANIFEST_SHA256 = (
    "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
)
VALIDATION_CACHE_INDEX_SHA256 = (
    "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
)
KD_VALIDATION_SHA256 = (
    "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
)
KD_EPOCH = 200
KD_SCAN = 0.7343843078447713
KD_SUBJECT = 0.7244341433615912
KD_POINT = 0.9351851851851852

# Frozen before formal validation: four same-epoch comparisons followed by four
# historical-KD200 comparisons. Do not change these after validation access.
BOOTSTRAP_SEEDS_SAME_EPOCH_CONTROL = {
    208: 20260910,
    216: 20260911,
    224: 20260912,
    232: 20260913,
}
BOOTSTRAP_SEEDS_HISTORICAL_KD200 = {
    208: 20260914,
    216: 20260915,
    224: 20260916,
    232: 20260917,
}
BOOTSTRAP_SAMPLES = 10_000

ACCESS_FLAGS = (
    "validation_access",
    "outer_selector_access",
    "confirmation_access",
    "test_access",
)
PAIRING_FIELDS = (
    "initial_student_state_sha256",
    "first_batch_case_ids",
    "first_batch_case_ids_sha256",
    "first_batch_raw_image_sha256",
    "first_batch_raw_label_sha256",
    "first_batch_augmented_image_sha256",
    "first_batch_augmented_label_sha256",
    "epoch_case_order_sha256",
    "epoch_cuda_rng_before_sha256",
    "epoch_cuda_rng_after_sha256",
)
TRAINING_GLOBAL_EPOCHS = tuple(range(201, 233))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_false_access(value: dict[str, Any], *, source: str) -> None:
    bad = {key: value.get(key) for key in ACCESS_FLAGS if value.get(key) is not False}
    if bad:
        raise ValueError(f"{source} does not declare all access flags false: {bad}")


def _require_exact_zero(value: Any, *, field: str, source: str) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} lacks numeric {field}") from exc
    if not math.isfinite(number) or number != 0.0:
        raise ValueError(f"{source} {field} must be exactly zero, got {number}")


def _require_finite_numbers(value: Any, *, source: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{source} contains a non-finite diagnostic")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, source=f"{source}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_numbers(item, source=f"{source}[{index}]")
        return
    raise ValueError(f"{source} contains unsupported diagnostic type {type(value)!r}")


def _validate_pairing_evidence(
    value: Any,
    *,
    source: str,
    through_epoch: int = 232,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} is not an object")
    for field in PAIRING_FIELDS:
        item = value.get(field)
        if item is None or item == "" or item == [] or item == {}:
            raise ValueError(f"{source} lacks {field}")
    case_ids = value["first_batch_case_ids"]
    if not isinstance(case_ids, list) or len(case_ids) != 8:
        raise ValueError(f"{source}.first_batch_case_ids must contain batch8")
    if len({str(case_id) for case_id in case_ids}) != 8:
        raise ValueError(f"{source}.first_batch_case_ids contains duplicates")
    if through_epoch not in TRAINING_GLOBAL_EPOCHS:
        raise ValueError(f"{source} has invalid pairing horizon {through_epoch}")
    expected_epoch_keys = {
        str(epoch) for epoch in TRAINING_GLOBAL_EPOCHS if epoch <= through_epoch
    }
    for field in (
        "epoch_case_order_sha256",
        "epoch_cuda_rng_before_sha256",
        "epoch_cuda_rng_after_sha256",
    ):
        mapping = value[field]
        if not isinstance(mapping, dict) or set(mapping) != expected_epoch_keys:
            raise ValueError(
                f"{source}.{field} must contain global epochs 201-{through_epoch}"
            )
        if any(not isinstance(item, str) or not item for item in mapping.values()):
            raise ValueError(f"{source}.{field} contains an empty digest")
    return value


def _validate_summary(summary: dict[str, Any], *, source: str) -> None:
    if int(summary.get("cases", -1)) != VALIDATION_CASES:
        raise ValueError(f"{source} does not contain 108 cases")
    if int(summary.get("subjects", -1)) != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} does not contain 50 subjects")
    per_case = summary.get("per_case")
    if not isinstance(per_case, list) or len(per_case) != VALIDATION_CASES:
        raise ValueError(f"{source} per_case is incomplete")
    case_ids = [str(item.get("case_id")) for item in per_case]
    if len(set(case_ids)) != VALIDATION_CASES:
        raise ValueError(f"{source} contains duplicate case IDs")
    subject_ids = {str(item.get("subject_id")) for item in per_case}
    if len(subject_ids) != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} per_case subject count differs")
    for field in (
        "locator_dsc_mean",
        "locator_subject_dsc_mean",
        "automatic_point_hit_rate",
    ):
        number = float(summary[field])
        if not math.isfinite(number):
            raise ValueError(f"{source} contains non-finite {field}")


def _validate_kd_report(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != KD_VALIDATION_SHA256:
        raise ValueError("KD validation report SHA-256 mismatch")
    report = _read_json(path)
    if report.get("schema") != KD_VALIDATION_SCHEMA or report.get("schema_version") != 1:
        raise ValueError("unexpected KD validation report schema")
    if report.get("outer_selector_access") is not False:
        raise ValueError("KD validation report declares outer-selector access")
    if report.get("confirmation_access") is not False:
        raise ValueError("KD validation report declares confirmation access")
    if report.get("test_access") is not False:
        raise ValueError("KD validation report declares test access")
    if report.get("selected_kd_epoch") != KD_EPOCH:
        raise ValueError("KD validation report no longer selects epoch200")
    summaries = report.get("summaries")
    if not isinstance(summaries, dict) or not isinstance(summaries.get("kd"), dict):
        raise ValueError("KD validation report lacks KD summaries")
    kd200 = summaries["kd"].get(str(KD_EPOCH))
    if not isinstance(kd200, dict):
        raise ValueError("KD validation report lacks KD epoch200")
    _validate_summary(kd200, source="historical KD epoch200")
    for field, expected in (
        ("locator_dsc_mean", KD_SCAN),
        ("locator_subject_dsc_mean", KD_SUBJECT),
        ("automatic_point_hit_rate", KD_POINT),
    ):
        if float(kd200[field]) != expected:
            raise ValueError(f"historical KD epoch200 {field} differs from preregistration")
    return report, digest


def _validate_implementation_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected IAD/KAS implementation receipt schema")
    _require_false_access(receipt, source="IAD/KAS implementation receipt")
    for field in (
        "synthetic_all_gates_passed",
        "gpu_smoke_all_gates_passed",
        "all_gates_passed",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"IAD/KAS implementation receipt has {field} != true")
    return receipt, sha256_file(path)


def _validate_training_report(
    path: Path,
    *,
    expected_arm: str,
) -> tuple[dict[str, Any], str]:
    report = _read_json(path)
    source = f"{expected_arm} TRAIN_DONE"
    if report.get("schema") != TRAINING_REPORT_SCHEMA:
        raise ValueError(f"unexpected {source} schema")
    if report.get("arm") != expected_arm:
        raise ValueError(f"{source} arm mismatch")
    if report.get("status") != "complete":
        raise ValueError(f"{source} is not complete")
    _require_false_access(report, source=source)
    _require_exact_zero(
        report.get("reload_max_abs_difference"),
        field="reload_max_abs_difference",
        source=source,
    )
    _require_exact_zero(
        report.get("stripped_deploy_max_abs_difference"),
        field="stripped_deploy_max_abs_difference",
        source=source,
    )
    _validate_pairing_evidence(report.get("pairing"), source=f"{source}.pairing")
    return report, sha256_file(path)


def _validate_pairing_receipt(
    path: Path,
    *,
    control_training_sha: str,
    kas_training_sha: str,
    control_training: dict[str, Any],
    kas_training: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != PAIRING_RECEIPT_SCHEMA:
        raise ValueError("unexpected IAD/KAS pairing receipt schema")
    _require_false_access(receipt, source="IAD/KAS pairing receipt")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("pairing receipt must contain exactly control and kas arms")
    expected_training_sha = {
        "control": control_training_sha,
        "kas": kas_training_sha,
    }
    training_reports = {
        "control": control_training,
        "kas": kas_training,
    }
    for arm in ARMS:
        item = arms[arm]
        source = f"pairing receipt {arm} arm"
        if not isinstance(item, dict):
            raise ValueError(f"{source} is not an object")
        if item.get("status") != "complete":
            raise ValueError(f"{source} is not complete")
        _require_exact_zero(
            item.get("reload_max_abs_difference"),
            field="reload_max_abs_difference",
            source=source,
        )
        _require_exact_zero(
            item.get("stripped_deploy_max_abs_difference"),
            field="stripped_deploy_max_abs_difference",
            source=source,
        )
        if item.get("training_report_sha256") != expected_training_sha[arm]:
            raise ValueError(f"{source} TRAIN_DONE SHA-256 mismatch")
        pairing = _validate_pairing_evidence(
            item.get("pairing"), source=f"{source}.pairing"
        )
        if pairing != training_reports[arm]["pairing"]:
            raise ValueError(f"{source} pairing differs from bound TRAIN_DONE")
    control_pairing = arms["control"]["pairing"]
    kas_pairing = arms["kas"]["pairing"]
    mismatched = [
        field
        for field in PAIRING_FIELDS
        if control_pairing[field] != kas_pairing[field]
    ]
    if mismatched:
        raise ValueError(f"control/KAS pairing evidence mismatch: {mismatched}")
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("pairing receipt all_gates_passed != true")
    return receipt, sha256_file(path)


def _load_model(
    path: Path,
    *,
    expected_arm: str,
    expected_epoch: int,
    device: torch.device,
) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = f"{expected_arm} checkpoint epoch{expected_epoch}"
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{source} is not a dictionary: {path}")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unexpected {source} schema")
    if checkpoint.get("arm") != expected_arm:
        raise ValueError(f"{source} arm mismatch")
    if checkpoint.get("global_epoch") != expected_epoch:
        raise ValueError(f"{source} global epoch mismatch")
    if checkpoint.get("continuation_epoch") != expected_epoch - 200:
        raise ValueError(f"{source} continuation epoch mismatch")
    if checkpoint.get("resolution") != 128:
        raise ValueError(f"{source} is not true-128")
    _require_false_access(checkpoint, source=source)
    learning_rate = float(checkpoint.get("learning_rate"))
    if not math.isfinite(learning_rate) or learning_rate != 1e-5:
        raise ValueError(f"{source} learning rate must equal frozen 1e-5")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{source} lacks deploy model_state_dict")
    for field in ("architecture", "kas", "loss_diagnostics", "pairing"):
        if not isinstance(checkpoint.get(field), dict):
            raise ValueError(f"{source} lacks {field}")
    _validate_pairing_evidence(
        checkpoint["pairing"],
        source=f"{source}.pairing",
        through_epoch=expected_epoch,
    )
    _require_finite_numbers(checkpoint["kas"], source=f"{source}.kas")
    _require_finite_numbers(
        checkpoint["loss_diagnostics"], source=f"{source}.loss_diagnostics"
    )
    kas_enabled = checkpoint["kas"].get("enabled")
    if kas_enabled is not (expected_arm == "kas"):
        raise ValueError(f"{source} KAS enabled flag mismatch")
    if expected_arm == "kas" and float(checkpoint["kas"].get("weight")) != 0.01:
        raise ValueError(f"{source} KAS weight differs from frozen 0.01")
    _require_exact_zero(
        checkpoint.get("checkpoint_reload_max_abs_difference"),
        field="checkpoint_reload_max_abs_difference",
        source=source,
    )
    _require_exact_zero(
        checkpoint.get("stripped_deploy_max_abs_difference"),
        field="stripped_deploy_max_abs_difference",
        source=source,
    )
    architecture = checkpoint["architecture"]
    expected_architecture = {
        "class": "AutomaticPromptLocator",
        "backbone": "MONAI SegResNet",
        "center_aware": False,
        "init_filters": 16,
        "dropout": 0.1,
        "input_channels": ["T1ce"],
        "outputs": ["WT segmentation logit"],
    }
    for key, expected in expected_architecture.items():
        if architecture.get(key) != expected:
            raise ValueError(f"{source} architecture mismatch for {key}")
    model = AutomaticPromptLocator(
        center_aware=False,
        init_filters=int(architecture["init_filters"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def _validate_cache(cache: Path, manifest: Path) -> BrainCropDataset:
    if sha256_file(manifest) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation manifest SHA-256 mismatch")
    index_path = cache / "index.json"
    if sha256_file(index_path) != VALIDATION_CACHE_INDEX_SHA256:
        raise ValueError("validation cache index SHA-256 mismatch")
    case_ids = read_case_ids(manifest)
    if case_ids is None:
        raise ValueError("validation manifest did not produce case IDs")
    data = BrainCropDataset(cache, expected_split="validation", case_ids=case_ids)
    if len(data) != VALIDATION_CASES:
        raise ValueError("unexpected validation case count")
    subjects = {case.rsplit("-", 1)[0] for case in data.case_ids}
    if len(subjects) != VALIDATION_SUBJECTS:
        raise ValueError("unexpected validation subject count")
    if data.index.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache manifest binding mismatch")
    for filename, field in (
        ("images.f32", "images_sha256"),
        ("labels.u8", "labels_sha256"),
    ):
        if sha256_file(cache / filename) != data.index.get(field):
            raise ValueError(f"validation cache {filename} SHA-256 mismatch")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--kas-dir", type=Path, required=True)
    parser.add_argument("--kd-validation", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--control-training-report", type=Path, required=True)
    parser.add_argument("--kas-training-report", type=Path, required=True)
    parser.add_argument("--pairing-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    started_path = args.output.with_suffix(".started.json")
    if args.output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated IAD/KAS validation")

    kd_report, kd_report_sha = _validate_kd_report(args.kd_validation)
    implementation, implementation_sha = _validate_implementation_receipt(
        args.implementation_receipt
    )
    control_training, control_training_sha = _validate_training_report(
        args.control_training_report, expected_arm="control"
    )
    kas_training, kas_training_sha = _validate_training_report(
        args.kas_training_report, expected_arm="kas"
    )
    pairing_receipt, pairing_receipt_sha = _validate_pairing_receipt(
        args.pairing_receipt,
        control_training_sha=control_training_sha,
        kas_training_sha=kas_training_sha,
        control_training=control_training,
        kas_training=kas_training,
    )
    data = _validate_cache(args.cache, args.validation_manifest)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("IAD/KAS validation requires CUDA")
    torch.cuda.set_device(device)

    arm_dirs = {"control": args.control_dir, "kas": args.kas_dir}
    checkpoint_paths = {
        arm: {
            epoch: arm_dirs[arm] / f"plain_128_{arm}_global_epoch{epoch}.pth"
            for epoch in EPOCHS
        }
        for arm in ARMS
    }
    preloaded: dict[str, dict[int, tuple[AutomaticPromptLocator, dict[str, Any]]]] = {
        arm: {} for arm in ARMS
    }
    checkpoint_sha: dict[str, dict[int, str]] = {arm: {} for arm in ARMS}
    # Verify all formal inputs before recording the first new validation access.
    for arm in ARMS:
        for epoch in EPOCHS:
            path = checkpoint_paths[arm][epoch]
            checkpoint_sha[arm][epoch] = sha256_file(path)
            preloaded[arm][epoch] = _load_model(
                path,
                expected_arm=arm,
                expected_epoch=epoch,
                device=device,
            )

    write_json_atomic(
        started_path,
        {
            "schema": SCHEMA + ".started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "fixed validation108; development selection only",
            "selection": (
                "maximum KAS scan Dice among global epochs 208/216/224/232; "
                "subject Dice exact-tie-break"
            ),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seeds_same_epoch_control": {
                str(epoch): BOOTSTRAP_SEEDS_SAME_EPOCH_CONTROL[epoch]
                for epoch in EPOCHS
            },
            "bootstrap_seeds_historical_kd200": {
                str(epoch): BOOTSTRAP_SEEDS_HISTORICAL_KD200[epoch]
                for epoch in EPOCHS
            },
            "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "validation_cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "kd_validation_sha256": kd_report_sha,
            "implementation_receipt_sha256": implementation_sha,
            "control_training_report_sha256": control_training_sha,
            "kas_training_report_sha256": kas_training_sha,
            "pairing_receipt_sha256": pairing_receipt_sha,
            "checkpoint_sha256": {
                arm: {
                    str(epoch): checkpoint_sha[arm][epoch] for epoch in EPOCHS
                }
                for arm in ARMS
            },
            "threshold": 0.0,
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "validation_access": True,
            "outer_selector_access": False,
            "confirmation_access": False,
            "test_access": False,
        },
    )

    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    summaries: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    checkpoints: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for epoch in EPOCHS:
            key = str(epoch)
            model, checkpoint = preloaded[arm].pop(epoch)
            summaries[arm][key] = validate(model, loader, device)
            _validate_summary(summaries[arm][key], source=f"{arm} epoch{epoch}")
            checkpoints[arm][key] = {
                "path": str(checkpoint_paths[arm][epoch].resolve()),
                "sha256": checkpoint_sha[arm][epoch],
                "arm": checkpoint["arm"],
                "global_epoch": checkpoint["global_epoch"],
                "continuation_epoch": checkpoint["continuation_epoch"],
                "learning_rate": checkpoint["learning_rate"],
                "architecture": checkpoint["architecture"],
                "kas": checkpoint["kas"],
                "loss_diagnostics": checkpoint["loss_diagnostics"],
                "pairing": checkpoint["pairing"],
                "checkpoint_reload_max_abs_difference": checkpoint[
                    "checkpoint_reload_max_abs_difference"
                ],
                "stripped_deploy_max_abs_difference": checkpoint[
                    "stripped_deploy_max_abs_difference"
                ],
            }
            del model
            torch.cuda.empty_cache()

    selected_epoch = max(
        EPOCHS,
        key=lambda epoch: (
            summaries["kas"][str(epoch)]["locator_dsc_mean"],
            summaries["kas"][str(epoch)]["locator_subject_dsc_mean"],
        ),
    )
    selected_key = str(selected_epoch)
    selected = summaries["kas"][selected_key]
    selected_control = summaries["control"][selected_key]
    kd200 = kd_report["summaries"]["kd"][str(KD_EPOCH)]

    versus_control = {
        str(epoch): compare(
            summaries["control"][str(epoch)],
            summaries["kas"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_SAME_EPOCH_CONTROL[epoch],
            samples=BOOTSTRAP_SAMPLES,
        )
        for epoch in EPOCHS
    }
    versus_kd200 = {
        str(epoch): compare(
            kd200,
            summaries["kas"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_HISTORICAL_KD200[epoch],
            samples=BOOTSTRAP_SAMPLES,
        )
        for epoch in EPOCHS
    }
    selected_control_delta = versus_control[selected_key]["ema_minus_raw"]
    selected_kd200_delta = versus_kd200[selected_key]["ema_minus_raw"]

    frozen_pairing = pairing_receipt["arms"]["control"]["pairing"]
    scalar_pairing_fields = PAIRING_FIELDS[:-3]
    epoch_pairing_fields = PAIRING_FIELDS[-3:]
    checkpoint_pairing_match = all(
        all(
            checkpoints[arm][str(epoch)]["pairing"][field]
            == frozen_pairing[field]
            for field in scalar_pairing_fields
        )
        and all(
            checkpoints[arm][str(epoch)]["pairing"][field]
            == {
                str(global_epoch): frozen_pairing[field][str(global_epoch)]
                for global_epoch in TRAINING_GLOBAL_EPOCHS
                if global_epoch <= epoch
            }
            for field in epoch_pairing_fields
        )
        for arm in ARMS
        for epoch in EPOCHS
    )
    checkpoint_reload_exact = all(
        float(checkpoints[arm][str(epoch)]["checkpoint_reload_max_abs_difference"])
        == 0.0
        for arm in ARMS
        for epoch in EPOCHS
    )
    stripped_deploy_exact = all(
        float(checkpoints[arm][str(epoch)]["stripped_deploy_max_abs_difference"])
        == 0.0
        for arm in ARMS
        for epoch in EPOCHS
    )
    prerequisite_gates = {
        "implementation_synthetic_gates_passed": implementation[
            "synthetic_all_gates_passed"
        ]
        is True,
        "implementation_gpu_smoke_gates_passed": implementation[
            "gpu_smoke_all_gates_passed"
        ]
        is True,
        "implementation_all_gates_passed": implementation["all_gates_passed"]
        is True,
        "control_training_complete": control_training["status"] == "complete",
        "kas_training_complete": kas_training["status"] == "complete",
        "pairing_receipt_all_gates_passed": pairing_receipt["all_gates_passed"]
        is True,
        "paired_initial_batch_epoch_order_and_cuda_rng_evidence_match": (
            checkpoint_pairing_match
        ),
        "all_checkpoint_reloads_exact": checkpoint_reload_exact,
        "all_stripped_deploy_logits_exact": stripped_deploy_exact,
    }
    promotion_gates = {
        "selected_beats_same_epoch_control_scan": float(
            selected["locator_dsc_mean"]
        )
        > float(selected_control["locator_dsc_mean"]),
        "selected_beats_same_epoch_control_subject": float(
            selected["locator_subject_dsc_mean"]
        )
        > float(selected_control["locator_subject_dsc_mean"]),
        "selected_beats_historical_voxel_kd200_scan": float(
            selected["locator_dsc_mean"]
        )
        > float(kd200["locator_dsc_mean"]),
        "selected_beats_historical_voxel_kd200_subject": float(
            selected["locator_subject_dsc_mean"]
        )
        > float(kd200["locator_subject_dsc_mean"]),
        "selected_point_within_one_case_of_historical_voxel_kd200": float(
            selected["automatic_point_hit_rate"]
        )
        >= float(kd200["automatic_point_hit_rate"]) - 1.0 / VALIDATION_CASES,
        "paired_scan_ci_lower_at_least_minus_0.003_vs_historical_voxel_kd200": float(
            selected_kd200_delta["scan_95ci"][0]
        )
        >= -0.003,
        "paired_subject_ci_lower_at_least_minus_0.003_vs_historical_voxel_kd200": float(
            selected_kd200_delta["subject_95ci"][0]
        )
        >= -0.003,
        "all_prerequisite_gates_passed": all(prerequisite_gates.values()),
    }
    all_gates_passed = all(prerequisite_gates.values()) and all(
        promotion_gates.values()
    )

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "fixed validation108; development selection only",
        "not_outer_selector": True,
        "not_external_confirmation": True,
        "protocol": {
            "trainer_entrypoint": TRAINER_ENTRYPOINT,
            "method": "teacher-free KAS-on-voxel-KD; not full IAD",
            "resolution": 128,
            "threshold": "raw logit > 0",
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "locator_forwards": 1,
            "epochs": list(EPOCHS),
            "selection": (
                "maximum KAS scan Dice; subject Dice exact-tie-break"
            ),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seeds_same_epoch_control": {
                str(epoch): BOOTSTRAP_SEEDS_SAME_EPOCH_CONTROL[epoch]
                for epoch in EPOCHS
            },
            "bootstrap_seeds_historical_kd200": {
                str(epoch): BOOTSTRAP_SEEDS_HISTORICAL_KD200[epoch]
                for epoch in EPOCHS
            },
            "checkpoint_selection_uses_test": False,
        },
        "data": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "cache": str(args.cache.resolve()),
            "cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "cases": VALIDATION_CASES,
            "subjects": VALIDATION_SUBJECTS,
        },
        "evidence_inputs": {
            "kd_validation": str(args.kd_validation.resolve()),
            "kd_validation_sha256": kd_report_sha,
            "implementation_receipt": str(args.implementation_receipt.resolve()),
            "implementation_receipt_sha256": implementation_sha,
            "control_training_report": str(args.control_training_report.resolve()),
            "control_training_report_sha256": control_training_sha,
            "kas_training_report": str(args.kas_training_report.resolve()),
            "kas_training_report_sha256": kas_training_sha,
            "pairing_receipt": str(args.pairing_receipt.resolve()),
            "pairing_receipt_sha256": pairing_receipt_sha,
        },
        "checkpoints": checkpoints,
        "summaries": summaries,
        "historical_voxel_kd200_summary": kd200,
        "kas_minus_same_epoch_control": versus_control,
        "kas_minus_historical_voxel_kd200": versus_kd200,
        "selection_rule": (
            "maximum KAS scan Dice among global epochs 208/216/224/232; "
            "subject Dice exact-tie-break"
        ),
        "selected_epoch": selected_epoch,
        "selected_checkpoint": checkpoints["kas"][selected_key],
        "selected_summary": selected,
        "selected_same_epoch_control_summary": selected_control,
        "selected_minus_same_epoch_control": selected_control_delta,
        "selected_minus_historical_voxel_kd200": selected_kd200_delta,
        "prerequisite_gates": prerequisite_gates,
        "promotion_gates": promotion_gates,
        "all_gates_passed": all_gates_passed,
        "kas_candidate_authorized_for_test": all_gates_passed,
        "validation_access": True,
        "outer_selector_access": False,
        "confirmation_access": False,
        "test_access": False,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "selected": {
                    "scan": selected["locator_dsc_mean"],
                    "subject": selected["locator_subject_dsc_mean"],
                    "point_hit": selected["automatic_point_hit_rate"],
                },
                "selected_minus_same_epoch_control": selected_control_delta,
                "selected_minus_historical_voxel_kd200": selected_kd200_delta,
                "prerequisite_gates": prerequisite_gates,
                "promotion_gates": promotion_gates,
                "kas_candidate_authorized_for_test": all_gates_passed,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
