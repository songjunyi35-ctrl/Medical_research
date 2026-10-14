#!/usr/bin/env python3
"""Evaluate the preregistered hard-mL1-ACE arm on fixed validation108."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import (  # noqa: E402
    AutomaticPromptLocator,
    automatic_prompt_points,
)
from sammed3d.t1ce_wt_hard_ml1ace import (  # noqa: E402
    HardML1ACEBatchStatistics,
    NUM_BINS,
    aggregate_foreground_calibration,
    hard_ml1ace_from_probabilities,
)
from scripts.evaluate_t1ce_wt_plain_ema_validation import compare  # noqa: E402
from scripts.train_t1ce_wt_capg3d import (  # noqa: E402
    BrainCropDataset,
    dice_per_case,
    read_case_ids,
    sha256_file,
    summarize_cases,
    write_json_atomic,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_validation"
HACE_CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_checkpoint"
CONTROL_CHECKPOINT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_checkpoint"
HACE_TRAINING_REPORT_SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_continuation"
CONTROL_TRAINING_REPORT_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_iad_kas_continuation"
IMPLEMENTATION_RECEIPT_SCHEMA = (
    "aflow.sammed3d.t1ce_wt_hard_ml1ace_implementation_receipt"
)
PAIRING_RECEIPT_SCHEMA = "aflow.sammed3d.t1ce_wt_hard_ml1ace_pairing_receipt"
KD_VALIDATION_SCHEMA = "aflow.sammed3d.t1ce_wt_plain_voxel_kd_validation"
TRAINER_ENTRYPOINT = "scripts/train_t1ce_wt_hard_ml1ace_continuation.py"

ARMS = ("control", "hace")
EPOCHS = (208, 216, 224, 232)
TRAINING_GLOBAL_EPOCHS = tuple(range(201, 233))
VALIDATION_CASES = 108
VALIDATION_SUBJECTS = 50
CALIBRATION_BINS = NUM_BINS
BOOTSTRAP_SAMPLES = 10_000
HACE_WEIGHT = 1.0
MAX_PEAK_GPU_MEMORY_BYTES = 24 * 1024**3

VALIDATION_MANIFEST_SHA256 = (
    "f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb"
)
VALIDATION_CACHE_INDEX_SHA256 = (
    "9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322"
)
KD_VALIDATION_SHA256 = (
    "db7c3bc6b5ff9e1076827d25c9f4de0dea88781bfc5de45da0fb2f79a09bfb57"
)
CONTROL_TRAINING_REPORT_SHA256 = (
    "ce987642413fbf86ccbb45ec263851b8a12cfbec6f21d2325521529b583c6392"
)
CONTROL_CHECKPOINT_SHA256 = {
    208: "3c62166b5ca48b686ef9956cae0b78713505aeefc4db60341b2995fb3d3bb845",
    216: "9140f1ef188fda4158582eff1b6e41df91bd6355ced79965ea3ed51f2624c3b4",
    224: "5c2746dc938064f85757d5547f52bcc301f2e20310d75a09cb9f592968faf60c",
    232: "2d13e2bda6557318775d8c083d1bb3c0ad3b1d49d2a4c98f9c4dca6dd553245e",
}
START_CHECKPOINT_SHA256 = (
    "56f54898ded11a0208e75f8000d57d4d053e80d6dedd760402730ad345c111b7"
)
KD_EPOCH = 200
KD_SCAN = 0.7343843078447713
KD_SUBJECT = 0.7244341433615912
KD_POINT = 0.9351851851851852

BOOTSTRAP_SEEDS_HACE_VS_CONTROL_SEG = {
    208: 20260930,
    216: 20260931,
    224: 20260932,
    232: 20260933,
}
BOOTSTRAP_SEEDS_HACE_VS_KD200_SEG = {
    208: 20260934,
    216: 20260935,
    224: 20260936,
    232: 20260937,
}
BOOTSTRAP_SEEDS_HACE_VS_CONTROL_MACRO_ACE = {
    208: 20260938,
    216: 20260939,
    224: 20260940,
    232: 20260941,
}

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
EXPECTED_ARCHITECTURE = {
    "class": "AutomaticPromptLocator",
    "backbone": "MONAI SegResNet",
    "center_aware": False,
    "init_filters": 16,
    "dropout": 0.1,
    "input_channels": ["T1ce"],
    "outputs": ["WT segmentation logit"],
    "deploy_parameter_count": 4_700_897,
}


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
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{source} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, source=f"{source}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_numbers(item, source=f"{source}[{index}]")
        return
    raise ValueError(f"{source} contains unsupported type {type(value)!r}")


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
    expected_epochs = {
        str(epoch) for epoch in TRAINING_GLOBAL_EPOCHS if epoch <= through_epoch
    }
    for field in (
        "epoch_case_order_sha256",
        "epoch_cuda_rng_before_sha256",
        "epoch_cuda_rng_after_sha256",
    ):
        mapping = value[field]
        if not isinstance(mapping, dict) or set(mapping) != expected_epochs:
            raise ValueError(f"{source}.{field} has the wrong epoch set")
        if any(not isinstance(digest, str) or not digest for digest in mapping.values()):
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
    if len({str(item.get("subject_id")) for item in per_case}) != VALIDATION_SUBJECTS:
        raise ValueError(f"{source} per_case subject count differs")
    for field in (
        "locator_dsc_mean",
        "locator_subject_dsc_mean",
        "automatic_point_hit_rate",
    ):
        if not math.isfinite(float(summary[field])):
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
            raise ValueError(f"historical KD epoch200 {field} differs")
    return report, digest


def _validate_implementation_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise ValueError("unexpected hard-mL1-ACE implementation receipt schema")
    if receipt.get("schema_version") != 1:
        raise ValueError("unexpected implementation receipt schema version")
    _require_false_access(receipt, source="hard-mL1-ACE implementation receipt")
    for field in (
        "synthetic_all_gates_passed",
        "gpu_smoke_all_gates_passed",
        "all_gates_passed",
    ):
        if receipt.get(field) is not True:
            raise ValueError(f"implementation receipt has {field} != true")
    if receipt.get("start_checkpoint_historical_validation_access") is not True:
        raise ValueError("implementation receipt omits historical validation access")
    if receipt.get("start_checkpoint_historical_test_access_after_selection") is not True:
        raise ValueError("implementation receipt omits historical test access")
    frozen = receipt.get("frozen_evidence")
    if not isinstance(frozen, dict):
        raise ValueError("implementation receipt lacks frozen_evidence")
    expected_frozen = {
        "start_checkpoint_sha256": START_CHECKPOINT_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
        "validation_cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
        "kd_validation_report_sha256": KD_VALIDATION_SHA256,
    }
    for field, expected in expected_frozen.items():
        if frozen.get(field) != expected:
            raise ValueError(f"implementation receipt frozen {field} mismatch")
    implementation = receipt.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("implementation receipt lacks implementation bindings")
    if implementation.get("validation_evaluator_path") != str(
        Path(__file__).resolve().relative_to(ROOT)
    ):
        raise ValueError("implementation receipt evaluator path mismatch")
    if implementation.get("validation_evaluator_sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("implementation receipt evaluator SHA-256 mismatch")
    if implementation.get("trainer_path") != TRAINER_ENTRYPOINT:
        raise ValueError("implementation receipt trainer path mismatch")
    if not isinstance(implementation.get("trainer_sha256"), str):
        raise ValueError("implementation receipt lacks trainer SHA-256")
    return receipt, sha256_file(path)


def _validate_training_report(
    path: Path,
    *,
    expected_arm: str,
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    expected_schema = (
        CONTROL_TRAINING_REPORT_SCHEMA
        if expected_arm == "control"
        else HACE_TRAINING_REPORT_SCHEMA
    )
    if expected_arm == "control" and digest != CONTROL_TRAINING_REPORT_SHA256:
        raise ValueError("historical control TRAIN_DONE SHA-256 mismatch")
    report = _read_json(path)
    source = f"{expected_arm} TRAIN_DONE"
    if report.get("schema") != expected_schema or report.get("schema_version") != 1:
        raise ValueError(f"unexpected {source} schema")
    if report.get("arm") != expected_arm or report.get("status") != "complete":
        raise ValueError(f"{source} arm/status mismatch")
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
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {str(e) for e in EPOCHS}:
        raise ValueError(f"{source} does not bind exactly the four frozen checkpoints")
    for epoch in EPOCHS:
        item = checkpoints[str(epoch)]
        if not isinstance(item, dict):
            raise ValueError(f"{source} epoch{epoch} checkpoint entry is not an object")
        if expected_arm == "control" and item.get("sha256") != CONTROL_CHECKPOINT_SHA256[epoch]:
            raise ValueError(f"{source} epoch{epoch} checkpoint SHA-256 mismatch")
        _require_exact_zero(
            item.get("checkpoint_reload_max_abs_difference"),
            field="checkpoint_reload_max_abs_difference",
            source=f"{source} epoch{epoch}",
        )
        _require_exact_zero(
            item.get("stripped_deploy_max_abs_difference"),
            field="stripped_deploy_max_abs_difference",
            source=f"{source} epoch{epoch}",
        )
    peak_memory = int(report.get("peak_gpu_memory_bytes", -1))
    if peak_memory < 0:
        raise ValueError(f"{source} lacks peak_gpu_memory_bytes")
    if expected_arm == "hace":
        if peak_memory > MAX_PEAK_GPU_MEMORY_BYTES:
            raise ValueError("hACE training exceeded the frozen 24-GiB technical stop")
        contract = report.get("hard_ml1ace")
        if not isinstance(contract, dict):
            raise ValueError("hACE TRAIN_DONE lacks hard_ml1ace contract")
        expected_contract = {
            "enabled": True,
            "num_bins": CALIBRATION_BINS,
            "right": False,
            "foreground_only": True,
            "weight": HACE_WEIGHT,
            "precision": "float32",
        }
        if contract != expected_contract:
            raise ValueError("hACE TRAIN_DONE hard_ml1ace contract mismatch")
        if report.get("pairing_all_gates_passed") is not True:
            raise ValueError("hACE TRAIN_DONE pairing_all_gates_passed != true")
        if report.get("pairing_mismatched_fields") != []:
            raise ValueError("hACE TRAIN_DONE records pairing mismatches")
        if report.get("start_checkpoint_sha256") != START_CHECKPOINT_SHA256:
            raise ValueError("hACE TRAIN_DONE start checkpoint SHA-256 mismatch")
        if report.get("start_checkpoint_historical_validation_access") is not True:
            raise ValueError("hACE TRAIN_DONE omits historical validation access")
        if (
            report.get("start_checkpoint_historical_test_access_after_selection")
            is not True
        ):
            raise ValueError("hACE TRAIN_DONE omits historical test access")
        history = report.get("history")
        if not isinstance(history, list) or len(history) != len(TRAINING_GLOBAL_EPOCHS):
            raise ValueError("hACE TRAIN_DONE history is not the frozen 32 epochs")
        if [item.get("global_epoch") for item in history] != list(
            TRAINING_GLOBAL_EPOCHS
        ):
            raise ValueError("hACE TRAIN_DONE history global epochs differ")
        if [item.get("continuation_epoch") for item in history] != list(
            range(1, 33)
        ):
            raise ValueError("hACE TRAIN_DONE continuation epochs differ")
        _require_finite_numbers(report.get("history"), source="hACE TRAIN_DONE.history")
    return report, digest


def _validate_pairing_receipt(
    path: Path,
    *,
    trainings: dict[str, dict[str, Any]],
    training_sha: dict[str, str],
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema") != PAIRING_RECEIPT_SCHEMA or receipt.get("schema_version") != 1:
        raise ValueError("unexpected hard-mL1-ACE pairing receipt schema")
    _require_false_access(receipt, source="hard-mL1-ACE pairing receipt")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("pairing receipt must contain exactly control and hace")
    for arm in ARMS:
        item = arms[arm]
        source = f"pairing receipt {arm} arm"
        if not isinstance(item, dict) or item.get("status") != "complete":
            raise ValueError(f"{source} is not complete")
        if item.get("training_report_sha256") != training_sha[arm]:
            raise ValueError(f"{source} TRAIN_DONE SHA-256 mismatch")
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
        pairing = _validate_pairing_evidence(item.get("pairing"), source=f"{source}.pairing")
        if pairing != trainings[arm]["pairing"]:
            raise ValueError(f"{source} pairing differs from bound TRAIN_DONE")
    mismatched = [
        field
        for field in PAIRING_FIELDS
        if arms["control"]["pairing"][field] != arms["hace"]["pairing"][field]
    ]
    if mismatched:
        raise ValueError(f"control/hACE pairing evidence mismatch: {mismatched}")
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("pairing receipt all_gates_passed != true")
    return receipt, sha256_file(path)


def _validate_architecture(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} lacks architecture")
    if value != EXPECTED_ARCHITECTURE:
        raise ValueError(f"{source} architecture contract mismatch")
    return value


def _validate_start_and_access(checkpoint: dict[str, Any], *, source: str) -> None:
    _require_false_access(checkpoint, source=source)
    if checkpoint.get("start_checkpoint_sha256") != START_CHECKPOINT_SHA256:
        raise ValueError(f"{source} start checkpoint SHA-256 mismatch")
    if checkpoint.get("start_checkpoint_historical_validation_access") is not True:
        raise ValueError(f"{source} must disclose historical validation access")
    if checkpoint.get("start_checkpoint_historical_test_access_after_selection") is not True:
        raise ValueError(f"{source} must disclose historical test access")


def _load_model(
    path: Path,
    *,
    expected_arm: str,
    expected_epoch: int,
    expected_sha: str,
    device: torch.device,
) -> tuple[AutomaticPromptLocator, dict[str, Any]]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"{expected_arm} epoch{expected_epoch} checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = f"{expected_arm} checkpoint epoch{expected_epoch}"
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{source} is not a dictionary")
    expected_schema = (
        CONTROL_CHECKPOINT_SCHEMA if expected_arm == "control" else HACE_CHECKPOINT_SCHEMA
    )
    if checkpoint.get("schema") != expected_schema or checkpoint.get("schema_version") != 1:
        raise ValueError(f"unexpected {source} schema")
    if checkpoint.get("arm") != expected_arm:
        raise ValueError(f"{source} arm mismatch")
    if checkpoint.get("global_epoch") != expected_epoch:
        raise ValueError(f"{source} global epoch mismatch")
    if checkpoint.get("continuation_epoch") != expected_epoch - 200:
        raise ValueError(f"{source} continuation epoch mismatch")
    if checkpoint.get("resolution") != 128:
        raise ValueError(f"{source} is not true-128")
    if float(checkpoint.get("learning_rate")) != 1e-5:
        raise ValueError(f"{source} learning rate differs from frozen 1e-5")
    _validate_start_and_access(checkpoint, source=source)
    architecture = _validate_architecture(checkpoint.get("architecture"), source=source)
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{source} lacks model_state_dict")
    _validate_pairing_evidence(
        checkpoint.get("pairing"),
        source=f"{source}.pairing",
        through_epoch=expected_epoch,
    )
    _require_finite_numbers(
        checkpoint.get("loss_diagnostics"), source=f"{source}.loss_diagnostics"
    )
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
    if expected_arm == "control":
        kas = checkpoint.get("kas")
        expected_control_contract = {
            "components": [],
            "deploy": "plain student final raw logit only",
            "enabled": False,
            "gamma": 0.0,
            "method": "matched teacher-free DiceCE control",
            "projector": None,
            "teacher_forwards": 0,
            "weight": 0.0,
        }
        if kas != expected_control_contract:
            raise ValueError(f"{source} is not the frozen teacher-free control")
    else:
        contract = checkpoint.get("hard_ml1ace")
        if not isinstance(contract, dict):
            raise ValueError(f"{source} lacks hard_ml1ace contract")
        expected_contract = {
            "enabled": True,
            "weight": HACE_WEIGHT,
            "num_bins": CALIBRATION_BINS,
            "right": False,
            "foreground_only": True,
            "precision": "float32",
        }
        if contract != expected_contract:
            raise ValueError(f"{source} hard_ml1ace contract mismatch")
        _require_finite_numbers(contract, source=f"{source}.hard_ml1ace")
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
    if len({case.rsplit("-", 1)[0] for case in data.case_ids}) != VALIDATION_SUBJECTS:
        raise ValueError("unexpected validation subject count")
    if data.index.get("manifest_sha256") != VALIDATION_MANIFEST_SHA256:
        raise ValueError("validation cache manifest binding mismatch")
    for filename, field in (("images.f32", "images_sha256"), ("labels.u8", "labels_sha256")):
        if sha256_file(cache / filename) != data.index.get(field):
            raise ValueError(f"validation cache {filename} SHA-256 mismatch")
    return data


@torch.inference_mode()
def _evaluate_model(
    model: AutomaticPromptLocator,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    segmentation_cases: list[dict[str, Any]] = []
    calibration_cases: list[dict[str, Any]] = []
    calibration_batches: list[HardML1ACEBatchStatistics] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(image)
        logits = output.segmentation_logits.float()
        dsc = dice_per_case(logits, label)
        points = automatic_prompt_points(output).long()
        probabilities = torch.sigmoid(logits)
        batch_calibration = hard_ml1ace_from_probabilities(probabilities, label)
        calibration_batches.append(batch_calibration)
        for index, case_id in enumerate(batch["case_id"]):
            z, y, x = (int(value) for value in points[index, 0].tolist())
            case_id = str(case_id)
            subject_id = str(batch["subject_id"][index])
            segmentation_cases.append(
                {
                    "case_id": case_id,
                    "subject_id": subject_id,
                    "locator_dsc": float(dsc[index]),
                    "automatic_point": [z, y, x],
                    "automatic_point_hits_gt": bool(label[index, 0, z, y, x] > 0.5),
                }
            )
            counts = batch_calibration.counts[index].detach().cpu()
            occupied = batch_calibration.occupied[index].detach().cpu()
            mean_probabilities = (
                batch_calibration.mean_probabilities[index].detach().cpu()
            )
            mean_targets = batch_calibration.mean_targets[index].detach().cpu()
            calibration = {
                "case_id": case_id,
                "subject_id": subject_id,
                "ace": float(batch_calibration.per_case_ace[index]),
                "ece": float(batch_calibration.per_case_ece[index]),
                "mce": float(batch_calibration.per_case_mce[index]),
                "target_nonempty": bool(batch_calibration.target_nonempty[index]),
                "counts": [int(value) for value in counts],
                "probability_sums": [
                    float(value)
                    for value in batch_calibration.probability_sums[index]
                    .detach()
                    .cpu()
                ],
                "target_sums": [
                    float(value)
                    for value in batch_calibration.target_sums[index].detach().cpu()
                ],
                "mean_probability": [
                    float(mean_probabilities[bin_index])
                    if bool(occupied[bin_index])
                    else None
                    for bin_index in range(CALIBRATION_BINS)
                ],
                "mean_target": [
                    float(mean_targets[bin_index])
                    if bool(occupied[bin_index])
                    else None
                    for bin_index in range(CALIBRATION_BINS)
                ],
            }
            calibration_cases.append(calibration)
    segmentation = summarize_cases(segmentation_cases)
    aggregate = aggregate_foreground_calibration(calibration_batches)
    occupied = aggregate.occupied.detach().cpu()
    calibration = {
        "cases": aggregate.case_count,
        "bins": CALIBRATION_BINS,
        "binning": "float32 linspace(0,1+eps,21); bucketize right=false",
        "macro_ace": float(aggregate.macro_ace),
        "macro_ece": float(aggregate.macro_ece),
        "macro_mce": float(aggregate.macro_mce),
        "official_micro_ace": float(aggregate.official_micro_ace),
        "pooled_voxel_ace": float(aggregate.pooled_voxel_ace),
        "per_case": calibration_cases,
        "official_micro_bins": {
            "counts": [int(value) for value in aggregate.bin_counts.detach().cpu()],
            "occupied_case_counts": [
                int(value) for value in aggregate.occupied_case_counts.detach().cpu()
            ],
            "mean_probability_across_cases": [
                float(aggregate.official_micro_mean_probabilities[index])
                if bool(occupied[index])
                else None
                for index in range(CALIBRATION_BINS)
            ],
            "count_weighted_mean_target": [
                float(aggregate.official_micro_mean_targets[index])
                if bool(occupied[index])
                else None
                for index in range(CALIBRATION_BINS)
            ],
        },
        "pooled_voxel_bins": {
            "counts": [int(value) for value in aggregate.bin_counts.detach().cpu()],
            "mean_probability": [
                float(aggregate.pooled_mean_probabilities[index])
                if bool(occupied[index])
                else None
                for index in range(CALIBRATION_BINS)
            ],
            "mean_target": [
                float(aggregate.pooled_mean_targets[index])
                if bool(occupied[index])
                else None
                for index in range(CALIBRATION_BINS)
            ],
        },
    }
    if [item["case_id"] for item in segmentation_cases] != [
        item["case_id"] for item in calibration_cases
    ]:
        raise ValueError("segmentation/calibration case order differs")
    return segmentation, calibration


def _compare_macro_ace(
    control: dict[str, Any],
    hace: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    control_cases = {str(item["case_id"]): float(item["ace"]) for item in control["per_case"]}
    hace_cases = {str(item["case_id"]): float(item["ace"]) for item in hace["per_case"]}
    if control_cases.keys() != hace_cases.keys():
        raise ValueError("control/hACE calibration case sets differ")
    delta = np.asarray(
        [hace_cases[key] - control_cases[key] for key in sorted(control_cases)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))
    interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    return {
        "control_macro_ace": float(control["macro_ace"]),
        "hace_macro_ace": float(hace["macro_ace"]),
        "hace_minus_control": float(delta.mean()),
        "hace_minus_control_95ci": [float(value) for value in interval],
        "cases_improved": int(np.sum(delta < 0.0)),
        "cases_tied": int(np.sum(delta == 0.0)),
        "cases_worse": int(np.sum(delta > 0.0)),
        "bootstrap_seed": seed,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--hace-dir", type=Path, required=True)
    parser.add_argument("--control-training-report", type=Path, required=True)
    parser.add_argument("--hace-training-report", type=Path, required=True)
    parser.add_argument("--pairing-receipt", type=Path, required=True)
    parser.add_argument("--kd-validation", type=Path, required=True)
    parser.add_argument("--implementation-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    started_path = args.output.with_suffix(".started.json")
    if args.output.exists() or started_path.exists():
        raise FileExistsError("refusing repeated hard-mL1-ACE validation")
    if args.batch_size != 8:
        raise ValueError("fixed validation batch size must remain 8")

    kd_report, kd_sha = _validate_kd_report(args.kd_validation)
    implementation, implementation_sha = _validate_implementation_receipt(
        args.implementation_receipt
    )
    trainings: dict[str, dict[str, Any]] = {}
    training_sha: dict[str, str] = {}
    for arm, path in (
        ("control", args.control_training_report),
        ("hace", args.hace_training_report),
    ):
        trainings[arm], training_sha[arm] = _validate_training_report(
            path, expected_arm=arm
        )
    if (
        implementation["implementation"]["trainer_sha256"]
        != trainings["hace"].get("runner_sha256")
    ):
        raise ValueError("implementation receipt trainer SHA differs from hACE TRAIN_DONE")
    pairing_receipt, pairing_sha = _validate_pairing_receipt(
        args.pairing_receipt,
        trainings=trainings,
        training_sha=training_sha,
    )
    data = _validate_cache(args.cache, args.validation_manifest)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("hard-mL1-ACE validation requires CUDA")
    torch.cuda.set_device(device)

    arm_dirs = {"control": args.control_dir, "hace": args.hace_dir}
    checkpoint_paths = {
        "control": {
            epoch: arm_dirs["control"] / f"plain_128_control_global_epoch{epoch}.pth"
            for epoch in EPOCHS
        },
        "hace": {
            epoch: arm_dirs["hace"] / f"plain_128_hace_global_epoch{epoch}.pth"
            for epoch in EPOCHS
        },
    }
    checkpoint_sha = {
        arm: {
            epoch: str(trainings[arm]["checkpoints"][str(epoch)]["sha256"])
            for epoch in EPOCHS
        }
        for arm in ARMS
    }
    preloaded: dict[str, dict[int, tuple[AutomaticPromptLocator, dict[str, Any]]]] = {
        arm: {} for arm in ARMS
    }
    # Verify every formal input before recording the first new validation access.
    for arm in ARMS:
        for epoch in EPOCHS:
            if arm == "control" and checkpoint_sha[arm][epoch] != CONTROL_CHECKPOINT_SHA256[epoch]:
                raise ValueError(f"control epoch{epoch} TRAIN_DONE binding changed")
            preloaded[arm][epoch] = _load_model(
                checkpoint_paths[arm][epoch],
                expected_arm=arm,
                expected_epoch=epoch,
                expected_sha=checkpoint_sha[arm][epoch],
                device=device,
            )

    frozen_pairing = pairing_receipt["arms"]["control"]["pairing"]
    scalar_pairing_fields = PAIRING_FIELDS[:-3]
    epoch_pairing_fields = PAIRING_FIELDS[-3:]
    for arm in ARMS:
        for epoch in EPOCHS:
            pairing = preloaded[arm][epoch][1]["pairing"]
            if any(pairing[field] != frozen_pairing[field] for field in scalar_pairing_fields):
                raise ValueError(f"{arm} epoch{epoch} scalar pairing mismatch")
            expected_epoch_pairing = {
                field: {
                    str(global_epoch): frozen_pairing[field][str(global_epoch)]
                    for global_epoch in TRAINING_GLOBAL_EPOCHS
                    if global_epoch <= epoch
                }
                for field in epoch_pairing_fields
            }
            if any(pairing[field] != expected_epoch_pairing[field] for field in epoch_pairing_fields):
                raise ValueError(f"{arm} epoch{epoch} per-epoch pairing mismatch")

    write_json_atomic(
        started_path,
        {
            "schema": SCHEMA + ".started",
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "fixed validation108; development selection only",
            "selection": (
                "maximum hACE scan Dice among global epochs 208/216/224/232; "
                "subject Dice exact-tie-break; calibration never selects checkpoint"
            ),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seeds_hace_vs_control_seg": {
                str(epoch): BOOTSTRAP_SEEDS_HACE_VS_CONTROL_SEG[epoch] for epoch in EPOCHS
            },
            "bootstrap_seeds_hace_vs_kd200_seg": {
                str(epoch): BOOTSTRAP_SEEDS_HACE_VS_KD200_SEG[epoch] for epoch in EPOCHS
            },
            "bootstrap_seeds_hace_vs_control_macro_ace": {
                str(epoch): BOOTSTRAP_SEEDS_HACE_VS_CONTROL_MACRO_ACE[epoch]
                for epoch in EPOCHS
            },
            "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
            "validation_cache_index_sha256": VALIDATION_CACHE_INDEX_SHA256,
            "kd_validation_sha256": kd_sha,
            "implementation_receipt_sha256": implementation_sha,
            "control_training_report_sha256": training_sha["control"],
            "hace_training_report_sha256": training_sha["hace"],
            "pairing_receipt_sha256": pairing_sha,
            "checkpoint_sha256": {
                arm: {str(epoch): checkpoint_sha[arm][epoch] for epoch in EPOCHS}
                for arm in ARMS
            },
            "threshold": 0.0,
            "calibration_bins": CALIBRATION_BINS,
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
    calibration: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    checkpoints: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for epoch in EPOCHS:
            key = str(epoch)
            model, checkpoint = preloaded[arm].pop(epoch)
            summaries[arm][key], calibration[arm][key] = _evaluate_model(
                model, loader, device
            )
            _validate_summary(summaries[arm][key], source=f"{arm} epoch{epoch}")
            checkpoints[arm][key] = {
                "path": str(checkpoint_paths[arm][epoch].resolve()),
                "sha256": checkpoint_sha[arm][epoch],
                "schema": checkpoint["schema"],
                "arm": checkpoint["arm"],
                "global_epoch": checkpoint["global_epoch"],
                "continuation_epoch": checkpoint["continuation_epoch"],
                "learning_rate": checkpoint["learning_rate"],
                "architecture": checkpoint["architecture"],
                "loss_contract": checkpoint[
                    "kas" if arm == "control" else "hard_ml1ace"
                ],
                "loss_diagnostics": checkpoint["loss_diagnostics"],
                "pairing": checkpoint["pairing"],
                "start_checkpoint_sha256": checkpoint["start_checkpoint_sha256"],
                "start_checkpoint_historical_validation_access": checkpoint[
                    "start_checkpoint_historical_validation_access"
                ],
                "start_checkpoint_historical_test_access_after_selection": checkpoint[
                    "start_checkpoint_historical_test_access_after_selection"
                ],
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
            summaries["hace"][str(epoch)]["locator_dsc_mean"],
            summaries["hace"][str(epoch)]["locator_subject_dsc_mean"],
        ),
    )
    selected_key = str(selected_epoch)
    selected = summaries["hace"][selected_key]
    selected_control = summaries["control"][selected_key]
    selected_calibration = calibration["hace"][selected_key]
    selected_control_calibration = calibration["control"][selected_key]
    kd200 = kd_report["summaries"]["kd"][str(KD_EPOCH)]

    versus_control_seg = {
        str(epoch): compare(
            summaries["control"][str(epoch)],
            summaries["hace"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_HACE_VS_CONTROL_SEG[epoch],
            samples=BOOTSTRAP_SAMPLES,
        )
        for epoch in EPOCHS
    }
    versus_kd200_seg = {
        str(epoch): compare(
            kd200,
            summaries["hace"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_HACE_VS_KD200_SEG[epoch],
            samples=BOOTSTRAP_SAMPLES,
        )
        for epoch in EPOCHS
    }
    versus_control_macro_ace = {
        str(epoch): _compare_macro_ace(
            calibration["control"][str(epoch)],
            calibration["hace"][str(epoch)],
            seed=BOOTSTRAP_SEEDS_HACE_VS_CONTROL_MACRO_ACE[epoch],
        )
        for epoch in EPOCHS
    }
    selected_control_seg_delta = versus_control_seg[selected_key]["ema_minus_raw"]
    selected_kd200_seg_delta = versus_kd200_seg[selected_key]["ema_minus_raw"]
    selected_macro_ace_delta = versus_control_macro_ace[selected_key]

    checkpoint_reload_exact = all(
        float(checkpoints[arm][str(epoch)]["checkpoint_reload_max_abs_difference"]) == 0.0
        for arm in ARMS
        for epoch in EPOCHS
    )
    stripped_deploy_exact = all(
        float(checkpoints[arm][str(epoch)]["stripped_deploy_max_abs_difference"]) == 0.0
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
        "implementation_all_gates_passed": implementation["all_gates_passed"] is True,
        "control_training_complete": trainings["control"]["status"] == "complete",
        "hace_training_complete": trainings["hace"]["status"] == "complete",
        "hace_peak_gpu_memory_at_most_24gib": int(
            trainings["hace"]["peak_gpu_memory_bytes"]
        )
        <= MAX_PEAK_GPU_MEMORY_BYTES,
        "pairing_receipt_all_gates_passed": pairing_receipt["all_gates_passed"] is True,
        "all_checkpoint_reloads_exact": checkpoint_reload_exact,
        "all_stripped_deploy_logits_exact": stripped_deploy_exact,
        "all_checkpoint_start_and_access_disclosures_valid": True,
    }
    promotion_gates = {
        "selected_beats_same_epoch_control_scan": float(selected["locator_dsc_mean"])
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
        "selected_point_at_least_100_of_108": float(selected["automatic_point_hit_rate"])
        >= 100.0 / VALIDATION_CASES,
        "paired_scan_ci_lower_at_least_minus_0.003_vs_kd200": float(
            selected_kd200_seg_delta["scan_95ci"][0]
        )
        >= -0.003,
        "paired_subject_ci_lower_at_least_minus_0.003_vs_kd200": float(
            selected_kd200_seg_delta["subject_95ci"][0]
        )
        >= -0.003,
        "selected_macro_ace_improves_by_at_least_0.005_vs_control": float(
            selected_control_calibration["macro_ace"]
        )
        - float(selected_calibration["macro_ace"])
        >= 0.005,
        "paired_macro_ace_delta_ci_upper_below_zero_vs_control": float(
            selected_macro_ace_delta["hace_minus_control_95ci"][1]
        )
        < 0.0,
        "selected_official_micro_ace_below_control": float(
            selected_calibration["official_micro_ace"]
        )
        < float(selected_control_calibration["official_micro_ace"]),
        "all_prerequisite_gates_passed": all(prerequisite_gates.values()),
    }
    all_gates_passed = all(prerequisite_gates.values()) and all(
        promotion_gates.values()
    )

    report = {
        "schema": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "raw fixed-validation report",
        "evidence_scope": "fixed validation108; development selection only",
        "not_outer_selector": True,
        "not_external_confirmation": True,
        "protocol": {
            "trainer_entrypoint": TRAINER_ENTRYPOINT,
            "method": "foreground-only 20-bin hard mL1-ACE increment on KD200",
            "resolution": 128,
            "threshold": "raw logit > 0",
            "tta": False,
            "sam": False,
            "postprocessing": False,
            "locator_forwards": 1,
            "calibration_probability": "float32 sigmoid raw foreground logit",
            "calibration_bins": CALIBRATION_BINS,
            "calibration_checkpoint_selection": False,
            "epochs": list(EPOCHS),
            "selection": "maximum hACE scan Dice; subject Dice exact-tie-break",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seeds_hace_vs_control_seg": {
                str(epoch): BOOTSTRAP_SEEDS_HACE_VS_CONTROL_SEG[epoch] for epoch in EPOCHS
            },
            "bootstrap_seeds_hace_vs_kd200_seg": {
                str(epoch): BOOTSTRAP_SEEDS_HACE_VS_KD200_SEG[epoch] for epoch in EPOCHS
            },
            "bootstrap_seeds_hace_vs_control_macro_ace": {
                str(epoch): BOOTSTRAP_SEEDS_HACE_VS_CONTROL_MACRO_ACE[epoch]
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
            "kd_validation_sha256": kd_sha,
            "implementation_receipt": str(args.implementation_receipt.resolve()),
            "implementation_receipt_sha256": implementation_sha,
            "control_training_report": str(args.control_training_report.resolve()),
            "control_training_report_sha256": training_sha["control"],
            "hace_training_report": str(args.hace_training_report.resolve()),
            "hace_training_report_sha256": training_sha["hace"],
            "pairing_receipt": str(args.pairing_receipt.resolve()),
            "pairing_receipt_sha256": pairing_sha,
        },
        "checkpoints": checkpoints,
        "summaries": summaries,
        "calibration": calibration,
        "historical_voxel_kd200_summary": kd200,
        "hace_minus_same_epoch_control_segmentation": versus_control_seg,
        "hace_minus_historical_voxel_kd200_segmentation": versus_kd200_seg,
        "hace_minus_same_epoch_control_macro_ace": versus_control_macro_ace,
        "selection_rule": (
            "maximum hACE scan Dice among global epochs 208/216/224/232; "
            "subject Dice exact-tie-break; calibration never selects checkpoint"
        ),
        "selected_epoch": selected_epoch,
        "selected_checkpoint": checkpoints["hace"][selected_key],
        "selected_summary": selected,
        "selected_calibration": selected_calibration,
        "selected_same_epoch_control_summary": selected_control,
        "selected_same_epoch_control_calibration": selected_control_calibration,
        "selected_minus_same_epoch_control_segmentation": selected_control_seg_delta,
        "selected_minus_historical_voxel_kd200_segmentation": selected_kd200_seg_delta,
        "selected_minus_same_epoch_control_macro_ace": selected_macro_ace_delta,
        "prerequisite_gates": prerequisite_gates,
        "promotion_gates": promotion_gates,
        "all_gates_passed": all_gates_passed,
        "hace_candidate_authorized_for_test": all_gates_passed,
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
                    "macro_ace": selected_calibration["macro_ace"],
                    "official_micro_ace": selected_calibration["official_micro_ace"],
                    "pooled_voxel_ace": selected_calibration["pooled_voxel_ace"],
                },
                "prerequisite_gates": prerequisite_gates,
                "promotion_gates": promotion_gates,
                "hace_candidate_authorized_for_test": all_gates_passed,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
