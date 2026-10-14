#!/usr/bin/env python3
"""One-time fresh-inner comparison of identity versus fixed Flip8 TTA."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator  # noqa: E402
from scripts.train_t1ce_wt_capg3d import BrainCropDataset, read_case_ids, sha256_file, summarize_cases, write_json_atomic  # noqa: E402
import scripts.train_t1ce_wt_flip8_tta_replication as train  # noqa: E402


SCHEMA = "aflow.sammed3d.t1ce_wt_flip8_tta_replication_inner"
BOOTSTRAP_SAMPLES = 10_000
LEADER_MARGIN = 0.000243638
REFLECTION_AXES: tuple[tuple[int, ...], ...] = (
    (), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1)
)
EXPECTED_TRAIN_ACCESS = {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}


def load_split(path: Path) -> dict[str, Any]:
    split = json.loads(path.read_text())
    if split.get("schema") != "aflow.brats2024.t1ce_wt_flip8_tta_replication_splits":
        raise ValueError("split schema mismatch")
    if split.get("documents") != {"idea_sha256": train.IDEA_SHA256, "prereg_sha256": train.PREREG_SHA256}:
        raise ValueError("split document binding mismatch")
    if split.get("access") != {"train_only_inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False}:
        raise ValueError("split access boundary changed")
    return split


def load_training_report(path: Path, replicate: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != train.SCHEMA or value.get("status") != "complete":
        raise ValueError(f"invalid training report: {path}")
    if value.get("replicate") != replicate or value.get("arm") != "base" or value.get("fixed_stop_epoch") != train.FORMAL_EPOCHS:
        raise ValueError(f"training identity mismatch: {path}")
    if value.get("all_gates_passed") is not True or len(value.get("history", [])) != train.FORMAL_EPOCHS:
        raise ValueError(f"training gates/epochs failed: {path}")
    if value.get("runner_sha256") != sha256_file(Path(train.__file__).resolve()):
        raise ValueError("training runner SHA mismatch")
    if value.get("access") != EXPECTED_TRAIN_ACCESS or value.get("data_roles", {}).get("inner_dataset_constructed") is not False:
        raise ValueError("training crossed evaluation boundary")
    if any(row.get("inner_metrics") is not None for row in value["history"]):
        raise ValueError("training history contains inner metrics")
    checkpoint_path = Path(value["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != value["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != train.CHECKPOINT_SCHEMA or checkpoint.get("replicate") != replicate or checkpoint.get("arm") != "base" or checkpoint.get("epoch") != train.FORMAL_EPOCHS:
        raise ValueError("checkpoint identity mismatch")
    if checkpoint.get("model_state_sha256") != train.base.state_dict_sha256(checkpoint["model_state_dict"]):
        raise ValueError("checkpoint model-state SHA mismatch")
    if checkpoint.get("access") != EXPECTED_TRAIN_ACCESS:
        raise ValueError("checkpoint crossed evaluation boundary")
    return value


def preflight_training_artifacts(work: Path) -> dict[str, dict[str, Any]]:
    reports = {
        replicate: load_training_report(work / f"formal_{replicate}_base/TRAIN_DONE.json", replicate)
        for replicate in ("a", "b")
    }
    if reports["a"]["initial_state_sha256"] == reports["b"]["initial_state_sha256"]:
        raise ValueError("replications unexpectedly share initial state")
    return reports


def load_model(report: dict[str, Any], device: torch.device) -> AutomaticPromptLocator:
    checkpoint = torch.load(report["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model = AutomaticPromptLocator(center_aware=False, init_filters=16, dropout=0.1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def flip8_logits(model: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total: torch.Tensor | None = None
    identity: torch.Tensor | None = None
    for axes in REFLECTION_AXES:
        transformed = torch.flip(image, dims=axes) if axes else image
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(transformed).segmentation_logits
        restored = torch.flip(logits, dims=axes) if axes else logits
        restored = restored.float()
        if identity is None:
            identity = restored.clone()
            total = torch.zeros_like(restored, dtype=torch.float32)
        assert total is not None
        total.add_(restored)
    assert identity is not None and total is not None
    return identity, total / float(len(REFLECTION_AXES))


def _case_row(logits: torch.Tensor, label: torch.Tensor, case_id: str, subject_id: str) -> dict[str, Any]:
    prediction = logits[0, 0] > 0
    target = label[0, 0] > 0.5
    intersection = int((prediction & target).sum().item())
    denominator = int(prediction.sum().item() + target.sum().item())
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    flat_index = int(torch.argmax(logits[0, 0]).item())
    width = int(logits.shape[-1])
    height = int(logits.shape[-2])
    z = flat_index // (height * width)
    remainder = flat_index % (height * width)
    y = remainder // width
    x = remainder % width
    return {
        "case_id": case_id,
        "subject_id": subject_id,
        "locator_dsc": float(dice),
        "automatic_point": [z, y, x],
        "automatic_point_hits_gt": bool(target[z, y, x].item()),
    }


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    identity_rows: list[dict[str, Any]] = []
    flip8_rows: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        label = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        identity_logits, averaged_logits = flip8_logits(model, image)
        case_id = str(batch["case_id"][0])
        subject_id = str(batch["subject_id"][0])
        identity_rows.append(_case_row(identity_logits, label, case_id, subject_id))
        flip8_rows.append(_case_row(averaged_logits, label, case_id, subject_id))
    return {"identity": summarize_cases(identity_rows), "flip8": summarize_cases(flip8_rows)}


def _bootstrap(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    return [float(value) for value in np.quantile(array[draws].mean(axis=1), (0.025, 0.975))]


def compare(identity: dict[str, Any], flip8: dict[str, Any], seed: int) -> dict[str, Any]:
    left = {row["case_id"]: row for row in identity["per_case"]}
    right = {row["case_id"]: row for row in flip8["per_case"]}
    if list(left) != list(right):
        raise ValueError("identity/Flip8 case order mismatch")
    scan_delta = [float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]) for key in left]
    by_subject: dict[str, list[float]] = defaultdict(list)
    for key in left:
        if left[key]["subject_id"] != right[key]["subject_id"]:
            raise ValueError("identity/Flip8 subject mismatch")
        by_subject[left[key]["subject_id"]].append(float(right[key]["locator_dsc"]) - float(left[key]["locator_dsc"]))
    subject_delta = [statistics.fmean(values) for values in by_subject.values()]
    point_delta = sum(bool(row["automatic_point_hits_gt"]) for row in flip8["per_case"]) - sum(bool(row["automatic_point_hits_gt"]) for row in identity["per_case"])
    return {
        "flip8_minus_identity": {"scan_dice": statistics.fmean(scan_delta), "subject_dice": statistics.fmean(subject_delta), "point_hits": point_delta},
        "scan_dice_95ci": _bootstrap(scan_delta, seed),
        "subject_dice_95ci": _bootstrap(subject_delta, seed + 100),
        "scan_improved": sum(value > 0 for value in scan_delta),
        "scan_worse": sum(value < 0 for value in scan_delta),
        "scan_tied": sum(value == 0 for value in scan_delta),
        "subjects_improved": sum(value > 0 for value in subject_delta),
        "subjects_worse": sum(value < 0 for value in subject_delta),
        "subjects_tied": sum(value == 0 for value in subject_delta),
    }


def replication_gates(comparison: dict[str, Any]) -> dict[str, bool]:
    delta = comparison["flip8_minus_identity"]
    gates = {
        "scan_strictly_positive": float(delta["scan_dice"]) > 0,
        "subject_strictly_positive": float(delta["subject_dice"]) > 0,
        "point_loses_at_most_one": int(delta["point_hits"]) >= -1,
        "scan_ci_lower_at_least_minus_0p003": float(comparison["scan_dice_95ci"][0]) >= -0.003,
        "subject_ci_lower_at_least_minus_0p003": float(comparison["subject_dice_95ci"][0]) >= -0.003,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preaccess", type=Path, required=True)
    parser.add_argument("--expected-preaccess-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    started_output = output.with_suffix(output.suffix + ".started.json")
    if output.exists() or started_output.exists():
        raise FileExistsError("refusing repeated inner evaluation")
    if sha256_file(args.preaccess) != args.expected_preaccess_sha256:
        raise ValueError("preaccess SHA mismatch")
    receipt = json.loads(args.preaccess.read_text())
    if receipt.get("status") != "frozen_after_two_trainings_before_inner_access" or receipt.get("access", {}).get("inner_metrics") is not False:
        raise ValueError("invalid preaccess receipt")
    if receipt.get("evaluator_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("preaccess evaluator SHA mismatch")
    split = load_split(args.split_manifest.resolve())
    if sha256_file(args.split_manifest) != receipt["split_manifest_sha256"]:
        raise ValueError("split changed after preaccess")
    if sha256_file(args.cache / "index.json") != train.CONTROL_CACHE_INDEX_SHA256:
        raise ValueError("train cache index mismatch")
    reports = preflight_training_artifacts(args.work_dir.resolve())
    for replicate, report in reports.items():
        frozen = receipt["training_artifacts"][replicate]
        report_path = args.work_dir.resolve() / f"formal_{replicate}_base/TRAIN_DONE.json"
        if sha256_file(report_path) != frozen["report_sha256"] or report["checkpoint"]["sha256"] != frozen["checkpoint_sha256"]:
            raise ValueError(f"training artifact changed after preaccess: {replicate}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("inner evaluation requires CUDA")
    torch.cuda.set_device(device)
    write_json_atomic(started_output, {
        "schema": SCHEMA + ".started", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preaccess_sha256": args.expected_preaccess_sha256,
        "inner_access": True, "fixed_validation108": False, "outer": False,
        "confirmation": False, "test270": False, "ucsd_external75": False,
    })
    results: dict[str, Any] = {}
    scan_deltas: list[float] = []
    subject_deltas: list[float] = []
    all_replication_gates = True
    for replicate in ("a", "b"):
        inner = split["replications"][replicate]["inner"]
        inner_path = Path(inner["path"])
        if sha256_file(inner_path) != inner["sha256"]:
            raise ValueError(f"replicate {replicate} inner manifest changed")
        case_ids = read_case_ids(inner_path)
        data = BrainCropDataset(args.cache, expected_split="train", case_ids=case_ids)
        if len(data) != 108 or len({train.base.subject_id(case_id) for case_id in data.case_ids}) != int(inner["subjects"]):
            raise ValueError("inner case/subject count mismatch")
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
        model = load_model(reports[replicate], device)
        arms = evaluate_model(model, loader, device)
        del model, loader, data
        torch.cuda.empty_cache()
        comparison = compare(arms["identity"], arms["flip8"], int(split["replications"][replicate]["bootstrap_seed"]))
        gates = replication_gates(comparison)
        all_replication_gates = all_replication_gates and gates["all_passed"]
        scan_deltas.append(float(comparison["flip8_minus_identity"]["scan_dice"]))
        subject_deltas.append(float(comparison["flip8_minus_identity"]["subject_dice"]))
        results[replicate] = {"cohort": inner, "arms": arms, "comparison": comparison, "gates": gates}
    cross = {"mean_scan_delta": statistics.fmean(scan_deltas), "mean_subject_delta": statistics.fmean(subject_deltas)}
    cross_gates = {"mean_scan_delta_exceeds_current_leader_margin": cross["mean_scan_delta"] > LEADER_MARGIN, "mean_subject_delta_positive": cross["mean_subject_delta"] > 0}
    test_authorized = all_replication_gates and all(cross_gates.values())
    report = {
        "schema": SCHEMA, "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "fixed_epoch": train.FORMAL_EPOCHS,
        "tta": {"axes": [list(axes) for axes in REFLECTION_AXES], "aggregation": "ordered float32 mean of inverse-reflected raw logits", "forwards_per_case": 8, "threshold": 0.0},
        "replications": results,
        "cross_replication": {**cross, "leader_margin": LEADER_MARGIN, "gates": cross_gates},
        "test270_authorized": test_authorized,
        "result_must_not_trigger_view_weight_threshold_or_checkpoint_tuning": True,
        "access": {"fit": True, "inner_metrics": True, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    write_json_atomic(output, report)
    print(json.dumps({
        "replications": {key: {"comparison": value["comparison"]["flip8_minus_identity"], "ci": {"scan": value["comparison"]["scan_dice_95ci"], "subject": value["comparison"]["subject_dice_95ci"]}, "gates": value["gates"]} for key, value in results.items()},
        "cross_replication": report["cross_replication"], "test270_authorized": test_authorized,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
