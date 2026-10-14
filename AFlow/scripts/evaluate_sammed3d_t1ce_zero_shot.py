#!/usr/bin/env python3
"""Evaluate the untouched SAM-Med3D checkpoint on the sealed T1ce test split.

Predictions are label-free. Ground truth is opened only after prompts have been
constructed and inference has completed for a case.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.paths import SAMMED3D_CHECKPOINT, SAMMED3D_ROOT  # noqa: E402

for path in (ROOT, SAMMED3D_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from segment_anything.build_sam3D import sam_model_registry3D  # noqa: E402

from sammed3d.t1ce_binary import (  # noqa: E402
    ALLOWED_RAW_LABELS,
    T1ceInferenceCase,
    T1ceSupervisedCase,
    load_t1ce_inference_view,
    load_t1ce_supervised_view,
)


SCHEMA = "aflow.sammed3d.t1ce_zero_shot_test"
SCHEMA_VERSION = 1
MODEL_TYPE = "vit_b_ori"
MODEL_SHAPE = (128, 128, 128)
DEFAULT_MANIFEST = (
    ROOT
    / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1"
    / "sealed_test/test_cases.jsonl"
)
EXPECTED_MANIFEST_SHA256 = (
    "7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2"
)
DEFAULT_CHECKPOINT = SAMMED3D_CHECKPOINT
EXPECTED_CHECKPOINT_SHA256 = (
    "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
)
PROMPT_MODES = ("unprompted", "t1ce_brain_center_point")
TARGETS = ("et", "residual")
GRIDS = ("model_grid", "source_grid")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(
    manifest: Path, num_shards: int, shard_index: int, max_cases: int | None
) -> tuple[list[dict[str, Any]], int]:
    if sha256_file(manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("sealed test manifest SHA-256 mismatch")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) != 270:
        raise RuntimeError(f"sealed test must contain 270 cases, found {len(rows)}")
    case_ids: list[str] = []
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("sealed test contains a non-test row")
        if not isinstance(row.get("modalities"), dict) or "t1c" not in row["modalities"]:
            raise ValueError("sealed test row is missing modalities.t1c")
        if not row.get("seg_path"):
            raise ValueError("sealed test row is missing seg_path")
        case_ids.append(str(row.get("case_id")))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("sealed test contains duplicate case IDs")
    selected = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("selected shard is empty")
    return selected, len(rows)


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("original checkpoint SHA-256 mismatch")
    model = sam_model_registry3D[MODEL_TYPE](checkpoint=None)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    if any(str(key).startswith("module.") for key in payload):
        payload = {
            (str(key)[7:] if str(key).startswith("module.") else str(key)): value
            for key, value in payload.items()
        }
    model.load_state_dict(payload, strict=True)
    del payload
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("zero-shot model has trainable parameters")
    return model


def brain_center_point(brain_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    locations = torch.nonzero(brain_mask, as_tuple=False)
    if locations.numel() == 0:
        raise ValueError("resized T1ce brain mask is empty")
    lower = locations.amin(dim=0)
    upper = locations.amax(dim=0)
    return ((lower + upper).float() / 2.0).to(device).reshape(1, 1, 3)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    brain_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoder_input = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        embedding = model.image_encoder(encoder_input)
        outputs: dict[str, torch.Tensor] = {}
        for mode in PROMPT_MODES:
            points = None
            if mode == "t1ce_brain_center_point":
                coords = brain_center_point(brain_mask, device)
                labels = torch.ones((1, 1), dtype=torch.int64, device=device)
                points = (coords, labels)
            sparse, dense = model.prompt_encoder(points=points, boxes=None, masks=None)
            low_res, _ = model.mask_decoder(
                image_embeddings=embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logits = F.interpolate(
                low_res, size=MODEL_SHAPE, mode="trilinear", align_corners=False
            )
            outputs[mode] = (logits[0, 0] >= 0).cpu()
    return outputs


def load_et_source(row: dict[str, Any], expected_shape: tuple[int, int, int]) -> torch.Tensor:
    image = nib.load(str(row["seg_path"]), mmap=True)
    raw = np.asanyarray(image.dataobj)
    if tuple(raw.shape) != expected_shape:
        raise ValueError(f"ET segmentation shape {raw.shape} != T1ce shape {expected_shape}")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1e-5):
        raise ValueError("segmentation contains non-integer values")
    labels = set(np.unique(rounded).astype(int).tolist())
    if not labels <= ALLOWED_RAW_LABELS:
        raise ValueError(f"unexpected raw labels: {sorted(labels)}")
    return torch.from_numpy(np.asarray(rounded == 3))


def resize_binary(mask: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(
        mask.float().reshape(1, 1, *mask.shape), size=shape, mode="nearest-exact"
    )[0, 0].bool()


def overlap_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    prediction = prediction.bool()
    target = target.bool()
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    intersection = int(torch.logical_and(prediction, target).sum())
    denominator = pred_count + target_count
    union = denominator - intersection
    dsc = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return {
        "dsc": float(dsc),
        "iou": float(iou),
        "prediction_voxels": pred_count,
        "target_voxels": target_count,
        "intersection_voxels": intersection,
        "prediction_present": pred_count > 0,
        "target_present": target_count > 0,
    }


def summarize(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    case_list = list(cases)
    result: dict[str, Any] = {}
    for mode in PROMPT_MODES:
        result[mode] = {}
        for target in TARGETS:
            result[mode][target] = {}
            for grid in GRIDS:
                rows = [case["metrics"][mode][target][grid] for case in case_list]
                dsc = [float(row["dsc"]) for row in rows]
                iou = [float(row["iou"]) for row in rows]
                present_dsc = [
                    float(row["dsc"]) for row in rows if bool(row["target_present"])
                ]
                result[mode][target][grid] = {
                    "cases": len(rows),
                    "target_present_cases": sum(bool(row["target_present"]) for row in rows),
                    "prediction_present_cases": sum(
                        bool(row["prediction_present"]) for row in rows
                    ),
                    "dsc_macro_mean": statistics.fmean(dsc),
                    "dsc_macro_std_population": statistics.pstdev(dsc),
                    "dsc_median": statistics.median(dsc),
                    "dsc_target_present_mean": (
                        statistics.fmean(present_dsc) if present_dsc else None
                    ),
                    "iou_macro_mean": statistics.fmean(iou),
                }
    return result


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    rows, total_cases = read_rows(
        manifest, args.num_shards, args.shard_index, args.max_cases
    )
    device = torch.device(args.device)
    model = load_model(checkpoint, device)
    started = time.monotonic()
    cases: list[dict[str, Any]] = []
    for sequence, row in enumerate(rows, start=1):
        case_started = time.monotonic()
        inference_view = load_t1ce_inference_view(
            T1ceInferenceCase(
                case_id=str(row["case_id"]),
                t1ce_path=Path(row["modalities"]["t1c"]),
            )
        )
        predictions = predict(
            model, inference_view.image, inference_view.brain_mask, device
        )
        # Supervision is not opened until all label-free predictions exist.
        view = load_t1ce_supervised_view(T1ceSupervisedCase.from_manifest_row(row))
        residual_source = view.source_target[0]
        et_source = load_et_source(row, view.geometry.original_shape)
        targets = {
            "et": {
                "model_grid": resize_binary(et_source, MODEL_SHAPE),
                "source_grid": et_source,
            },
            "residual": {
                "model_grid": view.target[0],
                "source_grid": residual_source,
            },
        }
        metrics: dict[str, Any] = {}
        for mode, model_prediction in predictions.items():
            source_prediction = resize_binary(
                model_prediction, view.geometry.original_shape
            )
            metrics[mode] = {}
            for target_name in TARGETS:
                metrics[mode][target_name] = {
                    "model_grid": overlap_metrics(
                        model_prediction, targets[target_name]["model_grid"]
                    ),
                    "source_grid": overlap_metrics(
                        source_prediction, targets[target_name]["source_grid"]
                    ),
                }
        cases.append(
            {
                "case_id": row["case_id"],
                "metrics": metrics,
                "original_shape": list(view.geometry.original_shape),
                "original_spacing_mm": list(view.geometry.original_spacing_mm),
                "normalization": {
                    "convention": view.normalization.convention,
                    "nonzero_voxels": view.normalization.nonzero_voxels,
                },
                "elapsed_seconds": time.monotonic() - case_started,
            }
        )
        print(
            f"[{sequence}/{len(rows)}] {row['case_id']} "
            f"unprompted_residual_dsc="
            f"{metrics['unprompted']['residual']['model_grid']['dsc']:.6f}",
            flush=True,
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": {
            "modality": "T1ce only (manifest key t1c)",
            "training": "none; untouched pretrained checkpoint",
            "prompt_modes": {
                "unprompted": "points=None, boxes=None, masks=None",
                "t1ce_brain_center_point": (
                    "one positive point at resized nonzero-T1ce bounding-box center"
                ),
            },
            "prompt_label_access": False,
            "threshold": "raw_logit >= 0 (sigmoid >= 0.5), fixed",
            "postprocessing": "none",
            "tta": "none",
            "input_geometry": "whole-volume resize to 128^3; no crop",
            "normalization": "source T1ce nonzero population z-score",
            "targets": {
                "et": "raw BraTS label 3",
                "residual": "raw BraTS labels {1,2,3}; label 4 excluded",
            },
            "metric": "case-macro binary DSC and IoU",
        },
        "bindings": {
            "manifest": str(manifest),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_total_cases": total_cases,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "model_type": MODEL_TYPE,
        },
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selected_cases": len(rows),
            "max_cases": args.max_cases,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": time.monotonic() - started,
        },
        "summary": summarize(cases),
        "cases": cases,
    }


def merge_reports(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports:
        raise ValueError("at least one merge input is required")
    for report in reports:
        if report.get("schema") != SCHEMA or report.get("status") != "complete":
            raise ValueError("cannot merge an incompatible or incomplete report")
        if report["bindings"]["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise ValueError("cannot merge report with another manifest")
        if report["bindings"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("cannot merge report with another checkpoint")
    cases = [case for report in reports for case in report["cases"]]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("merged reports contain duplicate cases")
    if len(cases) != 270:
        raise ValueError(f"complete merged report must contain 270 cases, found {len(cases)}")
    cases.sort(key=lambda item: item["case_id"])
    first = reports[0]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_merged_270_case_test",
        "protocol": first["protocol"],
        "bindings": first["bindings"],
        "merge": {
            "inputs": [str(path.resolve()) for path in paths],
            "cases": len(cases),
        },
        "summary": summarize(cases),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-input", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = merge_reports(args.merge_input) if args.merge_input else evaluate(args)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
