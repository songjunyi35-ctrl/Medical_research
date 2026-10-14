#!/usr/bin/env python3
"""Evaluate a SAM-Med3D checkpoint with the frozen T1ce/WT three-click protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_sammed3d_t1ce_corrective3 import case_seed, sample_click  # noqa: E402
from sammed3d import engine, peft  # noqa: E402
from sammed3d.t1ce_wt_prompt import (  # noqa: E402
    ALLOWED_RAW_LABELS,
    MODEL_SHAPE,
    SCHEMA as T1CE_WT_SCHEMA,
)
from sammed3d.t1ce_wt_training import (  # noqa: E402
    GT_CROP,
    WHOLE_RESIZE,
    load_whole_volume_resize,
)


SCHEMA = "aflow.sammed3d.t1ce_wt_checkpoint_evaluation"
SCHEMA_VERSION = 2
NUM_CLICKS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(
    manifest: Path,
    *,
    expected_sha256: str,
    expected_split: str,
    max_cases: int | None,
) -> tuple[list[dict[str, Any]], int]:
    actual_sha256 = sha256_file(manifest)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"manifest SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("evaluation manifest is empty")
    case_ids = []
    for row in rows:
        if row.get("split") != expected_split:
            raise ValueError(f"manifest contains a non-{expected_split} row")
        modalities = row.get("modalities")
        image_path = row.get("image_path")
        if not image_path and (
            not isinstance(modalities, dict) or not modalities.get("t1c")
        ):
            raise ValueError("row lacks T1ce path")
        if not row.get("seg_path", row.get("label_path")):
            raise ValueError("row lacks segmentation path")
        if not row.get("subject_id"):
            raise ValueError("row lacks subject_id")
        case_ids.append(str(row.get("case_id")))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("manifest contains duplicate case IDs")
    total_rows = len(rows)
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        rows = rows[:max_cases]
    return rows, total_rows


def _paths(row: dict[str, Any]) -> tuple[Path, Path]:
    image_path = row.get("image_path")
    if not image_path:
        image_path = row["modalities"]["t1c"]
    label_path = row.get("seg_path", row.get("label_path"))
    return Path(str(image_path)), Path(str(label_path))


def load_official_crop(
    row: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image_path, label_path = _paths(row)
    image_nifti = nib.load(str(image_path), mmap=True)
    label_nifti = nib.load(str(label_path), mmap=True)
    image = np.asanyarray(image_nifti.dataobj)
    raw = np.asanyarray(label_nifti.dataobj)
    if image.shape != raw.shape:
        raise ValueError(f"{row['case_id']} image/label shape mismatch")
    if not np.allclose(
        image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5
    ):
        raise ValueError(f"{row['case_id']} image/label affine mismatch")
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1e-5):
        raise ValueError("segmentation contains non-integer values")
    labels = set(np.unique(rounded).astype(int).tolist())
    if not labels <= ALLOWED_RAW_LABELS:
        raise ValueError(f"unexpected raw labels: {sorted(labels)}")
    residual = np.isin(rounded, (1, 2, 3))
    full_voxels = int(residual.sum())
    if full_voxels == 0:
        raise ValueError(f"{row['case_id']} has empty WT foreground")
    subject = tio.Subject(
        image=tio.ScalarImage(
            tensor=torch.from_numpy(np.ascontiguousarray(image))[None].float(),
            affine=np.asarray(image_nifti.affine, dtype=np.float64),
        ),
        label=tio.LabelMap(
            tensor=torch.from_numpy(np.ascontiguousarray(residual))[None].to(
                torch.uint8
            ),
            affine=np.asarray(label_nifti.affine, dtype=np.float64),
        ),
    )
    transformed = tio.Compose(
        [
            tio.ToCanonical(),
            tio.CropOrPad(mask_name="label", target_shape=MODEL_SHAPE),
        ]
    )(subject)
    normalized = tio.ZNormalization(masking_method=lambda value: value > 0)(
        transformed.image.data.float()
    )
    target = transformed.label.data[0].bool()
    crop_voxels = int(target.sum())
    if tuple(normalized.shape) != (1, *MODEL_SHAPE):
        raise RuntimeError("official crop did not produce [1,128,128,128]")
    if crop_voxels == 0:
        raise RuntimeError("official crop lost all WT foreground")
    return normalized, target, {
        "spatial_preprocessing": GT_CROP,
        "uses_gt_for_localization": True,
        "full_target_voxels": full_voxels,
        "crop_target_voxels": crop_voxels,
        "crop_target_coverage": crop_voxels / full_voxels,
    }


def _whole_volume_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt either frozen split schema to the strict training loader contract."""
    image_path, label_path = _paths(row)
    return {
        "schema": T1CE_WT_SCHEMA,
        "schema_version": 1,
        "case_id": str(row["case_id"]),
        "subject_id": str(row["subject_id"]),
        "image_path": str(image_path),
        "label_path": str(label_path),
        "modality": "t1ce",
        "target": "WT",
        "foreground_labels": [1, 2, 3],
        "excluded_labels": [4],
        "split": str(row["split"]),
    }


def load_whole_volume_case(
    row: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image, target = load_whole_volume_resize(_whole_volume_training_row(row))
    target = target[0].bool()
    return image, target, {
        "spatial_preprocessing": WHOLE_RESIZE,
        "uses_gt_for_localization": False,
        "model_shape": list(MODEL_SHAPE),
        "model_target_voxels": int(target.sum()),
    }


def load_evaluation_case(
    row: dict[str, Any], spatial_preprocessing: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if spatial_preprocessing == GT_CROP:
        return load_official_crop(row)
    if spatial_preprocessing == WHOLE_RESIZE:
        return load_whole_volume_case(row)
    raise ValueError(f"unsupported spatial preprocessing: {spatial_preprocessing}")


def load_model(
    checkpoint: Path,
    *,
    device: torch.device,
    expected_sha256: str | None,
) -> tuple[torch.nn.Module, str, dict[str, Any]]:
    checkpoint_sha256 = sha256_file(checkpoint)
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    model, adapter_config = peft.load_model_with_adapters(
        checkpoint,
        lambda model_type: engine.sam_model_registry3D[model_type](checkpoint=None),
        device=device,
        default_model_type=engine.MODEL_TYPE,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    return model, checkpoint_sha256, adapter_config


def overlap_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.bool()
    target = target.bool()
    intersection = int(torch.logical_and(prediction, target).sum())
    prediction_voxels = int(prediction.sum())
    target_voxels = int(target.sum())
    dice_denom = prediction_voxels + target_voxels
    union = prediction_voxels + target_voxels - intersection
    return {
        "dsc": 2.0 * intersection / dice_denom if dice_denom else 1.0,
        "iou": intersection / union if union else 1.0,
        "prediction_voxels": prediction_voxels,
        "target_voxels": target_voxels,
    }


@torch.inference_mode()
def predict_click_curve(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    generator: np.random.Generator,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    encoder_input = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    target_cpu = target.bool().cpu()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        embedding = model.image_encoder(encoder_input)
    low_res = torch.zeros((1, 1, 32, 32, 32), device=device)
    points: list[tuple[int, int, int]] = []
    labels: list[int] = []
    prediction: torch.Tensor | None = None
    predictions: dict[int, torch.Tensor] = {}
    records = []
    for click_index in range(1, NUM_CLICKS + 1):
        try:
            point, label, region = sample_click(target_cpu, prediction, generator)
        except RuntimeError:
            if prediction is None:
                raise
            predictions[click_index] = prediction.clone()
            records.append(
                {
                    "click_index": click_index,
                    "status": "skipped_prediction_already_exact",
                    "accumulated_points": len(points),
                }
            )
            continue
        points.append(point)
        labels.append(label)
        point_tensor = torch.tensor(
            points, dtype=torch.float32, device=device
        ).reshape(1, len(points), 3)
        label_tensor = torch.tensor(
            labels, dtype=torch.int64, device=device
        ).reshape(1, len(labels))
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sparse, dense = model.prompt_encoder(
                points=(point_tensor, label_tensor),
                boxes=None,
                masks=low_res,
            )
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
        prediction = (logits[0, 0] > 0).cpu()
        predictions[click_index] = prediction
        records.append(
            {
                "click_index": click_index,
                "status": "sampled",
                "point_crop_grid": list(point),
                "label": label,
                "sampled_region": region,
                "accumulated_points": len(points),
                "points_passed_to_encoder": len(points),
                "recurrent_mask_prompt": True,
            }
        )
    return predictions, records


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for click_index in range(1, NUM_CLICKS + 1):
        key = str(click_index)
        dsc = [float(case["metrics"][key]["dsc"]) for case in cases]
        iou = [float(case["metrics"][key]["iou"]) for case in cases]
        by_subject: dict[str, list[float]] = defaultdict(list)
        for case in cases:
            by_subject[str(case["subject_id"])].append(
                float(case["metrics"][key]["dsc"])
            )
        subject_means = [
            statistics.fmean(values) for values in by_subject.values()
        ]
        records = [
            record
            for case in cases
            for record in case["clicks"]
            if int(record["click_index"]) == click_index
            and record["status"] == "sampled"
        ]
        summary[key] = {
            "scan_macro": {
                "cases": len(cases),
                "dsc_mean": statistics.fmean(dsc),
                "dsc_std_population": statistics.pstdev(dsc),
                "dsc_median": statistics.median(dsc),
                "iou_mean": statistics.fmean(iou),
            },
            "subject_balanced": {
                "subjects": len(subject_means),
                "dsc_mean": statistics.fmean(subject_means),
                "dsc_std_population": statistics.pstdev(subject_means),
            },
            "clicks": {
                "sampled": len(records),
                "skipped_exact": len(cases) - len(records),
                "positive": sum(int(record["label"]) == 1 for record in records),
                "negative": sum(int(record["label"]) == 0 for record in records),
            },
        }
    return summary


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows, total_rows = read_rows(
        args.manifest.resolve(),
        expected_sha256=args.expected_manifest_sha256,
        expected_split=args.expected_split,
        max_cases=args.max_cases,
    )
    device = torch.device(args.device)
    model, checkpoint_sha256, adapter_config = load_model(
        args.checkpoint.resolve(),
        device=device,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    cases = []
    start = time.time()
    for index, row in enumerate(rows, start=1):
        image, target, geometry = load_evaluation_case(
            row, args.spatial_preprocessing
        )
        predictions, clicks = predict_click_curve(
            model,
            image,
            target,
            device=device,
            generator=np.random.default_rng(case_seed(args.seed, row["case_id"])),
        )
        metrics = {
            str(click_index): overlap_metrics(prediction, target)
            for click_index, prediction in predictions.items()
        }
        cases.append(
            {
                "case_id": row["case_id"],
                "subject_id": row["subject_id"],
                "geometry": geometry,
                "clicks": clicks,
                "metrics": metrics,
            }
        )
        print(
            f"[{index}/{len(rows)}] {row['case_id']} "
            f"c1={metrics['1']['dsc']:.4f} "
            f"c2={metrics['2']['dsc']:.4f} "
            f"c3={metrics['3']['dsc']:.4f}",
            flush=True,
        )
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "input": "T1ce single channel",
            "target": "WT = raw labels {1,2,3}; RC label 4 excluded",
            "geometry": (
                "canonical whole-volume resize to 128^3; no GT localization"
                if args.spatial_preprocessing == WHOLE_RESIZE
                else "TorchIO ToCanonical + GT-centered CropOrPad 128^3"
            ),
            "spatial_preprocessing": args.spatial_preprocessing,
            "normalization": (
                "source nonzero-voxel population z-score exactly once"
                if args.spatial_preprocessing == WHOLE_RESIZE
                else "TorchIO positive-voxel ZNormalization"
            ),
            "clicks": (
                "click1 uniform GT foreground; click2/3 uniform XOR error; "
                "FN positive, FP negative"
            ),
            "prompt_history": "all points accumulated",
            "mask_prompt": "previous low-resolution mask recurrently supplied",
            "threshold": "raw logit > 0",
            "num_clicks": NUM_CLICKS,
        },
        "data": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": args.expected_manifest_sha256,
            "expected_split": args.expected_split,
            "selected_cases": len(rows),
            "full_manifest_cases": total_rows,
        },
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "model_type": engine.MODEL_TYPE,
            "adapter_config": adapter_config,
        },
        "seed": args.seed,
        "summary": summarize(cases),
        "elapsed_seconds": time.time() - start,
        "cases": cases,
    }
    write_json_atomic(args.output.resolve(), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-split", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--spatial-preprocessing",
        choices=(GT_CROP, WHOLE_RESIZE),
        default=GT_CROP,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
