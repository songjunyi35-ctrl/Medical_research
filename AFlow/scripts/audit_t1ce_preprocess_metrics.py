#!/usr/bin/env python3
"""Read-only development audit for the T1ce binary preprocessing contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sammed3d.t1ce_binary import (  # noqa: E402
    DEFAULT_MODEL_SHAPE,
    T1ceSupervisedCase,
    load_t1ce_supervised_view,
    score_t1ce_supervised_view,
)


ALLOWED_SPLITS = frozenset({"train", "validation"})


def read_development_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"refusing non-development row {path}:{line_number}: split={split!r}"
                )
            T1ceSupervisedCase.from_manifest_row(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"empty development manifest: {path}")
    return rows


def resize_nearest(mask: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(
        mask.float().reshape(1, 1, *mask.shape), size=shape, mode="nearest"
    )[0, 0].bool()


@lru_cache(maxsize=None)
def torch_resize_indices(source: int, output: int, mode: str) -> np.ndarray:
    coordinates = torch.arange(source, dtype=torch.float32).reshape(1, 1, source)
    return F.interpolate(coordinates, size=output, mode=mode)[0, 0].numpy().astype(np.int64)


def resize_nearest_numpy(
    mask: np.ndarray, shape: tuple[int, int, int], mode: str
) -> np.ndarray:
    indices = [
        torch_resize_indices(source, output, mode)
        for source, output in zip(mask.shape, shape, strict=True)
    ]
    return mask[np.ix_(*indices)]


def overlap(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    intersection = int(torch.logical_and(prediction, target).sum())
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    denominator = pred_count + target_count
    union = denominator - intersection
    dsc = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return dsc, iou


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {}
    return {
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--formal-metric-cases-per-split", type=int, default=0)
    parser.add_argument("--image-content-cases-per-split", type=int, default=32)
    parser.add_argument("--torch-equivalence-cases-per-split", type=int, default=8)
    args = parser.parse_args()

    manifests = (args.train_jsonl, args.validation_jsonl)
    rows = [row for path in manifests for row in read_development_rows(path)]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case ids across development manifests")
    train_subjects = {
        str(row["subject_id"]) for row in rows if row["split"] == "train"
    }
    validation_subjects = {
        str(row["subject_id"]) for row in rows if row["split"] == "validation"
    }
    if train_subjects & validation_subjects:
        raise ValueError("subject leakage between train and validation")
    image_content_cases = {
        str(row["case_id"])
        for split in ALLOWED_SPLITS
        for row in sorted(
            (item for item in rows if item["split"] == split),
            key=lambda item: hashlib.sha256(str(item["case_id"]).encode("ascii")).digest(),
        )[: args.image_content_cases_per_split]
    }

    split_cases: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    spacings: Counter[str] = Counter()
    orientations: Counter[str] = Counter()
    raw_label_sets: Counter[str] = Counter()
    raw_label_case_presence: Counter[int] = Counter()
    raw_label_voxels: Counter[int] = Counter()
    geometry_mismatches = 0
    source_empty = 0
    model_empty = 0
    restored_empty = 0
    target_outside_t1ce_nonzero_cases = 0
    target_outside_t1ce_nonzero_voxels = 0
    roundtrip_dsc: list[float] = []
    roundtrip_iou: list[float] = []
    volume_ratios: list[float] = []
    exact_roundtrip_dsc: list[float] = []
    exact_roundtrip_iou: list[float] = []
    exact_volume_ratios: list[float] = []
    source_target_ml: list[float] = []
    formal_metric_dsc_delta: list[float] = []
    formal_metric_iou_delta: list[float] = []
    formal_metric_hd95_mm: list[float] = []
    sampled_by_split: Counter[str] = Counter()
    torch_sampled_by_split: Counter[str] = Counter()
    torch_nearest_mismatches = 0
    image_content_cases_read = 0

    for index, row in enumerate(rows, start=1):
        case = T1ceSupervisedCase.from_manifest_row(row)
        image_nifti = nib.load(str(case.t1ce_path), mmap=True)
        seg_nifti = nib.load(str(case.seg_path), mmap=True)
        image_affine = np.asarray(image_nifti.affine, dtype=np.float64)
        seg_affine = np.asarray(seg_nifti.affine, dtype=np.float64)
        if image_nifti.shape != seg_nifti.shape or not np.allclose(
            image_affine, seg_affine, rtol=0.0, atol=1e-5
        ):
            geometry_mismatches += 1
            continue
        segmentation = np.asanyarray(seg_nifti.dataobj)
        if len(image_nifti.shape) != 3 or segmentation.ndim != 3:
            raise ValueError(f"{case.case_id} is not 3D")
        if np.issubdtype(segmentation.dtype, np.integer):
            if int(segmentation.min()) < 0 or int(segmentation.max()) > 4:
                raise ValueError(f"{case.case_id} contains labels outside 0..4")
            counts_all = np.bincount(segmentation.reshape(-1), minlength=5)
        else:
            if not np.isfinite(segmentation).all():
                raise ValueError(
                    f"{case.case_id} segmentation contains non-finite voxels"
                )
            rounded = np.rint(segmentation)
            if not np.allclose(segmentation, rounded, rtol=0.0, atol=1e-5):
                raise ValueError(f"{case.case_id} contains non-integer segmentation")
            if float(rounded.min()) < 0 or float(rounded.max()) > 4:
                raise ValueError(f"{case.case_id} contains labels outside 0..4")
            counts_all = np.bincount(rounded.astype(np.int64).reshape(-1), minlength=5)
        labels = np.flatnonzero(counts_all)
        counts = counts_all[labels]
        label_set = tuple(int(value) for value in labels)
        if not set(label_set) <= {0, 1, 2, 3, 4}:
            raise ValueError(f"{case.case_id} unexpected labels {label_set}")

        split = str(row["split"])
        split_cases[split] += 1
        image_shape = tuple(int(value) for value in image_nifti.shape)
        shapes[str(image_shape)] += 1
        spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image_affine))
        spacings[str(tuple(round(value, 8) for value in spacing))] += 1
        orientations[str(tuple(nib.aff2axcodes(image_affine)))] += 1
        raw_label_sets[str(label_set)] += 1
        for label, count in zip(labels, counts, strict=True):
            raw_label_case_presence[int(label)] += 1
            raw_label_voxels[int(label)] += int(count)

        source_target = (segmentation >= 1) & (segmentation <= 3)
        model_target = resize_nearest_numpy(
            source_target, DEFAULT_MODEL_SHAPE, "nearest"
        )
        restored = resize_nearest_numpy(model_target, image_shape, "nearest")
        exact_model_target = resize_nearest_numpy(
            source_target, DEFAULT_MODEL_SHAPE, "nearest-exact"
        )
        exact_restored = resize_nearest_numpy(
            exact_model_target, image_shape, "nearest-exact"
        )
        source_tensor = torch.from_numpy(source_target)
        restored_tensor = torch.from_numpy(restored)
        dsc, iou = overlap(restored_tensor, source_tensor)
        roundtrip_dsc.append(dsc)
        roundtrip_iou.append(iou)
        exact_dsc, exact_iou = overlap(
            torch.from_numpy(exact_restored), source_tensor
        )
        exact_roundtrip_dsc.append(exact_dsc)
        exact_roundtrip_iou.append(exact_iou)
        source_count = int(source_target.sum())
        restored_count = int(restored.sum())
        source_empty += source_count == 0
        model_empty += not bool(model_target.any())
        restored_empty += restored_count == 0
        volume_ratios.append(restored_count / source_count if source_count else 1.0)
        exact_volume_ratios.append(
            int(exact_restored.sum()) / source_count if source_count else 1.0
        )
        source_target_ml.append(source_count * float(np.prod(spacing)) / 1000.0)
        if case.case_id in image_content_cases:
            image = np.asanyarray(image_nifti.dataobj)
            if not np.isfinite(image).all():
                raise ValueError(f"{case.case_id} T1ce contains non-finite voxels")
            outside = int(np.logical_and(source_target, image == 0).sum())
            target_outside_t1ce_nonzero_voxels += outside
            target_outside_t1ce_nonzero_cases += outside > 0
            image_content_cases_read += 1

        if torch_sampled_by_split[split] < args.torch_equivalence_cases_per_split:
            for mode, numpy_model, numpy_restored in (
                ("nearest", model_target, restored),
                ("nearest-exact", exact_model_target, exact_restored),
            ):
                torch_model = F.interpolate(
                    source_tensor.float()[None, None],
                    size=DEFAULT_MODEL_SHAPE,
                    mode=mode,
                )[0, 0].bool()
                torch_restored = F.interpolate(
                    torch_model.float()[None, None], size=image_shape, mode=mode
                )[0, 0].bool()
                if not np.array_equal(
                    numpy_model, torch_model.numpy()
                ) or not np.array_equal(numpy_restored, torch_restored.numpy()):
                    torch_nearest_mismatches += 1
            torch_sampled_by_split[split] += 1

        if sampled_by_split[split] < args.formal_metric_cases_per_split:
            view = load_t1ce_supervised_view(case)
            metric = score_t1ce_supervised_view(view.target, view)
            formal_metric_dsc_delta.append(abs(metric.dsc - dsc))
            formal_metric_iou_delta.append(abs(metric.iou - iou))
            formal_metric_hd95_mm.append(metric.hd95_mm)
            sampled_by_split[split] += 1
        if index % 100 == 0:
            print(f"audited {index}/{len(rows)}", file=sys.stderr, flush=True)

    report = {
        "scope": {
            "manifests": [str(path) for path in manifests],
            "allowed_splits": sorted(ALLOWED_SPLITS),
            "cases": len(rows),
            "outer_test_nifti_opened": False,
        },
        "split_cases": dict(sorted(split_cases.items())),
        "subjects": {
            "train": len(train_subjects),
            "validation": len(validation_subjects),
            "overlap": len(train_subjects & validation_subjects),
        },
        "geometry": {
            "mismatches": geometry_mismatches,
            "shapes": dict(shapes),
            "spacings_mm": dict(spacings),
            "affine_orientations": dict(orientations),
        },
        "labels": {
            "sets": dict(raw_label_sets),
            "case_presence": {str(k): v for k, v in sorted(raw_label_case_presence.items())},
            "voxel_counts": {str(k): v for k, v in sorted(raw_label_voxels.items())},
            "source_empty_123": source_empty,
            "model_empty_123": model_empty,
            "restored_empty_123": restored_empty,
            "target_outside_t1ce_nonzero_cases": target_outside_t1ce_nonzero_cases,
            "target_outside_t1ce_nonzero_voxels": target_outside_t1ce_nonzero_voxels,
            "target_outside_t1ce_scope": {
                "selection": "sha256(case_id)_ascending_per_split",
                "cases": image_content_cases_read,
            },
            "source_target_volume_ml": summary(source_target_ml),
        },
        "nearest_roundtrip": {
            "path": "source->{128,128,128}->source",
            "legacy_nearest": {
                "dsc": summary(roundtrip_dsc),
                "iou": summary(roundtrip_iou),
                "restored_over_source_volume": summary(volume_ratios),
            },
            "center_aligned_nearest_exact": {
                "dsc": summary(exact_roundtrip_dsc),
                "iou": summary(exact_roundtrip_iou),
                "restored_over_source_volume": summary(exact_volume_ratios),
            },
            "torch_numpy_exact_equivalence": {
                "cases": sum(torch_sampled_by_split.values()),
                "mismatches": torch_nearest_mismatches,
            },
        },
        "formal_metric_sample": {
            "cases": len(formal_metric_dsc_delta),
            "max_abs_dsc_delta_vs_direct": max(formal_metric_dsc_delta, default=0.0),
            "max_abs_iou_delta_vs_direct": max(formal_metric_iou_delta, default=0.0),
            "hd95_mm": summary(formal_metric_hd95_mm),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
