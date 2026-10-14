#!/usr/bin/env python3
"""Build the train-only T1ce cache with fixed source 1st/99th percentile clipping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays  # noqa: E402
from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402
from sammed3d.t1ce_wt_training import _load_canonical_nifti, validate_row  # noqa: E402
from scripts.build_t1ce_wt_cache import n4_correct, sha256_file  # noqa: E402


SCHEMA_VERSION = 1
SOURCE_SHA256 = "6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65"
CONTROL_INDEX_SHA256 = "7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd"
CONTROL_LABELS_SHA256 = "bf470d48a5f115f09407ad351bb6dee11ee36bdd0c2a4285ae0bd9cc22125724"
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0
_CFG: dict[str, Any] = {}


def assert_train_only_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    for token in ("validation", "test270", "external75", "confirmation", "sealed"):
        if token in lowered:
            raise ValueError(f"forbidden evaluation-role token: {path}")


def percentile_clip_nonzero(image: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("image must be finite 3D")
    mask = image != 0
    count = int(mask.sum())
    if count < 2:
        raise ValueError("image has fewer than two nonzero voxels")
    values = np.asarray(image[mask], dtype=np.float64)
    low, high = np.percentile(
        values, [LOW_PERCENTILE, HIGH_PERCENTILE], method="linear"
    )
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError("invalid percentile thresholds")
    clipped = np.zeros_like(image)
    clipped_values = np.clip(image[mask], low, high)
    clipped[mask] = clipped_values
    mean = float(np.asarray(clipped_values, dtype=np.float64).mean())
    std = float(np.asarray(clipped_values, dtype=np.float64).std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
        raise ValueError("invalid post-clip population statistics")
    return np.ascontiguousarray(clipped), {
        "brain_voxels": count,
        "low": float(low),
        "high": float(high),
        "post_clip_mean": mean,
        "post_clip_std_population": std,
        "below_low": int(np.count_nonzero(values < low)),
        "above_high": int(np.count_nonzero(values > high)),
    }


def _worker_init(config: dict[str, Any]) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    _CFG.update(config)
    _CFG["images_fd"] = os.open(config["images_path"], os.O_WRONLY)
    _CFG["labels_fd"] = os.open(config["labels_path"], os.O_WRONLY)


def prepare_row(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validate_row(row)
    case_id = str(row["case_id"])
    image_nifti = _load_canonical_nifti(Path(str(row["image_path"])), f"{case_id} T1ce")
    label_nifti = _load_canonical_nifti(Path(str(row["label_path"])), f"{case_id} segmentation")
    if image_nifti.shape != label_nifti.shape:
        raise ValueError(f"{case_id} image/label shape mismatch")
    if not np.allclose(image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"{case_id} image/label affine mismatch")
    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    label = np.ascontiguousarray(np.asanyarray(label_nifti.dataobj))
    spacing = nib.affines.voxel_sizes(image_nifti.affine)
    native_shape = [int(value) for value in image.shape]
    image = n4_correct(image, 2)
    nonzero = np.nonzero(image > 0)
    if any(axis.size == 0 for axis in nonzero):
        raise RuntimeError(f"{case_id} has no positive N4 voxels")
    origin = [int(axis.min()) for axis in nonzero]
    stop = [int(axis.max()) + 1 for axis in nonzero]
    extent = [stop[axis] - origin[axis] for axis in range(3)]
    box = tuple(slice(origin[axis], stop[axis]) for axis in range(3))
    clipped, clip_stats = percentile_clip_nonzero(image[box])
    view = prepare_t1ce_supervised_arrays(
        case_id, clipped, label[box], spacing, output_shape=MODEL_SHAPE
    )
    cached_image = np.ascontiguousarray(view.image.float().numpy()[0], dtype=np.float32)
    cached_label = np.ascontiguousarray(view.target.numpy()[0], dtype=np.uint8)
    if cached_image.shape != MODEL_SHAPE or cached_label.shape != MODEL_SHAPE:
        raise RuntimeError(f"{case_id} unexpected cached shape")
    positives = int(cached_label.sum())
    if positives <= 0 or not np.isfinite(cached_image).all():
        raise RuntimeError(f"{case_id} invalid candidate cache arrays")
    return cached_image, cached_label, {
        "case_id": case_id,
        "origin": origin,
        "extent": extent,
        "native_shape": native_shape,
        "resized_positive_voxels": positives,
        "clip": clip_stats,
    }


def _process(task: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    index, row = task
    image, label, metadata = prepare_row(row)
    voxels = int(np.prod(MODEL_SHAPE))
    image_bytes = image.tobytes()
    label_bytes = label.tobytes()
    if len(image_bytes) != voxels * 4 or len(label_bytes) != voxels:
        raise RuntimeError("unexpected serialized cache size")
    os.pwrite(_CFG["images_fd"], image_bytes, index * voxels * 4)
    os.pwrite(_CFG["labels_fd"], label_bytes, index * voxels)
    return index, metadata


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    case_ids = [str(row.get("case_id", "")) for row in rows]
    subjects = {str(row.get("subject_id", "")) for row in rows}
    if len(rows) != 972 or len(set(case_ids)) != 972 or len(subjects) != 444:
        raise ValueError("manifest must be Full972 train role")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("manifest contains non-train role")
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    control_cache = args.control_cache.resolve()
    output = args.output_dir.resolve()
    for path in (manifest, control_cache, output):
        assert_train_only_path(path)
    if sha256_file(manifest) != SOURCE_SHA256:
        raise ValueError("Full972 manifest SHA-256 mismatch")
    if sha256_file(control_cache / "index.json") != CONTROL_INDEX_SHA256:
        raise ValueError("control cache index SHA-256 mismatch")
    if sha256_file(control_cache / "labels.u8") != CONTROL_LABELS_SHA256:
        raise ValueError("control labels SHA-256 mismatch")
    if output.exists():
        raise FileExistsError(f"refusing existing candidate cache: {output}")
    rows = load_manifest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    images_path = output / "images.f32"
    labels_path = output / "labels.u8"
    count = len(rows)
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(count, *MODEL_SHAPE)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(count, *MODEL_SHAPE)).flush()
    config = {"images_path": str(images_path), "labels_path": str(labels_path)}
    started = time.time()
    results: list[tuple[int, dict[str, Any]]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers, initializer=_worker_init, initargs=(config,)
    ) as pool:
        for done, result in enumerate(
            pool.imap_unordered(_process, list(enumerate(rows)), chunksize=1), start=1
        ):
            results.append(result)
            if done % 50 == 0 or done == count:
                rate = done / max(time.time() - started, 1e-9)
                print(f"{done}/{count} {rate:.2f} case/s eta={(count-done)/max(rate,1e-9):.0f}s", flush=True)
    results.sort(key=lambda item: item[0])
    metadata = [item[1] for item in results]
    labels_sha256 = sha256_file(labels_path)
    if labels_sha256 != CONTROL_LABELS_SHA256:
        raise RuntimeError("candidate labels are not byte-exact to control")
    lows = [float(item["clip"]["low"]) for item in metadata]
    highs = [float(item["clip"]["high"]) for item in metadata]
    if not all(np.isfinite(lows)) or not all(np.isfinite(highs)) or not all(a < b for a, b in zip(lows, highs)):
        raise RuntimeError("invalid candidate percentile audit")
    elapsed = time.time() - started
    index = {
        "schema": CACHE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_sha256": SOURCE_SHA256,
        "count": count,
        "case_ids": [item["case_id"] for item in metadata],
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "split": "train",
        "test_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "preprocessing": {
            "order": ["canonical_RAS", "N4_shrink_2", "image_nonzero_bounding_box_crop", "source_nonzero_linear_percentile_clip_1_99", "source_clipped_nonzero_population_zscore", "resize_128_cubed"],
            "spatial": "braincrop_resize",
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": True,
            "n4_shrink_factor": 2,
            "n4_implementation": "SimpleITK N4BiasFieldCorrectionImageFilter, 4x50 iterations, nonzero mask",
            "percentile_method": "numpy.percentile linear",
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
            "normalization": "source clipped nonzero-voxel population z-score before resize",
            "image_interpolation": "trilinear_align_corners_false",
            "label_interpolation": "nearest-exact",
            "augmentation": "not cached",
        },
        "crop_geometry": {item["case_id"]: {"origin": item["origin"], "extent": item["extent"], "native_shape": item["native_shape"]} for item in metadata},
        "per_case_resized_positive_voxels": {item["case_id"]: item["resized_positive_voxels"] for item in metadata},
        "per_case_clip": {item["case_id"]: item["clip"] for item in metadata},
        "images_sha256": sha256_file(images_path),
        "labels_sha256": labels_sha256,
        "control_labels_sha256": CONTROL_LABELS_SHA256,
        "labels_byte_exact_to_control": True,
        "build_seconds": elapsed,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "access": {"fit": True, "inner_metrics": False, "fixed_validation108": False, "outer": False, "confirmation": False, "test270": False, "ucsd_external75": False},
    }
    (output / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def verify(args: argparse.Namespace, index: dict[str, Any]) -> None:
    # Match the build workers exactly.  The first V1 post-build verification
    # accidentally used the parent process's multi-thread ITK default; the
    # cache itself was always built by _worker_init with one ITK thread.
    for variable in (
        "OMP_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    rows = load_manifest(args.manifest.resolve())
    images = np.memmap(args.output_dir.resolve() / "images.f32", dtype=np.float32, mode="r", shape=(972, *MODEL_SHAPE))
    labels = np.memmap(args.output_dir.resolve() / "labels.u8", dtype=np.uint8, mode="r", shape=(972, *MODEL_SHAPE))
    rng = np.random.default_rng(20261110)
    for slot in sorted(rng.choice(972, size=min(args.verify, 972), replace=False).tolist()):
        image, label, metadata = prepare_row(rows[slot])
        if not np.array_equal(image, np.asarray(images[slot])):
            raise RuntimeError(f"{metadata['case_id']} image rebuild differs")
        if not np.array_equal(label, np.asarray(labels[slot])):
            raise RuntimeError(f"{metadata['case_id']} label rebuild differs")
    print(f"verified {min(args.verify,972)} independent exact rebuilds", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--verify", type=int, default=5)
    args = parser.parse_args()
    index = build(args)
    verify(args, index)
    print(json.dumps({
        "status": "complete",
        "count": index["count"],
        "images_sha256": index["images_sha256"],
        "labels_sha256": index["labels_sha256"],
        "labels_byte_exact_to_control": index["labels_byte_exact_to_control"],
        "build_seconds": index["build_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
