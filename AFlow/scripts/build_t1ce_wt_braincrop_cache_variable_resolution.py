#!/usr/bin/env python3
"""Build true 128^3/160^3 image-only brain-crop caches from raw train NIfTI.

The historical fixed-128 builder is hash-bound to completed work and remains
untouched.  This isolated entry point changes only ``output_shape`` while
preserving canonical RAS loading, N4 shrink-2 correction, image-nonzero crop,
source-population normalization, and trilinear image / nearest-exact label
resize.  It accepts only the frozen 756-case historical train-cohort view and
has no selector, confirmation, or test mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_binary import (  # noqa: E402
    CASE_ID_RE,
    prepare_t1ce_supervised_arrays,
    residual_tumor_target,
)
from sammed3d.t1ce_wt_training import _load_canonical_nifti  # noqa: E402
from sammed3d.t1ce_wt_variable_resolution import (  # noqa: E402
    CACHE_SCHEMA,
    FORMAL_RESOLUTIONS,
    SPATIAL,
    cubic_shape,
)
from scripts.build_t1ce_wt_cache import n4_correct  # noqa: E402


SCHEMA_VERSION = 1
FORBIDDEN_OUTPUT_TOKENS = ("selector", "confirmation", "sealed", "test270")
PART_RE = re.compile(r"^part([123])$")
_CFG: dict[str, Any] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_development_output(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in FORBIDDEN_OUTPUT_TOKENS):
        raise ValueError(f"forbidden data role in output path: {path}")


def read_train_cohort(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            case_id = str(value.get("case_id", ""))
            subject_id = str(value.get("subject_id", ""))
            if CASE_ID_RE.fullmatch(case_id) is None:
                raise ValueError(f"{path}:{line_number} invalid case_id")
            if subject_id != case_id.rsplit("-", 1)[0]:
                raise ValueError(f"{path}:{line_number} subject_id mismatch")
            if value.get("role") != "train":
                raise ValueError(f"{path}:{line_number} is not an authorized train row")
            rows.append({"case_id": case_id, "subject_id": subject_id})
    case_ids = [row["case_id"] for row in rows]
    if not rows or len(case_ids) != len(set(case_ids)):
        raise ValueError("train cohort is empty or contains duplicate cases")
    return rows


def read_case_views(paths: list[Path]) -> list[str]:
    case_ids: list[str] = []
    for path in paths:
        assert_development_output(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                case_id = str(value.get("case_id", ""))
                if CASE_ID_RE.fullmatch(case_id) is None:
                    raise ValueError(f"{path}:{line_number} invalid case_id")
                case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case views overlap or contain duplicate cases")
    return case_ids


def resolve_case_paths(raw_root: Path, case_id: str) -> tuple[Path, Path, str]:
    """Resolve one authorized case without enumerating any other cohort role."""

    matches: list[tuple[Path, Path, str]] = []
    for part_number in (1, 2, 3):
        part = f"part{part_number}"
        source_part = f"Part{part_number}"
        case_root = raw_root / part / source_part / case_id
        image = case_root / f"{case_id}-t1c.nii"
        label = case_root / f"{case_id}-seg.nii"
        image_exists = image.is_file()
        label_exists = label.is_file()
        if image_exists != label_exists:
            raise FileNotFoundError(f"incomplete raw T1c/seg pair for {case_id}: {case_root}")
        if image_exists:
            matches.append((image, label, source_part))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one raw T1c/seg pair for {case_id}, found {len(matches)}"
        )
    return matches[0]


def _worker_init(cfg: dict[str, Any]) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    _CFG.update(cfg)
    _CFG["images_fd"] = os.open(cfg["images_path"], os.O_WRONLY)
    _CFG["labels_fd"] = os.open(cfg["labels_path"], os.O_WRONLY)


def _prepare(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    case_id = row["case_id"]
    image_nifti = _load_canonical_nifti(Path(row["image_path"]), f"{case_id} T1ce")
    label_nifti = _load_canonical_nifti(Path(row["label_path"]), f"{case_id} segmentation")
    if image_nifti.shape != label_nifti.shape:
        raise ValueError(f"{case_id} image/label shape mismatch")
    if not np.allclose(image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"{case_id} canonical image/label affine mismatch")

    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    label = np.ascontiguousarray(np.asanyarray(label_nifti.dataobj))
    spacing = nib.affines.voxel_sizes(image_nifti.affine)
    native_shape = [int(value) for value in image.shape]
    if _CFG["n4"]:
        image = n4_correct(image, int(_CFG["n4_shrink"]))

    nonzero = np.nonzero(image > 0)
    if any(axis.size == 0 for axis in nonzero):
        raise RuntimeError(f"{case_id} has no nonzero image voxels")
    origin = [int(axis.min()) for axis in nonzero]
    stop = [int(axis.max()) + 1 for axis in nonzero]
    extent = [stop[axis] - origin[axis] for axis in range(3)]
    box = tuple(slice(origin[axis], stop[axis]) for axis in range(3))
    image = image[box]
    label = label[box]
    source_positives = int(residual_tumor_target(label).sum())
    if source_positives <= 0:
        raise RuntimeError(f"{case_id} has empty source WT after image-only crop")

    shape = tuple(int(value) for value in _CFG["model_shape"])
    view = prepare_t1ce_supervised_arrays(
        case_id, image, label, spacing, output_shape=shape
    )
    cached_image = np.ascontiguousarray(view.image.float().numpy()[0], dtype=np.float32)
    cached_label = np.ascontiguousarray(view.target.numpy()[0], dtype=np.uint8)
    if cached_image.shape != shape or cached_label.shape != shape:
        raise RuntimeError(f"{case_id} unexpected cached shape")
    resized_positives = int(cached_label.sum())
    if resized_positives <= 0 or not np.isfinite(cached_image).all():
        raise RuntimeError(f"{case_id} invalid resized image/WT")
    metadata = {
        "case_id": case_id,
        "source_positive_voxels": source_positives,
        "resized_positive_voxels": resized_positives,
        "origin": origin,
        "extent": extent,
        "native_shape": native_shape,
        "source_part": row["source_part"],
    }
    return cached_image, cached_label, metadata


def _process(task: tuple[int, dict[str, str]]) -> tuple[int, dict[str, Any]]:
    index, row = task
    image, label, metadata = _prepare(row)
    voxels = int(np.prod(_CFG["model_shape"]))
    image_bytes = image.tobytes()
    label_bytes = label.tobytes()
    if len(image_bytes) != voxels * 4 or len(label_bytes) != voxels:
        raise RuntimeError(f"{row['case_id']} unexpected serialized size")
    os.pwrite(_CFG["images_fd"], image_bytes, index * voxels * 4)
    os.pwrite(_CFG["labels_fd"], label_bytes, index * voxels)
    return index, metadata


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve()
    if sha256_file(manifest) != args.expected_manifest_sha256:
        raise ValueError("train cohort manifest SHA-256 mismatch")
    source_rows = read_train_cohort(manifest)
    if len(source_rows) != args.expected_source_cases:
        raise ValueError(
            f"train cohort has {len(source_rows)} rows, expected {args.expected_source_cases}"
        )
    if not args.smoke and args.resolution not in FORMAL_RESOLUTIONS:
        raise ValueError("formal resolution must be 128 or 160")
    if not args.smoke and args.case_view:
        raise ValueError("formal cache must materialize the complete frozen train cohort")

    rows_by_case = {row["case_id"]: row for row in source_rows}
    if args.case_view:
        requested = read_case_views(args.case_view)
        missing = [case_id for case_id in requested if case_id not in rows_by_case]
        if missing:
            raise ValueError("case view contains cases outside the frozen train cohort")
        selected = [rows_by_case[case_id] for case_id in requested]
    else:
        selected = source_rows

    raw_root = args.raw_root.resolve()
    resolved: list[dict[str, str]] = []
    for row in selected:
        image, label, source_part = resolve_case_paths(raw_root, row["case_id"])
        resolved.append(
            {
                **row,
                "image_path": str(image),
                "label_path": str(label),
                "source_part": source_part,
            }
        )

    output = args.output_dir.resolve()
    assert_development_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty cache: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shape = cubic_shape(args.resolution)
    images_path = output / "images.f32"
    labels_path = output / "labels.u8"
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(len(resolved), *shape)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(len(resolved), *shape)).flush()

    cfg = {
        "n4": bool(args.n4),
        "n4_shrink": int(args.n4_shrink),
        "model_shape": list(shape),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
    }
    started = time.time()
    results: list[tuple[int, dict[str, Any]]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers, initializer=_worker_init, initargs=(cfg,)
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(_process, enumerate(resolved), chunksize=1), start=1
        ):
            results.append(result)
            if completed % 25 == 0 or completed == len(resolved):
                elapsed = max(time.time() - started, 1e-9)
                rate = completed / elapsed
                print(
                    f"{completed}/{len(resolved)} rate={rate:.3f} case/s "
                    f"eta={(len(resolved) - completed) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )
    results.sort(key=lambda item: item[0])
    metadata = [item[1] for item in results]
    source_positive = {
        item["case_id"]: item["source_positive_voxels"] for item in metadata
    }
    resized_positive = {
        item["case_id"]: item["resized_positive_voxels"] for item in metadata
    }
    elapsed = time.time() - started
    index = {
        "schema": CACHE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_from_raw_nifti": True,
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest),
        "source_manifest_count": len(source_rows),
        "count": len(metadata),
        "case_ids": [item["case_id"] for item in metadata],
        "model_shape": list(shape),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "split": "historical_train_cohort_only",
        "test_access": False,
        "outer_selector_access": False,
        "confirmation_access": False,
        "preprocessing": {
            "order": [
                "canonical_RAS",
                "N4_shrink_2",
                "image_nonzero_bounding_box_crop",
                "source_nonzero_population_zscore",
                "resize_to_declared_cubic_grid",
            ],
            "spatial": SPATIAL,
            "output_resolution": args.resolution,
            "image_interpolation": "trilinear_align_corners_false",
            "label_interpolation": "nearest-exact",
            "normalization": "source nonzero-voxel population z-score before resize",
            "uses_gt_for_localization": False,
            "n4_bias_field_correction": bool(args.n4),
            "n4_shrink_factor": int(args.n4_shrink) if args.n4 else None,
            "n4_implementation": (
                "SimpleITK N4BiasFieldCorrectionImageFilter, 4x50 iterations, nonzero mask"
                if args.n4
                else None
            ),
            "source_is_existing_128_cache": False,
        },
        "per_case_source_positive_voxels": source_positive,
        "source_positive_voxels_sha256": stable_sha256(source_positive),
        "per_case_resized_positive_voxels": resized_positive,
        "crop_geometry": {
            item["case_id"]: {
                "origin": item["origin"],
                "extent": item["extent"],
                "native_shape": item["native_shape"],
                "source_part": item["source_part"],
            }
            for item in metadata
        },
        "images_sha256": sha256_file(images_path),
        "labels_sha256": sha256_file(labels_path),
        "builder": "scripts/build_t1ce_wt_braincrop_cache_variable_resolution.py",
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "build_seconds": round(elapsed, 3),
        "smoke": bool(args.smoke),
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def verify(args: argparse.Namespace, index: dict[str, Any]) -> None:
    output = args.output_dir.resolve()
    shape = tuple(index["model_shape"])
    images = np.memmap(
        output / "images.f32", dtype=np.float32, mode="r", shape=(index["count"], *shape)
    )
    labels = np.memmap(
        output / "labels.u8", dtype=np.uint8, mode="r", shape=(index["count"], *shape)
    )
    source_rows = read_train_cohort(args.manifest.resolve())
    rows_by_case = {row["case_id"]: row for row in source_rows}
    raw_root = args.raw_root.resolve()
    verify_count = min(int(args.verify), int(index["count"]))
    picks = np.linspace(0, index["count"] - 1, num=verify_count, dtype=int).tolist()
    _CFG.update({"n4": bool(args.n4), "n4_shrink": args.n4_shrink, "model_shape": shape})
    for slot in picks:
        case_id = index["case_ids"][slot]
        image_path, label_path, source_part = resolve_case_paths(raw_root, case_id)
        row = {
            **rows_by_case[case_id],
            "image_path": str(image_path),
            "label_path": str(label_path),
            "source_part": source_part,
        }
        image, label, metadata = _prepare(row)
        if not np.array_equal(image, np.asarray(images[slot])):
            raise RuntimeError(f"{case_id} image byte reproduction failed")
        if not np.array_equal(label, np.asarray(labels[slot])):
            raise RuntimeError(f"{case_id} label byte reproduction failed")
        if metadata["source_positive_voxels"] != index[
            "per_case_source_positive_voxels"
        ][case_id]:
            raise RuntimeError(f"{case_id} source WT volume reproduction failed")
        print(f"verified raw reproduction: {case_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-cases", type=int, default=756)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=FORMAL_RESOLUTIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-view", type=Path, action="append", default=[])
    parser.add_argument("--n4", action="store_true")
    parser.add_argument("--n4-shrink", type=int, default=2)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--verify", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.n4 or args.n4_shrink != 2:
        raise ValueError("frozen protocol requires N4 with shrink factor 2")
    index = build(args)
    if args.verify:
        verify(args, index)
    print(json.dumps({
        "cache": str(args.output_dir.resolve()),
        "count": index["count"],
        "resolution": args.resolution,
        "images_sha256": index["images_sha256"],
        "labels_sha256": index["labels_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
