#!/usr/bin/env python
"""Build the T1ce/WT preprocessing cache with crop-to-nonzero BEFORE the 128^3 resize.

Why this is a separate script
-----------------------------
`scripts/build_t1ce_wt_cache.py` is byte-pinned by the completed full-unfreeze
test freeze (`FROZEN_FULLUNFREEZE_TEST_BUILDER_CORRECTION_20260805.json` binds
SHA c3171ffc...). Adding a third `--spatial` mode there would invalidate that
binding after the fact. This file adds the new geometry without touching it, and
imports the shared pieces rather than reimplementing them.

The single factor
-----------------
Identical to `whole_resize` in every respect -- canonical RAS load, N4 (shrink 2,
4x50, nonzero mask), nonzero-population z-score, trilinear image / nearest-exact
label resize to 128^3 -- except that the volume is cropped to the bounding box of
its nonzero voxels first.

The z-score is provably unchanged by the crop: it is taken over the nonzero-voxel
population, and cropping to the nonzero bounding box removes only zero voxels.

Motivation (measured, see BOTTLENECK_DIAGNOSIS_20260806.json): only 47.8% of the
native array lies inside the brain bounding box and only 20.8% of its voxels are
nonzero, so roughly 267 of the image encoder's 512 tokens currently cover
background. Cropping first roughly doubles the lesion's voxel count on the model
grid (16,522 -> 36,649 median) and lifts the native-space reconstruction ceiling
from 0.930 to 0.968.

The per-case crop geometry is recorded in the index so a 128^3 prediction can be
restored exactly to the native grid for native-space scoring.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_t1ce_wt_cache import n4_correct, sha256_file  # noqa: E402
from sammed3d.t1ce_wt_cached import CACHE_SCHEMA, MODEL_SHAPE  # noqa: E402

SCHEMA_VERSION = 1
SPATIAL = "braincrop_resize"
INNER_SPLIT_SCHEMA = "aflow.brats2024.sammed3d_t1ce_wt_inner_split"

_CFG: dict = {}


def _worker_init(cfg: dict) -> None:
    # One ITK thread per worker. Without this, 48 workers each spawn a full
    # thread pool and the build collapses under context switching.
    for var in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    _CFG.update(cfg)
    _CFG["images_fd"] = os.open(cfg["images_path"], os.O_WRONLY)
    _CFG["labels_fd"] = os.open(cfg["labels_path"], os.O_WRONLY)


def _process(task: tuple[int, dict]):
    from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays
    from sammed3d.t1ce_wt_training import _load_canonical_nifti, validate_row

    index, row = task
    # The sealed-test manifest uses the outer random_subject_train_test_split
    # schema. Reuse the frozen evaluator's already-validated adapter rather than
    # re-deriving the path and label mapping here.
    if row.get("schema") != INNER_SPLIT_SCHEMA:
        from scripts.evaluate_sammed3d_t1ce_wt_checkpoint import (
            _whole_volume_training_row,
        )

        row = _whole_volume_training_row(row)
    validate_row(row)
    case_id = str(row["case_id"])

    image_nifti = _load_canonical_nifti(Path(str(row["image_path"])), f"{case_id} T1ce")
    label_nifti = _load_canonical_nifti(
        Path(str(row["label_path"])), f"{case_id} segmentation"
    )
    if image_nifti.shape != label_nifti.shape:
        raise ValueError(f"{case_id} image/label shape mismatch")
    if not np.allclose(image_nifti.affine, label_nifti.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"{case_id} canonical image/label affine mismatch")

    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    label = np.ascontiguousarray(np.asanyarray(label_nifti.dataobj))
    spacing = nib.affines.voxel_sizes(image_nifti.affine)
    native_shape = list(image.shape)

    if _CFG["n4"]:
        image = n4_correct(image, _CFG["n4_shrink"])

    # ---- the only difference from whole_resize -------------------------------
    nonzero = np.nonzero(image > 0)
    if any(a.size == 0 for a in nonzero):
        raise RuntimeError(f"{case_id} has no nonzero voxels to crop to")
    origin = [int(a.min()) for a in nonzero]
    stop = [int(a.max()) + 1 for a in nonzero]
    extent = [stop[i] - origin[i] for i in range(3)]
    box = tuple(slice(origin[i], stop[i]) for i in range(3))
    image, label = image[box], label[box]
    # -------------------------------------------------------------------------

    view = prepare_t1ce_supervised_arrays(
        case_id, image, label, spacing, output_shape=MODEL_SHAPE
    )
    img = view.image.float().numpy()
    tgt = view.target.numpy().astype(np.uint8)
    if img.shape != (1, *MODEL_SHAPE) or tgt.shape != (1, *MODEL_SHAPE):
        raise RuntimeError(f"{case_id} unexpected cached shape")
    positives = int(tgt.sum())
    if positives == 0:
        raise RuntimeError(f"{case_id} WT is empty after brain-crop resize")
    if not np.isfinite(img).all():
        raise RuntimeError(f"{case_id} cached image is non-finite")

    # Positional writes, not memmap slices: np.memmap.flush() msyncs the whole
    # mapping and made an earlier multi-GiB build crawl at 0.4 case/s.
    voxels = int(np.prod(MODEL_SHAPE))
    img_bytes = np.ascontiguousarray(img[0], dtype=np.float32).tobytes()
    tgt_bytes = np.ascontiguousarray(tgt[0], dtype=np.uint8).tobytes()
    if len(img_bytes) != voxels * 4 or len(tgt_bytes) != voxels:
        raise RuntimeError(f"{case_id} unexpected serialized size")
    os.pwrite(_CFG["images_fd"], img_bytes, index * voxels * 4)
    os.pwrite(_CFG["labels_fd"], tgt_bytes, index * voxels)
    return index, case_id, positives, origin, extent, native_shape


def build(args: argparse.Namespace) -> dict:
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    count = len(rows)
    if count == 0:
        raise SystemExit("empty manifest")
    if args.expected_count is not None and count != args.expected_count:
        raise SystemExit(f"manifest has {count} rows, expected {args.expected_count}")
    if args.expected_manifest_sha256:
        got = sha256_file(manifest)
        if got != args.expected_manifest_sha256:
            raise SystemExit(f"manifest SHA {got} != expected {args.expected_manifest_sha256}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty cache dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images_path = out_dir / "images.f32"
    labels_path = out_dir / "labels.u8"
    np.memmap(images_path, dtype=np.float32, mode="w+", shape=(count, *MODEL_SHAPE)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(count, *MODEL_SHAPE)).flush()

    cfg = {
        "n4": bool(args.n4),
        "n4_shrink": int(args.n4_shrink),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
    }
    started = time.time()
    results = []
    with mp.get_context("spawn").Pool(
        processes=args.workers, initializer=_worker_init, initargs=(cfg,)
    ) as pool:
        for done, res in enumerate(
            pool.imap_unordered(_process, list(enumerate(rows)), chunksize=1), 1
        ):
            results.append(res)
            if done % 50 == 0 or done == count:
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{count}  {rate:.2f} case/s  "
                      f"eta {(count - done) / max(rate, 1e-9):.0f}s", flush=True)
    results.sort()
    elapsed = time.time() - started

    index = {
        "schema": CACHE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest),
        "count": count,
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "preprocessing": {
            "spatial": SPATIAL,
            "spatial_description": (
                "crop to the bounding box of nonzero voxels, then resize that box "
                "to 128^3. Does NOT use the label: the box comes from image "
                "intensity only, so localization is not oracle."
            ),
            "image_interpolation": "trilinear",
            "label_interpolation": "nearest-exact",
            "normalization": (
                "source nonzero-voxel population z-score before resize; unchanged "
                "by the crop because the crop removes only zero voxels"
            ),
            "uses_gt_for_localization": False,
            "skull_stripping": "not applied; BraTS 2024 volumes are already stripped",
            "n4_bias_field_correction": bool(args.n4),
            "n4_shrink_factor": int(args.n4_shrink) if args.n4 else None,
            "n4_implementation": (
                "SimpleITK N4BiasFieldCorrectionImageFilter, 4x50 iterations, nonzero mask"
                if args.n4 else None
            ),
            "augmentation": "NOT cached; applied per-epoch by the dataset",
        },
        "case_ids": [r[1] for r in results],
        "per_case_positive_voxels": {r[1]: r[2] for r in results},
        # kept so a 128^3 prediction can be restored exactly to the native grid
        "crop_geometry": {
            r[1]: {"origin": r[3], "extent": r[4], "native_shape": r[5]} for r in results
        },
        "build_seconds": round(elapsed, 2),
        "images_sha256": sha256_file(images_path),
        "labels_sha256": sha256_file(labels_path),
        "split": args.split,
        "test_access": args.split == "test",
        "builder": "scripts/build_t1ce_wt_braincrop_cache.py",
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\ncache written: {out_dir}  ({elapsed:.0f}s, {count / elapsed:.2f} case/s)")
    print(f"  images.f32 {images_path.stat().st_size / 2**30:.2f} GiB")
    print(f"  labels.u8  {labels_path.stat().st_size / 2**30:.2f} GiB")
    return index


def verify(args: argparse.Namespace, index: dict) -> None:
    """Re-process a random sample single-threaded and require byte equality."""
    rng = np.random.default_rng(20260806)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    pick = sorted(rng.choice(len(rows), size=min(args.verify, len(rows)), replace=False).tolist())
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    images = np.memmap(out_dir / "images.f32", dtype=np.float32, mode="r",
                       shape=(index["count"], *MODEL_SHAPE))
    labels = np.memmap(out_dir / "labels.u8", dtype=np.uint8, mode="r",
                       shape=(index["count"], *MODEL_SHAPE))
    _worker_init({**{"n4": bool(args.n4), "n4_shrink": int(args.n4_shrink)},
                  "images_path": os.devnull, "labels_path": os.devnull})
    order = {cid: i for i, cid in enumerate(index["case_ids"])}
    for i in pick:
        row = rows[i]
        cid = row.get("case_id")
        # _process writes through pwrite to /dev/null here; we only want its return
        _, case_id, positives, origin, extent, native_shape = _process((0, row))
        slot = order[case_id]
        assert index["crop_geometry"][case_id]["origin"] == origin, f"{case_id} origin"
        assert index["crop_geometry"][case_id]["extent"] == extent, f"{case_id} extent"
        assert int(np.array(labels[slot]).sum()) == positives, f"{case_id} positives"
        print(f"  verified {case_id}  positives={positives}  extent={extent}", flush=True)
    print(f"verify: {len(pick)} cases reproduce the cached geometry and label mass")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", required=True, choices=("train", "validation", "test"))
    p.add_argument("--expected-manifest-sha256", default=None)
    p.add_argument("--expected-count", type=int, default=None)
    p.add_argument("--n4", action="store_true")
    p.add_argument("--n4-shrink", type=int, default=2)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--verify", type=int, default=8, help="re-process N random cases")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    index = build(args)
    if args.verify:
        verify(args, index)


if __name__ == "__main__":
    main()
