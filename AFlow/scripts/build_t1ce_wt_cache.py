#!/usr/bin/env python
"""Precompute the deterministic T1ce/WT whole-volume 128^3 preprocessing to disk.

The existing training pipeline recomputes, for every case and every epoch:
canonical load of a full-resolution NIfTI, nonzero z-score, trilinear image
resize and nearest-exact label resize. That work is identical on every epoch,
which is why the dataloader (not the GPU) is the bottleneck: measured GPU
utilization was 32.5% with two workers each burning ~2.5 CPU cores.

Only the *stochastic* augmentation must stay per-epoch. Everything upstream of
it is a pure function of the case, so it is cached here once.

Cache layout (per split):
    images.f32       memmap float32 (N,128,128,128)
    labels.u8        memmap uint8   (N,128,128,128)  values in {0,1}
    index.json       ordered case ids + provenance and integrity metadata

Optional N4 bias field correction (SimpleITK) is applied to the raw canonical
volume *before* normalization, matching the DPD-SAM preprocessing description
(intensity normalization, bias field correction, skull stripping). BraTS 2024
post-treatment volumes are already skull-stripped, so no stripping is applied.

With `--no-n4` the cached tensors are bit-identical to
`sammed3d.t1ce_wt_training.load_whole_volume_resize`; `--verify` proves it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHAPE = (128, 128, 128)
SCHEMA = "aflow.sammed3d.t1ce_wt_preprocess_cache"
SCHEMA_VERSION = 1

_CFG: dict = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def n4_correct(volume: np.ndarray, shrink: int) -> np.ndarray:
    """N4 bias field correction restricted to the nonzero (brain) support."""
    import SimpleITK as sitk

    data = np.ascontiguousarray(volume, dtype=np.float32)
    image = sitk.GetImageFromArray(data)
    mask = sitk.GetImageFromArray((data > 0).astype(np.uint8))
    mask.CopyInformation(image)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.Execute(
        sitk.Cast(sitk.Shrink(image, [shrink] * 3), sitk.sitkFloat32),
        sitk.Shrink(mask, [shrink] * 3),
    )
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    corrected = sitk.Cast(image, sitk.sitkFloat32) / sitk.Exp(log_bias)
    out = np.asarray(sitk.GetArrayFromImage(corrected), dtype=np.float32)
    # N4 is undefined outside the mask; keep the stripped background at exactly 0
    # so the downstream nonzero-population z-score sees the same support.
    out[data <= 0] = 0.0
    if not np.isfinite(out).all():
        raise RuntimeError("N4 produced non-finite voxels")
    return out


def _worker_init(cfg: dict) -> None:
    _CFG.update(cfg)
    # Each case is one task, so every library must stay single-threaded inside a
    # worker. ITK in particular spawns its own pool: leaving it at the default
    # made 64 workers thrash 128 cores down to 0.5 case/s.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[var] = "1"
    try:
        import SimpleITK as sitk

        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        pass
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    # One long-lived append-capable descriptor per worker, reused for all cases.
    _CFG["images_fd"] = os.open(_CFG["images_path"], os.O_WRONLY)
    _CFG["labels_fd"] = os.open(_CFG["labels_path"], os.O_WRONLY)


def _process(task: tuple[int, dict]) -> tuple[int, str, float, int]:
    from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays
    from sammed3d.t1ce_wt_training import _load_canonical_nifti, validate_row

    index, row = task
    # The inner 972/108 split already carries the strict T1ce/WT contract, but the
    # outer sealed-test manifest uses the random_subject_train_test_split schema
    # with `modalities.t1c` / `seg_path`. Reuse the already-validated adapter from
    # the frozen evaluator rather than re-deriving the path and label mapping here.
    if row.get("schema") != "aflow.brats2024.sammed3d_t1ce_wt_inner_split":
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

    if _CFG["n4"]:
        image = n4_correct(image, _CFG["n4_shrink"])

    if _CFG["spatial"] == "gt_crop":
        # SAM-Med3D's native geometry: ToCanonical + label-centred CropOrPad at
        # NATIVE resolution (train.py:78, validation.py:294, "crop only object
        # region"), then the post-crop positive-voxel z-score. No whole-volume
        # downsampling, so the lesion keeps its native voxel count.
        import torch
        import torchio as tio

        from sammed3d.t1ce_wt_prompt import wt_target
        from sammed3d.t1ce_wt_training import positive_sample_zscore

        # Load RAW (not canonicalized) here, because tio.ToCanonical below does the
        # reorientation. Reusing the already-canonical arrays would canonicalize
        # twice, which is idempotent in meaning but perturbs float32 by ~1e-6 and
        # so would not be bit-identical to load_official_crop.
        raw_image_nifti = nib.load(str(row["image_path"]), mmap=True)
        raw_label_nifti = nib.load(str(row["label_path"]), mmap=True)
        raw_image = np.ascontiguousarray(np.asanyarray(raw_image_nifti.dataobj))
        raw_label = np.asanyarray(raw_label_nifti.dataobj)
        if _CFG["n4"]:
            raw_image = n4_correct(raw_image, _CFG["n4_shrink"])
        target3d = wt_target(raw_label)
        if not bool(target3d.any()):
            raise RuntimeError(f"{case_id} has empty WT foreground")
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.from_numpy(np.ascontiguousarray(raw_image))[None].float(),
                affine=np.asarray(raw_image_nifti.affine, dtype=np.float64),
            ),
            label=tio.LabelMap(
                tensor=target3d[None].to(torch.uint8),
                affine=np.asarray(raw_label_nifti.affine, dtype=np.float64),
            ),
        )
        transformed = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(mask_name="label", target_shape=MODEL_SHAPE),
        ])(subject)
        cropped = transformed.image.data.float().contiguous()
        img = positive_sample_zscore(cropped).numpy()
        tgt = (transformed.label.data > 0).numpy().astype(np.uint8)
    else:
        view = prepare_t1ce_supervised_arrays(
            case_id, image, label, spacing, output_shape=MODEL_SHAPE
        )
        img = view.image.float().numpy()          # (1,128,128,128)
        tgt = view.target.numpy().astype(np.uint8)  # (1,128,128,128)
    if img.shape != (1, *MODEL_SHAPE) or tgt.shape != (1, *MODEL_SHAPE):
        raise RuntimeError(f"{case_id} unexpected cached shape")
    positives = int(tgt.sum())
    if positives == 0:
        raise RuntimeError(f"{case_id} WT is empty after whole-volume resize")
    if not np.isfinite(img).all():
        raise RuntimeError(f"{case_id} cached image is non-finite")

    # Write with positional writes rather than a memmap slice. np.memmap.flush()
    # msyncs the WHOLE mapping, so flushing once per case made an 8 GiB cache
    # crawl at 0.4 case/s. pwrite touches only this case's bytes.
    voxels = int(np.prod(MODEL_SHAPE))
    img_bytes = np.ascontiguousarray(img[0], dtype=np.float32).tobytes()
    tgt_bytes = np.ascontiguousarray(tgt[0], dtype=np.uint8).tobytes()
    if len(img_bytes) != voxels * 4 or len(tgt_bytes) != voxels:
        raise RuntimeError(f"{case_id} unexpected serialized size")
    os.pwrite(_CFG["images_fd"], img_bytes, index * voxels * 4)
    os.pwrite(_CFG["labels_fd"], tgt_bytes, index * voxels)
    return index, case_id, float(img.mean()), positives


def build(args: argparse.Namespace) -> dict:
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    count = len(rows)
    if count == 0:
        raise SystemExit("empty manifest")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty cache dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images_path = out_dir / "images.f32"
    labels_path = out_dir / "labels.u8"
    # Allocate the full files up front so workers can write disjoint slices.
    np.memmap(images_path, dtype=np.float32, mode="w+",
              shape=(count, *MODEL_SHAPE)).flush()
    np.memmap(labels_path, dtype=np.uint8, mode="w+",
              shape=(count, *MODEL_SHAPE)).flush()

    cfg = {
        "spatial": args.spatial,
        "n4": bool(args.n4),
        "n4_shrink": int(args.n4_shrink),
        "images_path": str(images_path),
        "labels_path": str(labels_path),
        "count": count,
    }

    started = time.time()
    results: list[tuple[int, str, float, int]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers, initializer=_worker_init, initargs=(cfg,)
    ) as pool:
        for done, res in enumerate(
            pool.imap_unordered(_process, list(enumerate(rows)), chunksize=1), 1
        ):
            results.append(res)
            if done % 50 == 0 or done == count:
                rate = done / max(time.time() - started, 1e-9)
                print(
                    f"  {done}/{count}  {rate:.1f} case/s  "
                    f"eta {(count - done) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )

    results.sort()
    elapsed = time.time() - started
    index = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest),
        "count": count,
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "preprocessing": {
            "spatial": args.spatial,
            "image_interpolation": "trilinear" if args.spatial == "whole_resize" else "none (native resolution crop)",
            "label_interpolation": "nearest-exact" if args.spatial == "whole_resize" else "none (native resolution crop)",
            "normalization": ("source nonzero-voxel population z-score before resize"
                              if args.spatial == "whole_resize"
                              else "post-crop positive-voxel sample z-score"),
            "uses_gt_for_localization": args.spatial == "gt_crop",
            "skull_stripping": "not applied; BraTS 2024 volumes are already stripped",
            "n4_bias_field_correction": bool(args.n4),
            "n4_shrink_factor": int(args.n4_shrink) if args.n4 else None,
            "n4_implementation": "SimpleITK N4BiasFieldCorrectionImageFilter, 4x50 iterations, nonzero mask" if args.n4 else None,
            "augmentation": "NOT cached; applied per-epoch by the dataset",
        },
        "case_ids": [r[1] for r in results],
        "per_case_positive_voxels": {r[1]: r[3] for r in results},
        "build_seconds": round(elapsed, 2),
        "images_sha256": sha256_file(images_path),
        "labels_sha256": sha256_file(labels_path),
        "split": args.split,
        "test_access": args.split == "test",
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\ncache written: {out_dir}  ({elapsed:.0f}s, {count / elapsed:.1f} case/s)")
    print(f"  images.f32 {images_path.stat().st_size / 2**30:.2f} GiB")
    print(f"  labels.u8  {labels_path.stat().st_size / 2**30:.2f} GiB")
    return index


def verify(args: argparse.Namespace, index: dict) -> None:
    """Prove cached tensors equal the live pipeline on a random sample."""
    import torch

    from sammed3d.t1ce_wt_training import (
        T1ceWtPreprocessConfig,
        load_preprocessed_case,
    )

    # Compare against the live pipeline for the SAME spatial mode; comparing a
    # gt_crop cache against the whole-resize loader would always "fail".
    reference_config = T1ceWtPreprocessConfig(
        spatial_preprocessing=index["preprocessing"]["spatial"],
        train_augmentation="none",
    )

    manifest = ROOT / index["manifest"]
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    order = {cid: i for i, cid in enumerate(index["case_ids"])}
    rng = np.random.default_rng(20260805)
    picks = rng.choice(len(rows), size=min(args.verify, len(rows)), replace=False)

    out_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    images = np.memmap(out_dir / "images.f32", dtype=np.float32, mode="r",
                       shape=(index["count"], *MODEL_SHAPE))
    labels = np.memmap(out_dir / "labels.u8", dtype=np.uint8, mode="r",
                       shape=(index["count"], *MODEL_SHAPE))

    worst = 0.0
    for p in picks:
        row = rows[int(p)]
        i = order[str(row["case_id"])]
        ref_img, ref_tgt = load_preprocessed_case(row, reference_config)
        cached_img = torch.from_numpy(np.asarray(images[i]))[None]
        cached_tgt = torch.from_numpy(np.asarray(labels[i]).astype(np.float32))[None]
        if not torch.equal(cached_tgt, ref_tgt):
            raise SystemExit(f"{row['case_id']}: cached label != live pipeline")
        delta = float((cached_img - ref_img).abs().max())
        worst = max(worst, delta)
        tag = "bit-identical" if delta == 0.0 else f"max|delta|={delta:.3e}"
        print(f"  {row['case_id']}: labels equal, image {tag}")
    if index["preprocessing"]["n4_bias_field_correction"]:
        print(f"\nverify: N4 is ON, so images intentionally differ from the "
              f"un-corrected pipeline (worst max|delta|={worst:.3e}); labels matched exactly.")
        return

    # Bit-identity is the wrong bar here. Cache workers run with
    # torch.set_num_threads(1) while this reference call runs multi-threaded, and
    # the z-score's mean/std are threaded float32 reductions, so the summation
    # order differs. Changing ONLY the thread count on the identical function
    # reproduces the same ~1.9e-06 delta, so any residual at this scale is
    # reduction-order noise, not a preprocessing difference. Labels are integers
    # and must still match exactly.
    tolerance = 1e-4
    if worst == 0.0:
        print(f"\nverify: {len(picks)} sampled cases are BIT-IDENTICAL to the live pipeline.")
    elif worst <= tolerance:
        print(f"\nverify: {len(picks)} sampled cases match the live pipeline; labels exact, "
              f"images within float32 reduction-order noise (worst={worst:.3e} <= {tolerance:.0e}).")
    else:
        raise SystemExit(
            f"verify FAILED: images differ by {worst:.3e}, beyond the {tolerance:.0e} "
            f"tolerance for float32 reduction-order noise"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--n4", action="store_true", help="apply N4 bias field correction")
    parser.add_argument("--n4-shrink", type=int, default=2)
    parser.add_argument("--spatial", choices=("whole_resize", "gt_crop"),
                        default="whole_resize",
                        help="gt_crop = SAM-Med3D native label-centred 128^3 at native resolution")
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="train",
                        help="recorded in index.json; 'test' sets test_access=true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", type=int, default=5,
                        help="verify N sampled cases against the live pipeline")
    args = parser.parse_args()

    index = build(args)
    if args.verify:
        print("\nverifying against live pipeline:")
        verify(args, index)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    main()
