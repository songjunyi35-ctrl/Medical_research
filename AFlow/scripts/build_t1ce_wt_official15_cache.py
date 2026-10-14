#!/usr/bin/env python
"""Build the official SAM-Med3D 1.5 mm T1ce/WT development cache.

The task contract stays T1ce single-channel and WT={1,2,3}.  Only the image
geometry and deterministic intensity pipeline are aligned to SAM-Med3D:

1. convert the selected target to one binary LabelMap;
2. resample the paired subject to 1.5 mm isotropic (linear image, nearest label);
3. convert to canonical orientation;
4. label-centred CropOrPad to 128^3;
5. compute mean/sample-std from positive image voxels and normalize the whole
   patch, matching TorchIO ZNormalization(masking_method=lambda x: x > 0).

BraTS images are already registered and skull-stripped.  N4 is deliberately
absent because it is not part of the official SAM-Med3D conversion script.
Only train/validation manifests are accepted; sealed test data cannot be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sammed3d.t1ce_wt_prompt import wt_target  # noqa: E402
from sammed3d.t1ce_wt_training import positive_sample_zscore, validate_row  # noqa: E402

MODEL_SHAPE = (128, 128, 128)
TARGET_SPACING = (1.5, 1.5, 1.5)
CACHE_SCHEMA = "aflow.sammed3d.t1ce_wt_preprocess_cache"
CACHE_SCHEMA_VERSION = 1
PROTOCOL_SCHEMA = "aflow.sammed3d.t1ce_wt_official15_cache"

_CFG: dict[str, object] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_init(cfg: dict[str, object]) -> None:
    _CFG.update(cfg)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        import SimpleITK as sitk

        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        pass
    import torch

    torch.set_num_threads(1)
    _CFG["images_fd"] = os.open(str(_CFG["images_path"]), os.O_WRONLY)
    _CFG["labels_fd"] = os.open(str(_CFG["labels_path"]), os.O_WRONLY)


def preprocess_row(row: dict) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply the public SAM-Med3D conversion and patch pipeline to one case."""
    import torch
    import torchio as tio

    validate_row(row)
    case_id = str(row["case_id"])
    image_path = Path(str(row["image_path"]))
    label_path = Path(str(row["label_path"]))
    image_nii = nib.load(str(image_path), mmap=True)
    label_nii = nib.load(str(label_path), mmap=True)
    if image_nii.shape != label_nii.shape:
        raise ValueError(f"{case_id} image/label shape mismatch")
    if not np.allclose(image_nii.affine, label_nii.affine, rtol=0.0, atol=1e-5):
        raise ValueError(f"{case_id} image/label affine mismatch")

    image = np.ascontiguousarray(np.asanyarray(image_nii.dataobj), dtype=np.float32)
    raw_label = np.asanyarray(label_nii.dataobj)
    target = wt_target(raw_label)
    if not bool(target.any()):
        raise RuntimeError(f"{case_id} has empty WT foreground")

    subject = tio.Subject(
        image=tio.ScalarImage(
            tensor=torch.from_numpy(image)[None],
            affine=np.asarray(image_nii.affine, dtype=np.float64),
        ),
        label=tio.LabelMap(
            tensor=target[None].to(torch.uint8),
            affine=np.asarray(label_nii.affine, dtype=np.float64),
        ),
    )

    # This order matches utils/infer_utils.py::data_preprocess.  LabelMap is
    # automatically resampled with nearest-neighbour interpolation by TorchIO.
    resampled = tio.Resample(
        target=TARGET_SPACING,
        image_interpolation="linear",
    )(subject)
    canonical = tio.ToCanonical()(resampled)
    resampled_target = canonical.label.data > 0
    before_crop = int(resampled_target.sum())
    if before_crop == 0:
        raise RuntimeError(f"{case_id} WT vanished after 1.5 mm resampling")

    cropped = tio.CropOrPad(
        mask_name="label",
        target_shape=MODEL_SHAPE,
    )(canonical)
    image_patch = positive_sample_zscore(cropped.image.data.float().contiguous())
    label_patch = (cropped.label.data > 0).to(torch.uint8).contiguous()
    after_crop = int(label_patch.sum())
    if after_crop == 0:
        raise RuntimeError(f"{case_id} WT vanished after CropOrPad")
    if tuple(image_patch.shape) != (1, *MODEL_SHAPE):
        raise RuntimeError(f"{case_id} unexpected image patch shape")
    if tuple(label_patch.shape) != (1, *MODEL_SHAPE):
        raise RuntimeError(f"{case_id} unexpected label patch shape")
    if not bool(torch.isfinite(image_patch).all()):
        raise RuntimeError(f"{case_id} produced non-finite normalized voxels")

    actual_spacing = tuple(
        float(value) for value in nib.affines.voxel_sizes(canonical.image.affine)
    )
    if not np.allclose(actual_spacing, TARGET_SPACING, rtol=0.0, atol=1e-5):
        raise RuntimeError(f"{case_id} resampled spacing {actual_spacing} != {TARGET_SPACING}")

    positive = image_patch[image_patch > 0]
    qc = {
        "case_id": case_id,
        "source_shape": list(image_nii.shape),
        "source_spacing": [
            float(value) for value in nib.affines.voxel_sizes(image_nii.affine)
        ],
        "resampled_shape": list(canonical.spatial_shape),
        "resampled_spacing": list(actual_spacing),
        "wt_voxels_before_resample": int(target.sum()),
        "wt_voxels_after_resample_before_crop": before_crop,
        "wt_voxels_after_crop": after_crop,
        "wt_crop_retention": after_crop / before_crop,
        "normalized_patch_mean": float(image_patch.mean()),
        "normalized_patch_min": float(image_patch.min()),
        "normalized_patch_max": float(image_patch.max()),
        "normalized_positive_voxels": int(positive.numel()),
    }
    return image_patch.numpy(), label_patch.numpy(), qc


def _process(task: tuple[int, dict]) -> tuple[int, dict[str, object]]:
    index, row = task
    image, label, qc = preprocess_row(row)
    voxels = int(np.prod(MODEL_SHAPE))
    image_bytes = np.ascontiguousarray(image[0], dtype=np.float32).tobytes()
    label_bytes = np.ascontiguousarray(label[0], dtype=np.uint8).tobytes()
    if len(image_bytes) != voxels * 4 or len(label_bytes) != voxels:
        raise RuntimeError(f"{qc['case_id']} unexpected serialized size")
    os.pwrite(int(_CFG["images_fd"]), image_bytes, index * voxels * 4)
    os.pwrite(int(_CFG["labels_fd"]), label_bytes, index * voxels)
    return index, qc


def _read_rows(manifest: Path) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("manifest is empty")
    for row in rows:
        if row.get("schema") != "aflow.brats2024.sammed3d_t1ce_wt_inner_split":
            raise ValueError("only the frozen inner train/validation T1ce manifest is allowed")
        validate_row(row)
    return rows


def build(args: argparse.Namespace) -> dict[str, object]:
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    manifest = manifest.resolve()
    rows = _read_rows(manifest)
    expected_split = args.split
    for row in rows:
        if row.get("split") != expected_split:
            raise ValueError(
                f"{row['case_id']} manifest split {row.get('split')!r} != {expected_split!r}"
            )

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty cache: {output}")
    output.mkdir(parents=True, exist_ok=True)

    count = len(rows)
    images_path = output / "images.f32"
    labels_path = output / "labels.u8"
    np.memmap(
        images_path,
        dtype=np.float32,
        mode="w+",
        shape=(count, *MODEL_SHAPE),
    ).flush()
    np.memmap(
        labels_path,
        dtype=np.uint8,
        mode="w+",
        shape=(count, *MODEL_SHAPE),
    ).flush()

    cfg: dict[str, object] = {
        "images_path": str(images_path),
        "labels_path": str(labels_path),
    }
    started = time.time()
    records: list[tuple[int, dict[str, object]]] = []
    with mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(cfg,),
    ) as pool:
        for completed, record in enumerate(
            pool.imap_unordered(_process, enumerate(rows), chunksize=1),
            start=1,
        ):
            records.append(record)
            if completed % 50 == 0 or completed == count:
                elapsed = max(time.time() - started, 1e-9)
                rate = completed / elapsed
                print(
                    f"{completed}/{count} {rate:.2f} case/s "
                    f"eta={(count - completed) / max(rate, 1e-9):.0f}s",
                    flush=True,
                )

    records.sort(key=lambda item: item[0])
    qc_rows = [record for _, record in records]
    elapsed = time.time() - started
    retention = np.asarray([float(row["wt_crop_retention"]) for row in qc_rows])
    index: dict[str, object] = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "protocol_schema": PROTOCOL_SCHEMA,
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest),
        "count": count,
        "model_shape": list(MODEL_SHAPE),
        "image_dtype": "float32",
        "label_dtype": "uint8",
        "preprocessing": {
            "target": "T1ce single-channel / WT={1,2,3}; RC={4} excluded",
            "target_spacing_mm": list(TARGET_SPACING),
            "order": [
                "WT123 binary LabelMap",
                "TorchIO Resample 1.5mm isotropic",
                "TorchIO ToCanonical",
                "TorchIO CropOrPad(mask_name=label, target_shape=128^3)",
                "positive-voxel sample z-score applied to whole patch",
            ],
            "spatial": "gt_crop_1p5mm",
            "image_interpolation": "linear",
            "label_interpolation": "nearest (TorchIO LabelMap)",
            "normalization": "post-crop positive-voxel sample z-score",
            "uses_gt_for_localization": True,
            "skull_stripping": "not applied; BraTS volumes are already stripped",
            "n4_bias_field_correction": False,
            "augmentation": "NOT cached; official RandomFlip applied by train dataset",
            "source": "uni-medical/SAM-Med3D prepare_data_from_nnUNet.py + infer_utils.py",
        },
        "case_ids": [str(row["case_id"]) for row in qc_rows],
        "per_case_positive_voxels": {
            str(row["case_id"]): int(row["wt_voxels_after_crop"]) for row in qc_rows
        },
        "qc": {
            "all_source_spacing_mm": sorted(
                {tuple(row["source_spacing"]) for row in qc_rows}
            ),
            "all_resampled_spacing_mm": sorted(
                {tuple(row["resampled_spacing"]) for row in qc_rows}
            ),
            "crop_retention_min": float(retention.min()),
            "crop_retention_mean": float(retention.mean()),
            "cases_with_crop_loss": int((retention < 1.0).sum()),
            "cases_with_more_than_1pct_crop_loss": int((retention < 0.99).sum()),
            "per_case": qc_rows,
        },
        "build_seconds": round(elapsed, 2),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "images_sha256": sha256_file(images_path),
        "labels_sha256": sha256_file(labels_path),
        "split": expected_split,
        "test_access": False,
    }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        f"cache written: {output} ({elapsed:.1f}s) "
        f"retention_min={retention.min():.6f}",
        flush=True,
    )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    build(parse_args())
