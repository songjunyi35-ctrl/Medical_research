#!/usr/bin/env python3
"""Verify the frozen WAW152 MRI archive and selected T1c files before caching."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-06_waw_gliomai152_frozen_panel"
REVIEW = ROOT / "archive/reviews/2026-09-06_waw_gliomai_source_audit"
SOURCE_ROOT = ROOT / "workspace/BraTS/waw_gliomai_source_20260906"
ARCHIVE = SOURCE_ROOT / "2_1_preop_scans_coreg.zip"
SELECTED_ROOT = SOURCE_ROOT / "selected_t1c"
MASK_ROOT = SOURCE_ROOT / "metadata/preop_masks"
IDEA_SHA = "5ccf61e564d321cc32fd4c92d193f1802020aae0d1c55e189d3067d0387fb0db"
PREREG_SHA = "5dd8156a7740b1b115e7e54ca20e727a47e7913360015bd7033ff9a58e721448"
AUDIT_SHA = "62503b3e680f0fc6922f20d6732a608d226429357bb8e5bb52da67e7ade7d3e7"
COHORT_SHA = "25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a"
ARCHIVE_BYTES = 3164137716
ARCHIVE_MD5 = "0d24a3a2c76bc6fa9b5b6468679676b1"
ARCHIVE_SHA256 = "680a90c8bad679dad5f2bbf8dfdd7efd992a635de6f9b3f7a91422349f2b5887"


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> None:
    if hash_file(path) != expected:
        raise ValueError(f"SHA mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RECORD / "SOURCE_RECEIPT.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checked(RECORD / "IDEA.md", IDEA_SHA)
    checked(RECORD / "PREREG.md", PREREG_SHA)
    checked(REVIEW / "SOURCE_AUDIT.json", AUDIT_SHA)
    checked(REVIEW / "FROZEN_COHORT.json", COHORT_SHA)
    if ARCHIVE.stat().st_size != ARCHIVE_BYTES or hash_file(ARCHIVE, "md5") != ARCHIVE_MD5:
        raise ValueError("official archive size/MD5 mismatch")
    checked(ARCHIVE, ARCHIVE_SHA256)

    cohort = json.loads((REVIEW / "FROZEN_COHORT.json").read_text())
    if cohort["case_count"] != 152 or cohort["subject_count"] != 152:
        raise ValueError("cohort count mismatch")
    expected_basenames = {row["image_basename"] for row in cohort["cases"]}
    with ZipFile(ARCHIVE) as archive:
        archive.testzip_result = archive.testzip()
        if archive.testzip_result is not None:
            raise ValueError(f"zip CRC failure: {archive.testzip_result}")
        all_members = [name for name in archive.namelist() if not name.endswith("/")]
        t1c_members = [name for name in all_members if name.endswith("_brain_t1c.nii.gz")]
    if len(all_members) != 609 or len(t1c_members) != 152:
        raise ValueError("zip member count mismatch")
    if {Path(name).name for name in t1c_members} != expected_basenames:
        raise ValueError("zip T1c members differ from frozen cohort")

    selected_paths = sorted(SELECTED_ROOT.glob("**/*_brain_t1c.nii.gz"))
    if len(selected_paths) != 152 or {path.name for path in selected_paths} != expected_basenames:
        raise ValueError("selected extraction differs from frozen cohort")
    selected_by_name = {path.name: path for path in selected_paths}
    mask_by_name = {path.name: path for path in MASK_ROOT.glob("**/*.nii.gz")}
    files = []
    global_labels = set()
    for row in cohort["cases"]:
        image_path = selected_by_name[row["image_basename"]]
        mask_name = Path(row["mask_member"]).name
        mask_path = mask_by_name[mask_name]
        checked(mask_path, row["mask_sha256"])
        image = nib.as_closest_canonical(nib.load(str(image_path), mmap=True))
        mask = nib.as_closest_canonical(nib.load(str(mask_path), mmap=True))
        affine_error = float(np.max(np.abs(image.affine - mask.affine)))
        if image.shape != mask.shape or affine_error > 1e-5:
            raise ValueError(f"native geometry mismatch: {row['case_id']}")
        image_array = np.asanyarray(image.dataobj)
        mask_array = np.asanyarray(mask.dataobj)
        rounded = np.rint(mask_array)
        labels = sorted(int(value) for value in np.unique(rounded))
        global_labels.update(labels)
        if (
            not np.isfinite(image_array).all()
            or not (image_array > 0).any()
            or not np.isfinite(mask_array).all()
            or not np.allclose(mask_array, rounded, rtol=0, atol=1e-5)
            or not set(labels).issubset({0, 1, 2, 3})
            or not np.any(rounded > 0)
        ):
            raise ValueError(f"array/domain failure: {row['case_id']} {labels}")
        files.append(
            {
                "case_id": row["case_id"],
                "subject_id": row["subject_id"],
                "image_path": str(image_path),
                "image_bytes": image_path.stat().st_size,
                "image_sha256": hash_file(image_path),
                "mask_path": str(mask_path),
                "mask_sha256": row["mask_sha256"],
                "native_shape": list(image.shape),
                "spacing_mm": [float(value) for value in nib.affines.voxel_sizes(image.affine)],
                "axis_codes": list(nib.aff2axcodes(image.affine)),
                "affine_error": affine_error,
                "image_dtype": str(image.get_data_dtype()),
                "mask_dtype": str(mask.get_data_dtype()),
                "raw_labels": labels,
                "raw_wt_voxels": int(np.count_nonzero(rounded > 0)),
            }
        )
    if global_labels != {0, 1, 2, 3}:
        raise ValueError("global label union mismatch")

    receipt = {
        "schema": "aflow.waw_gliomai152.source_receipt",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "source_preaccess_sha256": hash_file(RECORD / "SOURCE_PREACCESS.json"),
        "archive": {
            "path": str(ARCHIVE),
            "bytes": ARCHIVE_BYTES,
            "official_md5": ARCHIVE_MD5,
            "sha256": ARCHIVE_SHA256,
            "zip_crc_all_passed": True,
            "file_members": len(all_members),
            "t1c_members": len(t1c_members),
        },
        "selected": {
            "cases": len(files),
            "subjects": len({row["subject_id"] for row in files}),
            "image_bytes": sum(row["image_bytes"] for row in files),
            "all_native_geometry_exact": True,
            "all_finite_and_positive_support": True,
            "all_masks_integer_nonempty": True,
            "global_raw_label_union": sorted(global_labels),
            "files": files,
        },
        "cohort_sha256": COHORT_SHA,
        "checkpoint_loaded": False,
        "model_forwards": 0,
        "performance_metrics": False,
        "verifier_sha256": hash_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": len(files),
                "selected_image_bytes": receipt["selected"]["image_bytes"],
                "source_receipt_sha256": hash_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
