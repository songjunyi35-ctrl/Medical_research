"""Rebuild the frozen UTSW primary-FeTS no-N4 input from two copies."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.build_no_n4_clean_cache import arrays as no_n4_arrays


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_no_n4_input_bridge"
SOURCE_RECORD = ROOT / "archive/research_records/2026-09-05_utsw_label_variant_bridge"
WORK = Path("/tmp/aflow-utsw-label-variant-20260905")
SUBJECTS = ("BT0001", "BT0002", "BT0003")
IMAGE = "brain_t1ce.nii.gz"
LABEL = "rtumorseg_manual_correction.nii.gz"
EXPECTED = {
    "idea": "25b79efca83b49fd9e5a725edbb75115031d19c621e0512c977b6150b70b0718",
    "prereg": "3015e099a066e2a36be99f49727ac554ea49ad16158a528d4ce137fbdfce65a2",
    "transfer": "f37fa324c413ba7a9c91db0f80b764501c29d1b926501c9f89237d2f96111c90",
    "source_raw": "0f31398be358b4c38d3c961adcf08b63eae4617002ea5b2814aa773b484ee80f",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_source_arrays(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, dict]:
    if image.ndim != 3 or label.shape != image.shape:
        raise ValueError("source image/label shape mismatch")
    if not np.isfinite(image).all() or float(image.min()) < 0:
        raise ValueError("primary FeTS image must be finite and nonnegative")
    zeros = int(np.count_nonzero(image == 0))
    positives = int(np.count_nonzero(image > 0))
    if zeros == 0 or positives == 0:
        raise ValueError("primary FeTS image must contain zero background and positive brain")
    if not np.isfinite(label).all() or not np.array_equal(label, np.rint(label)):
        raise ValueError("manual label must be finite and integer-exact")
    label_values = np.unique(label)
    if not np.any(label_values == 0) or not np.any(label_values > 0):
        raise ValueError("manual label must contain background and tumor")
    binary = np.ascontiguousarray(label > 0, dtype=np.uint8)
    return binary, {
        "image_min": float(image.min()), "image_max": float(image.max()),
        "image_zero_voxels": zeros, "image_positive_voxels": positives,
        "label_values": [float(value) for value in label_values.tolist()],
        "all_positive_labels_mapped_to_wt": True,
        "source_wt_voxels": int(binary.sum()),
    }


def prepare_case(root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    image_nifti = nib.as_closest_canonical(nib.load(str(root / subject / IMAGE), mmap=True))
    label_nifti = nib.as_closest_canonical(nib.load(str(root / subject / LABEL), mmap=True))
    affine_max_abs = float(np.max(np.abs(np.asarray(image_nifti.affine) - np.asarray(label_nifti.affine))))
    if image_nifti.shape != label_nifti.shape or affine_max_abs > 1e-6:
        raise ValueError(f"{subject} canonical geometry mismatch")
    image = np.ascontiguousarray(np.asanyarray(image_nifti.dataobj))
    label = np.ascontiguousarray(np.asanyarray(label_nifti.dataobj))
    binary, source = validate_source_arrays(image, label)
    output_image, output_label, geometry = no_n4_arrays(
        subject, image, binary, nib.affines.voxel_sizes(image_nifti.affine), use_n4=False
    )
    if not np.isfinite(output_image).all() or not bool(output_label.any()):
        raise RuntimeError("invalid 128-cube output")
    source.update({
        "canonical_shape": list(image_nifti.shape),
        "canonical_affine_max_abs": affine_max_abs,
        "source_image_sha256": sha_array(image),
        "source_binary_label_sha256": sha_array(binary),
        "output_image_sha256": sha_array(output_image),
        "output_label_sha256": sha_array(output_label),
    })
    return output_image, output_label, geometry, source


def main() -> None:
    dependencies = {
        "idea": RECORD / "IDEA.md", "prereg": RECORD / "PREREG.md",
        "transfer": SOURCE_RECORD / "TRANSFER_RECEIPT.json", "source_raw": SOURCE_RECORD / "RAW_QC.json",
    }
    actual = {name: sha_file(path) for name, path in dependencies.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen dependency hash mismatch: {actual}")
    destination = RECORD / "INPUT_QC.json"
    if destination.exists():
        raise FileExistsError(destination)
    copies = {"copy_a": WORK / "copy_a", "copy_b": WORK / "copy_b"}
    results = {name: {} for name in copies}
    arrays_by_copy = {name: {} for name in copies}
    for copy_name, copy_root in copies.items():
        for subject in SUBJECTS:
            image, label, geometry, source = prepare_case(copy_root, subject)
            arrays_by_copy[copy_name][subject] = (image, label, geometry)
            results[copy_name][subject] = source
    comparisons = {}
    for subject in SUBJECTS:
        image_a, label_a, geometry_a = arrays_by_copy["copy_a"][subject]
        image_b, label_b, geometry_b = arrays_by_copy["copy_b"][subject]
        comparisons[subject] = {
            "image_exact": bool(np.array_equal(image_a, image_b)),
            "label_exact": bool(np.array_equal(label_a, label_b)),
            "geometry_exact": geometry_a == geometry_b,
            "geometry": geometry_a,
        }
    passed = all(
        row["image_exact"] and row["label_exact"] and row["geometry_exact"]
        for row in comparisons.values()
    )
    report = {
        "schema": "aflow.utsw_primary_fets_no_n4_input_bridge.input_qc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "subjects": list(SUBJECTS), "image": IMAGE, "label": LABEL,
        "input_policy": "official primary FeTS image; ANTs forbidden; no N4; all positive labels form WT",
        "copies": results, "comparisons": comparisons,
        "dependency_sha256": actual,
        "image_voxels_accessed": True, "label_voxels_accessed": True,
        "model_inference": False, "old_cohort_accessed": False,
        "remaining_utsw_subjects_accessed": False,
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "cases_exact": sum(
            row["image_exact"] and row["label_exact"] and row["geometry_exact"] for row in comparisons.values()
        ), "expected": 3, "report_sha256": sha_file(destination),
    }, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("UTSW primary-FeTS no-N4 rebuild gate failed")


if __name__ == "__main__":
    main()
