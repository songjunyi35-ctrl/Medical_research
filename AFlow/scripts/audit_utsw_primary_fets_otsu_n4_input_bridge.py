"""Cross-process UTSW primary-FeTS Otsu-N4 input bridge."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from sammed3d import t1ce_binary
from sammed3d.n4_otsu_input import n4_otsu_correct
from scripts.audit_utsw_primary_fets_no_n4_input_bridge import validate_source_arrays
from scripts.build_no_n4_clean_cache import arrays as shared_arrays


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_otsu_n4_input_bridge"
SOURCE_RECORD = ROOT / "archive/research_records/2026-09-05_utsw_label_variant_bridge"
NO_N4_RECORD = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_no_n4_input_bridge"
CANONICAL_RECORD = ROOT / "archive/research_records/2026-09-05_utsw_primary_fets_canonical_n4_input_bridge"
WORK = Path("/tmp/aflow-utsw-label-variant-20260905")
SUBJECTS = ("BT0001", "BT0002", "BT0003")
IMAGE = "brain_t1ce.nii.gz"
LABEL = "rtumorseg_manual_correction.nii.gz"
EXPECTED = {
    "idea": "4c5b774337006c1b82d8dcf3774026fd45e41db4fca350a969c7eb9118d924f3",
    "prereg": "f4ea405820c81a02b5e0c0c8ac1d9639cb92262c27b1024461a4d6f2669cf3b7",
    "transfer": "f37fa324c413ba7a9c91db0f80b764501c29d1b926501c9f89237d2f96111c90",
    "source_raw": "0f31398be358b4c38d3c961adcf08b63eae4617002ea5b2814aa773b484ee80f",
    "no_n4": "7573bed6d03f194a429d7eacb4ffb3b711feeda15e91a47539aaf025020deb59",
    "canonical": "1cec4ba98fae05e54caca2a1eb0a14681758e79ed6ac35ec47b0a28a58353be0",
    "canonical_copy_a": "a1776d09d7e84a786e3fc27f114ec4119a6f02bf1cdb35102906a3e8876a3665",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def dependencies() -> dict[str, str]:
    paths = {
        "idea": RECORD / "IDEA.md", "prereg": RECORD / "PREREG.md",
        "transfer": SOURCE_RECORD / "TRANSFER_RECEIPT.json", "source_raw": SOURCE_RECORD / "RAW_QC.json",
        "no_n4": NO_N4_RECORD / "INPUT_QC.json", "canonical": CANONICAL_RECORD / "INPUT_QC.json",
        "canonical_copy_a": CANONICAL_RECORD / "COPY_A_QC.json",
    }
    actual = {name: sha_file(path) for name, path in paths.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen dependency hash mismatch: {actual}")
    return actual


ORIGINAL_CASE_ID = t1ce_binary._case_id


def external_case_id(value: object) -> str:
    if isinstance(value, str) and value in SUBJECTS:
        return value
    return ORIGINAL_CASE_ID(value)


def child(copy_name: str) -> None:
    dependencies()
    if copy_name not in ("copy_a", "copy_b"):
        raise ValueError("unknown frozen copy")
    destination = RECORD / (copy_name.upper() + "_QC.json")
    if destination.exists():
        raise FileExistsError(destination)
    t1ce_binary._case_id = external_case_id
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    cases = {}
    for subject in SUBJECTS:
        root = WORK / copy_name / subject
        image_nii = nib.as_closest_canonical(nib.load(str(root / IMAGE), mmap=True))
        label_nii = nib.as_closest_canonical(nib.load(str(root / LABEL), mmap=True))
        affine_delta = float(np.max(np.abs(np.asarray(image_nii.affine) - np.asarray(label_nii.affine))))
        if image_nii.shape != label_nii.shape or affine_delta > 1e-6:
            raise ValueError(f"{subject} source geometry mismatch")
        image = np.ascontiguousarray(np.asanyarray(image_nii.dataobj))
        label = np.ascontiguousarray(np.asanyarray(label_nii.dataobj))
        binary, source = validate_source_arrays(image, label)
        if not set(source["label_values"]) <= {0.0, 1.0, 2.0, 4.0}:
            raise ValueError(f"{subject} unexpected UTSW label domain")
        corrected, mask = n4_otsu_correct(image)
        output_image, output_label, geometry = shared_arrays(
            subject, corrected, binary, nib.affines.voxel_sizes(image_nii.affine), use_n4=False
        )
        if not np.isfinite(output_image).all() or not bool(output_label.any()):
            raise RuntimeError(f"{subject} invalid Otsu output")
        cases[subject] = {
            "source": source, "canonical_affine_max_abs": affine_delta,
            "output_image_sha256": sha_array(output_image), "output_label_sha256": sha_array(output_label),
            "geometry": geometry, "fit_mask": mask,
        }
    report = {
        "schema": "aflow.utsw_primary_fets_otsu_n4_input_bridge.child",
        "created_utc": datetime.now(timezone.utc).isoformat(), "copy": copy_name,
        "simpleitk_threads": 1, "cases": cases, "model_inference": False, "gpu_access": False,
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"copy": copy_name, "cases": len(cases), "report_sha256": sha_file(destination)}))


def comparison_row(a: dict, b: dict, no_n4: dict, canonical: dict) -> dict:
    return {
        "cross_process_image_exact": a["output_image_sha256"] == b["output_image_sha256"],
        "cross_process_label_exact": a["output_label_sha256"] == b["output_label_sha256"],
        "cross_process_geometry_exact": a["geometry"] == b["geometry"],
        "fit_mask_exact": a["fit_mask"] == b["fit_mask"],
        "label_exact_vs_no_n4": a["output_label_sha256"] == no_n4["output_label_sha256"],
        "geometry_exact_vs_no_n4": a["geometry"] == no_n4["geometry"],
        "image_different_vs_no_n4": a["output_image_sha256"] != no_n4["output_image_sha256"],
        "label_exact_vs_canonical": a["output_label_sha256"] == canonical["output_label_sha256"],
        "geometry_exact_vs_canonical": a["geometry"] == canonical["geometry"],
        "image_different_vs_canonical": a["output_image_sha256"] != canonical["output_image_sha256"],
    }


def formal() -> None:
    actual = dependencies()
    outputs = [RECORD / name for name in ("PREACCESS.json", "COPY_A_QC.json", "COPY_B_QC.json", "INPUT_QC.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("UTSW Otsu bridge already started")
    outputs[0].write_text(json.dumps({
        "schema": "aflow.utsw_primary_fets_otsu_n4_input_bridge.preaccess",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_otsu_n4",
        "dependency_sha256": actual, "model_inference": False, "old_cohort_accessed": False,
    }, indent=2) + "\n")
    for copy_name in ("copy_a", "copy_b"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "child", "--copy", copy_name], cwd=ROOT, check=True)
    a = json.loads((RECORD / "COPY_A_QC.json").read_text())
    b = json.loads((RECORD / "COPY_B_QC.json").read_text())
    no_raw = json.loads((NO_N4_RECORD / "INPUT_QC.json").read_text())
    canonical = json.loads((CANONICAL_RECORD / "COPY_A_QC.json").read_text())
    comparisons = {}
    for subject in SUBJECTS:
        no = dict(no_raw["copies"]["copy_a"][subject]); no["geometry"] = no_raw["comparisons"][subject]["geometry"]
        comparisons[subject] = comparison_row(a["cases"][subject], b["cases"][subject], no, canonical["cases"][subject])
    passed = all(all(row.values()) for row in comparisons.values())
    report = {
        "schema": "aflow.utsw_primary_fets_otsu_n4_input_bridge.input_qc",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if passed else "failed",
        "subjects": list(SUBJECTS), "comparisons": comparisons,
        "copy_a_qc_sha256": sha_file(RECORD / "COPY_A_QC.json"),
        "copy_b_qc_sha256": sha_file(RECORD / "COPY_B_QC.json"),
        "no_n4_qc_sha256": EXPECTED["no_n4"], "canonical_qc_sha256": EXPECTED["canonical"],
        "simpleitk_threads_per_process": 1, "model_inference": False, "gpu_access": False,
        "old_cohort_accessed": False, "remaining_utsw_subjects_accessed": False,
    }
    (RECORD / "INPUT_QC.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(comparisons), "report_sha256": sha_file(RECORD / "INPUT_QC.json")}, indent=2))
    if not passed:
        raise RuntimeError("UTSW Otsu-N4 fixed gates failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("formal", "child")); parser.add_argument("--copy", choices=("copy_a", "copy_b"))
    args = parser.parse_args()
    if args.mode == "child":
        if args.copy is None: parser.error("child requires --copy")
        child(args.copy)
    else:
        if args.copy is not None: parser.error("formal forbids --copy")
        formal()


if __name__ == "__main__":
    main()
