#!/usr/bin/env python3
"""Audit the small WAW-GliomAI metadata/mask release before MRI access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np


OFFICIAL_MD5 = {
    "1_0_clinical_molecular_data.csv": "7dd4f66797b768853898e77de76cda44",
    "1_1_clinical_molecular_metadata_dictionary.csv": "f1908b5913c2e3472ab9e24286a1a38d",
    "2_1_preop_scans_coreg_meta.csv": "39b5d0e2ff1a3f522f80ab4842fa511f",
    "3_1_preop_masks.zip": "cca9e9da6d7cc4a06e0c31bebb86b592",
    "5_technical_validation.zip": "d0294ed8767df6d51b9b52e15f21c1dd",
    "6_reproducibility_code.zip": "4e9ac5cfa3dd569f72b48bead346bad0",
}
IMAGE_ARCHIVE = {
    "name": "2_1_preop_scans_coreg.zip",
    "bytes": 3164137716,
    "md5": "0d24a3a2c76bc6fa9b5b6468679676b1",
    "url": "https://zenodo.org/api/records/22205631/files/2_1_preop_scans_coreg.zip/content",
}


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output_dir / "SOURCE_AUDIT.json"
    cohort_path = args.output_dir / "FROZEN_COHORT.json"
    if audit_path.exists() or cohort_path.exists():
        raise FileExistsError("audit outputs already exist")

    source_hashes = {}
    for name, expected_md5 in OFFICIAL_MD5.items():
        path = args.metadata_dir / name
        actual_md5 = hash_file(path, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(f"official MD5 mismatch: {name}")
        source_hashes[name] = {
            "bytes": path.stat().st_size,
            "official_md5": expected_md5,
            "sha256": hash_file(path, "sha256"),
        }

    meta = read_csv(args.metadata_dir / "2_1_preop_scans_coreg_meta.csv")
    by_exam: dict[str, dict[str, dict[str, str]]] = {}
    for row in meta:
        basename = Path(row["file_name"]).name
        stem = basename.removesuffix(".nii.gz")
        exam = stem.split("_brain_")[0]
        modality = stem.split("_brain_")[1]
        by_exam.setdefault(exam, {})[modality] = row
    if len(meta) != 608 or len(by_exam) != 152:
        raise ValueError("unexpected final metadata counts")
    for exam, modalities in by_exam.items():
        if set(modalities) != {"t1c", "t1n", "t2f", "t2w"}:
            raise ValueError(f"incomplete modalities: {exam}")

    mask_zip = args.metadata_dir / "3_1_preop_masks.zip"
    with ZipFile(mask_zip) as archive:
        mask_members = sorted(name for name in archive.namelist() if name.endswith(".nii.gz"))
    extracted_root = args.metadata_dir / "preop_masks"
    extracted_masks = sorted(extracted_root.glob("**/*.nii.gz"))
    if len(mask_members) != 152 or len(extracted_masks) != 152:
        raise ValueError("unexpected mask count")

    clinical = {row["id"]: row for row in read_csv(args.metadata_dir / "1_0_clinical_molecular_data.csv", ";")}
    cases = []
    label_case_counts: Counter[int] = Counter()
    shape_counts: Counter[str] = Counter()
    nonzero_counts = []
    for mask_path in extracted_masks:
        mask_name = mask_path.name
        exam = mask_name.split("_tumor_mask_pre")[0]
        if exam not in by_exam:
            raise ValueError(f"mask lacks metadata exam: {exam}")
        patient_id, exam_number = exam.split("_", 1)
        if patient_id not in clinical or clinical[patient_id]["mri_preop"] != "1":
            raise ValueError(f"mask lacks preop clinical row: {exam}")
        image = nib.load(mask_path)
        array = np.asanyarray(image.dataobj)
        labels = sorted(int(value) for value in np.unique(array))
        if not set(labels).issubset({0, 1, 2, 3}) or not any(labels):
            raise ValueError(f"invalid preop label domain: {exam} {labels}")
        if not np.allclose(image.header.get_zooms()[:3], (1.0, 1.0, 1.0), rtol=0, atol=1e-6):
            raise ValueError(f"mask spacing mismatch: {exam}")
        meta_affine = np.asarray(json.loads(by_exam[exam]["t1c"]["affine"]), dtype=float)
        if array.shape != tuple(json.loads(by_exam[exam]["t1c"]["shape"])) or not np.allclose(
            image.affine, meta_affine, rtol=0, atol=1e-6
        ):
            raise ValueError(f"mask-to-T1c metadata geometry mismatch: {exam}")
        for label in labels:
            label_case_counts[label] += 1
        shape_counts["x".join(map(str, array.shape))] += 1
        nonzero = int(np.count_nonzero(array))
        if nonzero == 0:
            raise ValueError(f"empty mask: {exam}")
        nonzero_counts.append(nonzero)
        cases.append(
            {
                "case_id": f"WAW-{exam}",
                "subject_id": f"WAW-{patient_id}",
                "publisher_patient_id": patient_id,
                "exam_number": exam_number,
                "image_basename": by_exam[exam]["t1c"]["file_name"],
                "mask_member": next(member for member in mask_members if Path(member).name == mask_name),
                "mask_sha256": hash_file(mask_path, "sha256"),
                "mask_labels": labels,
                "mask_positive_voxels": nonzero,
                "shape": list(array.shape),
                "voxel_sizes_mm": [1.0, 1.0, 1.0],
                "scanner_name": by_exam[exam]["t1c"]["ScannerName"],
                "field_strength": by_exam[exam]["t1c"]["FieldStrengthLabel"],
                "who_grade": int(clinical[patient_id]["who_grade"]),
            }
        )
    cases.sort(key=lambda row: int(row["publisher_patient_id"]))
    if len({row["subject_id"] for row in cases}) != 152:
        raise ValueError("subjects are not unique")
    if set(label_case_counts) != {0, 1, 2, 3}:
        raise ValueError("global label union mismatch")

    cohort = {
        "schema": "aflow.waw_gliomai.preop_frozen_cohort",
        "schema_version": 1,
        "status": "frozen_before_coreg_image_archive_access",
        "selection": "exact intersection of final-release preop coreg T1c metadata and preop masks",
        "cases": cases,
        "case_count": len(cases),
        "subject_count": len({row["subject_id"] for row in cases}),
        "target": "all positive preoperative labels {1,2,3}; excludes postoperative label4 by cohort construction",
    }
    cohort_payload = json.dumps(cohort, indent=2, sort_keys=True) + "\n"
    cohort_path.write_text(cohort_payload)

    validation_summary = read_csv(
        args.metadata_dir / "technical_validation/5_interobserver_reproducibility_summary.csv"
    )
    audit = {
        "schema": "aflow.waw_gliomai.source_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "zenodo": {
            "record": 22205631,
            "concept_record": 15621811,
            "doi": "10.5281/zenodo.22205631",
            "publication_date": "2026-09-01",
            "access_right": "open",
            "license": "CC BY 4.0",
            "publisher": "Scientific Data",
            "source_institution": "Medical University of Warsaw",
        },
        "release_description_vs_final_files": {
            "described_preoperative_exams": 156,
            "final_coreg_t1c_exams": len(by_exam),
            "final_preop_masks": len(extracted_masks),
            "strict_pairs": len(cases),
            "interpretation": "use the 152-case final-file intersection; do not infer or seek four missing exams",
        },
        "source_files": source_hashes,
        "future_image_archive_contract": IMAGE_ARCHIVE,
        "cohort": {
            "cases": len(cases),
            "subjects": len({row["subject_id"] for row in cases}),
            "all_exam_number_zero": all(row["exam_number"] == "0" for row in cases),
            "all_positive": all(row["mask_positive_voxels"] > 0 for row in cases),
            "global_label_union": sorted(label_case_counts),
            "label_case_counts": {str(key): value for key, value in sorted(label_case_counts.items())},
            "shape_counts": dict(shape_counts),
            "mask_positive_voxels": {
                "min": min(nonzero_counts),
                "median": float(np.median(nonzero_counts)),
                "max": max(nonzero_counts),
            },
            "who_grade_counts": dict(sorted(Counter(row["who_grade"] for row in cases).items())),
            "scanner_counts": dict(sorted(Counter(row["scanner_name"] or "missing" for row in cases).items())),
            "field_strength_counts": dict(sorted(Counter(row["field_strength"] or "missing" for row in cases).items())),
        },
        "annotation": {
            "initial_method": "FL-PoST/FeTS workflow",
            "final_method": "reviewed and manually refined by radiologists with neuro-oncologic imaging expertise",
            "preop_interobserver_cases": 15,
            "technical_validation_rows": len(validation_summary),
            "technical_validation_zip_sha256": source_hashes["5_technical_validation.zip"]["sha256"],
        },
        "source_disjointness": {
            "brats2024_training_sites": [
                "Duke",
                "UCSF",
                "Missouri",
                "UCSD",
                "Heidelberg",
                "Michigan",
                "Indiana",
            ],
            "institution_disjoint": True,
            "patient_id_bridge_available": False,
            "claim": "institution-disjoint/no-known-overlap; not patient-ID-proven disjoint",
        },
        "model_or_large_image_access": {
            "coreg_image_archive_downloaded": False,
            "checkpoint_loaded": False,
            "model_forwards": 0,
            "performance_metrics": False,
        },
        "cohort_path": str(cohort_path),
        "cohort_sha256": hash_file(cohort_path, "sha256"),
        "analyzer_sha256": hash_file(Path(__file__), "sha256"),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "cohort_sha256": audit["cohort_sha256"],
                "audit_sha256": hash_file(audit_path, "sha256"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
