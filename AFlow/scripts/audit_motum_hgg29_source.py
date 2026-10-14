#!/usr/bin/env python3
"""Audit the official MOTUM snapshot and freeze its 29 HGG subjects."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspace/BraTS/motum_source_20260906"
ARCHIVES = [SOURCE / "10.12751_g-node.tvzqc5.zip", SOURCE / "10.12751_g-node.tvzqc5.copy_b.zip"]
SELECTED = SOURCE / "selected_hgg29"
METADATA = SOURCE / "metadata"
ARCHIVE_BYTES = 984434960
ARCHIVE_SHA = "2626862599a3fcfe4ac0cefcea3af5b190625275036cc8eb4c9039cbd54e2d7c"
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_xlsx(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.findall(".//x:t", NS)) for item in shared_root.findall("x:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    raw_rows = []
    for row in sheet.findall(".//x:sheetData/x:row", NS):
        values = {}
        for cell in row.findall("x:c", NS):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            node = cell.find("x:v", NS)
            value = "" if node is None else str(node.text)
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values[column] = value
        raw_rows.append(values)
    headers = raw_rows[0]
    return [{headers[column]: row.get(column, "") for column in headers} for row in raw_rows[1:]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.output_dir / "SOURCE_AUDIT.json"
    cohort_path = args.output_dir / "FROZEN_HGG29_COHORT.json"
    if audit_path.exists() or cohort_path.exists():
        raise FileExistsError("source audit outputs exist")
    for archive in ARCHIVES:
        if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA:
            raise ValueError("dual archive mismatch")
        with ZipFile(archive) as handle:
            if handle.testzip() is not None:
                raise ValueError("zip CRC failure")

    participants_path = METADATA / "Participants.xlsx"
    participants = read_xlsx(participants_path)
    if len(participants) != 67 or len({row["ID"] for row in participants}) != 67:
        raise ValueError("participant table")
    hgg_rows = [row for row in participants if row["WHO CNS Grade"] in {"3", "4"} and row["Origin"] == "/"]
    if len(hgg_rows) != 29 or [row["ID"] for row in hgg_rows] != [f"sub-{index:04d}" for index in range(1, 30)]:
        raise ValueError("HGG selection")
    if any(row["Pathology diagnosis"] not in {"Oligodendroglioma", "Astrocytoma", "Glioblastoma"} for row in hgg_rows):
        raise ValueError("HGG pathology")

    cases = []
    label_patterns = Counter()
    shape_patterns = Counter()
    spacing_patterns = Counter()
    axis_patterns = Counter()
    wt_image_positive_fraction = []
    for row in hgg_rows:
        subject = row["ID"]
        image_path = SELECTED / subject / "anat" / f"{subject}_t1ce.nii.gz"
        flair_mask_path = SELECTED / "derivatives" / subject / "flair_seg_label1.nii.gz"
        t1ce_mask_path = SELECTED / "derivatives" / subject / "t1ce_seg_label2.nii.gz"
        image = nib.load(image_path)
        flair_mask = nib.load(flair_mask_path)
        t1ce_mask = nib.load(t1ce_mask_path)
        affine_errors = [
            float(np.max(np.abs(image.affine - flair_mask.affine))),
            float(np.max(np.abs(image.affine - t1ce_mask.affine))),
        ]
        if image.shape != flair_mask.shape or image.shape != t1ce_mask.shape or max(affine_errors) > 1e-5:
            raise ValueError(f"geometry: {subject}")
        image_array = np.asanyarray(image.dataobj)
        flair_array = np.asanyarray(flair_mask.dataobj)
        t1ce_array = np.asanyarray(t1ce_mask.dataobj)
        flair_values = sorted(int(value) for value in np.unique(flair_array))
        t1ce_values = sorted(int(value) for value in np.unique(t1ce_array))
        if not set(flair_values).issubset({0, 1}) or not set(t1ce_values).issubset({0, 1, 2}):
            raise ValueError(f"mask domain: {subject}")
        wt = (flair_array > 0) | (t1ce_array > 0)
        if not np.isfinite(image_array).all() or not (image_array > 0).any() or not wt.any():
            raise ValueError(f"array support: {subject}")
        positive_fraction = float(np.mean(image_array[wt] > 0))
        wt_image_positive_fraction.append(positive_fraction)
        label_patterns[(tuple(flair_values), tuple(t1ce_values))] += 1
        shape_patterns["x".join(map(str, image.shape))] += 1
        spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
        spacing_patterns[str(tuple(round(value, 6) for value in spacing))] += 1
        axis_patterns[str(nib.aff2axcodes(image.affine))] += 1
        cases.append(
            {
                "case_id": f"MOTUM-HGG-{subject.removeprefix('sub-')}",
                "subject_id": subject,
                "who_grade": int(row["WHO CNS Grade"]),
                "pathology_diagnosis": row["Pathology diagnosis"],
                "image_quality_t1ce": int(float(row["Image rating: T1-ce"])),
                "image_path": str(image_path),
                "image_bytes": image_path.stat().st_size,
                "image_sha256": sha256(image_path),
                "flair_mask_path": str(flair_mask_path),
                "flair_mask_sha256": sha256(flair_mask_path),
                "t1ce_mask_path": str(t1ce_mask_path),
                "t1ce_mask_sha256": sha256(t1ce_mask_path),
                "native_shape": list(image.shape),
                "spacing_mm": list(spacing),
                "axis_codes": list(nib.aff2axcodes(image.affine)),
                "affine_max_abs": max(affine_errors),
                "flair_mask_values": flair_values,
                "t1ce_mask_values": t1ce_values,
                "flair_positive_voxels": int(np.count_nonzero(flair_array > 0)),
                "t1ce_positive_voxels": int(np.count_nonzero(t1ce_array > 0)),
                "mask_overlap_voxels": int(np.count_nonzero((flair_array > 0) & (t1ce_array > 0))),
                "wt_union_voxels": int(np.count_nonzero(wt)),
                "wt_voxels_with_positive_t1ce_fraction": positive_fraction,
            }
        )
    if min(wt_image_positive_fraction) < 0.95:
        raise ValueError("WT largely outside positive T1ce support")

    cohort = {
        "schema": "aflow.motum.hgg29.frozen_cohort",
        "schema_version": 1,
        "status": "frozen_after_source_arrays_before_model_or_cache",
        "selection": "Participants.xlsx WHO CNS Grade in {3,4} and Origin='/' before performance; excludes all 38 metastases",
        "target": "WT is positive union of semantic flair_seg_label1 and t1ce_seg_label2 files; values inside each file are not reinterpreted",
        "case_count": len(cases),
        "subject_count": len(cases),
        "cases": cases,
    }
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema": "aflow.motum.hgg29.source_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_snapshot": {
            "doi": "10.12751/g-node.tvzqc5",
            "published": "2024-04-30",
            "license": "CC BY 4.0",
            "bytes": ARCHIVE_BYTES,
            "publisher_checksum_available": False,
            "dual_transfer_sha256": ARCHIVE_SHA,
            "dual_transfer_byte_exact": True,
            "zip_crc_both_passed": True,
        },
        "participants": {
            "total": len(participants),
            "origin_counts": dict(Counter(row["Origin"] for row in participants)),
            "hgg29_grade_counts": dict(Counter(row["WHO CNS Grade"] for row in hgg_rows)),
            "hgg29_pathology_counts": dict(Counter(row["Pathology diagnosis"] for row in hgg_rows)),
            "metastases_excluded_before_model": 38,
            "participants_xlsx_sha256": sha256(participants_path),
        },
        "hgg29_arrays": {
            "cases": len(cases),
            "selected_files": len(cases) * 3,
            "selected_bytes": sum(row["image_bytes"] for row in cases)
            + sum(Path(row["flair_mask_path"]).stat().st_size + Path(row["t1ce_mask_path"]).stat().st_size for row in cases),
            "native_geometry_exact": True,
            "affine_max_abs": max(row["affine_max_abs"] for row in cases),
            "shape_counts": dict(shape_patterns),
            "spacing_counts": dict(spacing_patterns),
            "axis_counts": dict(axis_patterns),
            "label_value_patterns": {str(key): value for key, value in label_patterns.items()},
            "wt_union_nonempty": True,
            "wt_positive_t1ce_fraction_min_median_max": [
                min(wt_image_positive_fraction),
                float(np.median(wt_image_positive_fraction)),
                max(wt_image_positive_fraction),
            ],
        },
        "annotation": {
            "regions": ["non-contrast-enhancing FLAIR signal abnormality", "contrast-enhancing tumor"],
            "method": "semi-automated segmentations carefully validated and corrected by experienced physicians",
            "t1ce_label_value_anomaly": "one semantic t1ce_seg_label2 file uses positive value1 rather than2; target is positive-by-file-semantics",
        },
        "source_disjointness": {
            "collection_sites": [
                "Second Affiliated Hospital of Anhui Medical University",
                "Changzheng Hospital",
                "First Affiliated Hospital of USTC",
            ],
            "institution_disjoint_from_brats2024_seven_sites": True,
            "patient_id_bridge_available": False,
            "claim": "institution-disjoint/no-known-overlap; not patient-ID-proven disjoint",
        },
        "grid_caveat": "processed clinical volumes have only 21 or 24 slices and heterogeneous in-plane shapes/spacings; any 128^3 evaluation strongly rasterizes through-plane information",
        "model_or_cache": {"cache": False, "checkpoint_loaded": False, "model_forwards": 0, "performance_metrics": False},
        "cohort_sha256": sha256(cohort_path),
        "analyzer_sha256": sha256(Path(__file__)),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(cases), "cohort_sha256": audit["cohort_sha256"], "audit_sha256": sha256(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
