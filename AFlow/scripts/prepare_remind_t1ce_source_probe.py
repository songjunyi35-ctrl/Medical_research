"""Freeze ReMIND43 metadata cohort and the first-case format probe."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-05_remind_t1ce_source_probe"
SOURCE_RECORD = ROOT / "archive/reviews/2026-09-05_remind_source_audit"
SOURCE = SOURCE_RECORD / "REMIND_PUBLIC_SOURCE.json"
SUMS = SOURCE_RECORD / "source_metadata/ReMIND_NRRD_Segs_September_2023.sums"
CLINICAL = SOURCE_RECORD / "source_metadata/ReMIND-Dataset-Clinical-Data-September-2023.xlsx"
IDEA_SHA = "bc8cd0c5651e4c5ab22204603717252e70a321b640ac2784a5ee469b199269b1"
PREREG_SHA = "b861cb7365878a66c5230b45dc68598ca99d9b57f938374302ad84da65d01064"
SOURCE_SHA = "b2f281023dda7ba2fc95767a00f7af67e66a945f158e8f5b1c9bb9ccce2f2b9a"
SUMS_SHA = "6daa7391546fca3d4a9bc985241c8f499726f9b3116617efcc89d53f2ece5116"
CLINICAL_SHA = "5c416ec4a4aa40247feee993af8af017e73ec3455fd05bab8c30ee03627343e8"
DIAGNOSES = {"Glioblastoma", "Astrocytoma", "Oligodendroglioma"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_clinical(path: Path) -> list[dict[str, str]]:
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.iter("{%s}t" % namespace["m"]))
                  for item in shared_root.findall("m:si", namespace)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    table = []
    for row in sheet.findall(".//m:sheetData/m:row", namespace):
        values = {}
        for cell in row.findall("m:c", namespace):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            node = cell.find("m:v", namespace)
            value = "" if node is None else node.text
            if cell.attrib.get("t") == "s" and value != "":
                value = shared[int(value)]
            values[column] = value
        table.append(values)
    headers = table[0]
    return [{headers[column]: row.get(column, "") for column in headers} for row in table[1:]]


def is_t1_tumor_path(path: str) -> bool:
    return bool(re.search(r"/ReMIND-\d{3}-preop-SEG-tumor-MR-.*T1_postcontrast(?:-r)?\.nrrd$", path, re.I))


def build_candidates() -> list[dict]:
    source = json.loads(SOURCE.read_text())
    series = source["series_api"]["preop_t1_postcontrast"]["series"]
    series_by_id = {row["PatientID"]: row for row in series}
    if len(series_by_id) != len(series) or len(series) != 105:
        raise ValueError("ReMIND preop T1 postcontrast is not unique/105")
    checksums = {}
    for line in SUMS.read_text().splitlines():
        md5, path = line.split(maxsplit=1)
        checksums[path.lstrip("*")] = md5
    tumor_by_id = {}
    cerebrum_by_id = {}
    for path, md5 in checksums.items():
        match = re.search(r"ReMIND-(\d{3})", path, re.I)
        if not match:
            continue
        patient = "ReMIND-" + match.group(1)
        if is_t1_tumor_path(path):
            if patient in tumor_by_id:
                raise ValueError(f"multiple T1-postcontrast manual tumor labels: {patient}")
            tumor_by_id[patient] = {"official_path": "/" + path, "official_md5": md5}
        elif re.search(r"-preop-SEG-cerebrum-MR-.*T1_postcontrast(?:-r)?\.nrrd$", path, re.I):
            if patient in cerebrum_by_id:
                raise ValueError(f"multiple T1-postcontrast cerebrum labels: {patient}")
            cerebrum_by_id[patient] = {"official_path": "/" + path, "official_md5": md5}
    browsed = {row["path"]: row for row in source["segmentation_package"]["files"]}
    clinical = {"ReMIND-" + str(row["Case Number"]).zfill(3): row for row in read_clinical(CLINICAL)}
    if len(clinical) != 114:
        raise ValueError("ReMIND clinical table is not 114 unique cases")
    candidates = []
    for patient in sorted(set(series_by_id) & set(tumor_by_id)):
        row = clinical[patient]
        if row["Histopathology"] not in DIAGNOSES:
            continue
        tumor = dict(tumor_by_id[patient])
        tumor["bytes"] = browsed.get(tumor["official_path"], {}).get("size")
        cerebrum = None
        if patient in cerebrum_by_id:
            cerebrum = dict(cerebrum_by_id[patient])
            cerebrum["bytes"] = browsed.get(cerebrum["official_path"], {}).get("size")
        candidates.append({
            "case_id": patient,
            "subject_id": patient,
            "histopathology": row["Histopathology"],
            "who_grade": row["WHO Grade"],
            "enhancement": row["Enhancement"],
            "previous_craniotomy": row["Previous Craniotomy"],
            "scanner": row["Preoperative MRI scanner"],
            "tesla": row["Tesla"],
            "t1ce": series_by_id[patient],
            "manual_tumor": tumor,
            "automatic_cerebrum": cerebrum,
        })
    return candidates


def main() -> None:
    for path, expected in ((RECORD / "IDEA.md", IDEA_SHA), (RECORD / "PREREG.md", PREREG_SHA),
                           (SOURCE, SOURCE_SHA), (SUMS, SUMS_SHA), (CLINICAL, CLINICAL_SHA)):
        if sha(path) != expected:
            raise ValueError(f"frozen ReMIND source changed: {path}")
    candidates = build_candidates()
    if len(candidates) != 43 or candidates[0]["case_id"] != "ReMIND-002":
        raise ValueError("ReMIND candidate count/order differs from preregistration")
    if Counter(row["histopathology"] for row in candidates) != {
        "Glioblastoma": 19, "Astrocytoma": 15, "Oligodendroglioma": 9
    }:
        raise ValueError("ReMIND diagnosis counts changed")
    cohort = {
        "schema": "aflow.remind_t1ce_manual_tumor.candidates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "metadata_frozen_before_any_candidate_pixels",
        "count": 43,
        "subjects": 43,
        "selection": "exact histopathology in Glioblastoma/Astrocytoma/Oligodendroglioma; unique preop T1_postcontrast; unique manual tumor NRRD referencing T1_postcontrast",
        "diagnosis_counts": dict(Counter(row["histopathology"] for row in candidates)),
        "enhancement_counts": dict(Counter(row["enhancement"] for row in candidates)),
        "provided_t1ce_cerebrum_masks": sum(row["automatic_cerebrum"] is not None for row in candidates),
        "candidates": candidates,
        "source_sha256": SOURCE_SHA,
        "sums_sha256": SUMS_SHA,
        "clinical_sha256": CLINICAL_SHA,
        "institution": "Brigham and Women's Hospital; not one of the seven named BraTS2024 contributing sites",
        "patient_identity_caveat": "different institution/namespace but no cross-dataset patient ID bridge; pixel-level overlap audit remains required",
        "pixel_access": False,
        "segmentation_voxel_access": False,
        "model_inference": False,
        "runner_sha256": sha(Path(__file__)),
    }
    cohort_path = RECORD / "COHORT_CANDIDATES.json"
    with cohort_path.open("x") as handle:
        json.dump(cohort, handle, indent=2, sort_keys=True)
        handle.write("\n")
    probe = candidates[0]
    if probe["automatic_cerebrum"] is None:
        raise ValueError("fixed ReMIND-002 lacks T1ce cerebrum mask")
    preaccess = {
        "schema": "aflow.remind_t1ce_source_probe.preaccess",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_single_probe_download",
        "probe": probe,
        "cohort_candidates_sha256": sha(cohort_path),
        "authorized_download": "only ReMIND-002 preop T1_postcontrast DICOM series plus its T1ce-referenced manual tumor and automatic cerebrum NRRDs",
        "geometry_tolerance": 1e-5,
        "model_inference_authorized": False,
        "access": {"candidate_dicom_pixels": False, "manual_tumor_voxels": False,
                   "automatic_cerebrum_voxels": False, "other_candidate_pixels": False,
                   "checkpoint": False, "model_forwards": False, "metrics": False},
        "idea_sha256": IDEA_SHA,
        "prereg_sha256": PREREG_SHA,
        "runner_sha256": sha(Path(__file__)),
    }
    with (RECORD / "PROBE_PREACCESS.json").open("x") as handle:
        json.dump(preaccess, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"count": 43, "diagnoses": cohort["diagnosis_counts"],
                      "provided_cerebrum": cohort["provided_t1ce_cerebrum_masks"],
                      "probe": probe, "cohort_sha256": preaccess["cohort_candidates_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
