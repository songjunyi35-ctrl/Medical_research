#!/usr/bin/env python3
"""Audit MU-Glioma-Post official XLSX metadata without optional dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any


CLINICAL_SHA256 = "4341badf954511d67fc52b82b3b488ed7323f7e46d249a354ff9b3a62269c79b"
VOLUMES_SHA256 = "4ba60e04279e91d920974f5641b313fd04fc2d0319f782f37567b48bc7795fa6"
MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
EXPECTED_VOLUME_SHEETS = {
    "Necrotic Tumor Core (Label1)": {"rows": 334, "unique_base": 146, "bare": 272, "suffixed": 62},
    "Tumor Infiltration and Edema": {"rows": 614, "unique_base": 202, "bare": 419, "suffixed": 195},
    "Enhancing Tumor Core (Label3)": {"rows": 569, "unique_base": 195, "bare": 408, "suffixed": 161},
    "Resection Cavity (Label4)": {"rows": 542, "unique_base": 189, "bare": 368, "suffixed": 174},
}


class MetadataAuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def column_index(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if match is None:
        raise MetadataAuditError(f"invalid cell reference: {reference}")
    value = 0
    for character in match.group():
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def read_xlsx(path: Path, expected_sha256: str) -> dict[str, list[dict[int, str]]]:
    if sha256_file(path) != expected_sha256:
        raise MetadataAuditError(f"official XLSX SHA-256 mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iter(MAIN_NS + "t")) for item in root.findall(MAIN_NS + "si")]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        result: dict[str, list[dict[int, str]]] = {}
        for sheet in workbook.find(MAIN_NS + "sheets") or []:
            name = sheet.attrib["name"]
            target = targets[sheet.attrib[REL_NS + "id"]]
            member = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
            root = ET.fromstring(archive.read(member))
            rows: list[dict[int, str]] = []
            for row in root.findall(".//" + MAIN_NS + "row"):
                values: dict[int, str] = {}
                for cell in row.findall(MAIN_NS + "c"):
                    node = cell.find(MAIN_NS + "v")
                    value = "" if node is None else node.text or ""
                    kind = cell.attrib.get("t")
                    if kind == "s" and value:
                        value = shared[int(value)]
                    elif kind == "inlineStr":
                        value = "".join(item.text or "" for item in cell.iter(MAIN_NS + "t"))
                    values[column_index(cell.attrib["r"])] = value
                rows.append(values)
            result[name] = rows
    return result


def audit(clinical_path: Path, volumes_path: Path) -> dict[str, Any]:
    clinical_book = read_xlsx(clinical_path, CLINICAL_SHA256)
    volumes_book = read_xlsx(volumes_path, VOLUMES_SHA256)
    if "MU Glioma Post" not in clinical_book or set(volumes_book) != set(EXPECTED_VOLUME_SHEETS):
        raise MetadataAuditError("official workbook sheet set changed")
    clinical = clinical_book["MU Glioma Post"]
    header = clinical[0]
    rows = clinical[1:]
    ids = [row.get(0, "").strip() for row in rows if row.get(0, "").strip()]
    timepoint_columns = [index for index, name in header.items() if "MRI (Timepoint_" in name]
    timepoints = sum(sum(row.get(index, "").strip() not in {"", "NA"} for index in timepoint_columns) for row in rows)
    if len(rows) != 335 or len(set(ids)) != 203 or len(timepoint_columns) != 6 or timepoints != 597:
        raise MetadataAuditError("clinical workbook counts changed")
    if any(re.search(r"BraTS|UCSD", value, re.IGNORECASE) for sheet in clinical_book.values() for row in sheet for value in row.values()):
        raise MetadataAuditError("unexpected cross-dataset identifier bridge appeared")
    volume_summary = {}
    for name, expected in EXPECTED_VOLUME_SHEETS.items():
        sheet_rows = volumes_book[name][1:]
        sheet_ids = [row.get(0, "").strip() for row in sheet_rows if row.get(0, "").strip()]
        bases = [match.group() for value in sheet_ids if (match := re.match(r"PatientID_\d{4}", value))]
        summary = {
            "rows": len(sheet_ids),
            "unique_exact_ids": len(set(sheet_ids)),
            "unique_patient_bases": len(set(bases)),
            "bare_ids": sum(bool(re.fullmatch(r"PatientID_\d{4}", value)) for value in sheet_ids),
            "post_treatment_suffixed_ids": sum("Post-treatment" in value for value in sheet_ids),
        }
        observed = {"rows": summary["rows"], "unique_base": summary["unique_patient_bases"], "bare": summary["bare_ids"], "suffixed": summary["post_treatment_suffixed_ids"]}
        if observed != expected:
            raise MetadataAuditError(f"volume sheet counts changed: {name}")
        volume_summary[name] = summary
    return {
        "schema": "aflow.brats.mu_glioma_post.metadata_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clinical": {"rows": len(rows), "unique_patients": len(set(ids)), "timepoint_columns": len(timepoint_columns), "nonempty_timepoint_cells": timepoints},
        "segmentation_volumes": volume_summary,
        "reported_imaging_timepoints": 596,
        "clinical_minus_reported_timepoints": timepoints - 596,
        "cross_dataset_identifier_bridge_found": False,
        "nifti_or_segmentation_voxel_access": False,
        "model_inference": False,
        "source": {
            "clinical_path": str(clinical_path.resolve()), "clinical_sha256": CLINICAL_SHA256,
            "volumes_path": str(volumes_path.resolve()), "volumes_sha256": VOLUMES_SHA256,
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--volumes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite metadata audit: {output}")
    value = audit(args.clinical.resolve(), args.volumes.resolve())
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"clinical": value["clinical"], "clinical_minus_reported_timepoints": value["clinical_minus_reported_timepoints"], "cross_dataset_identifier_bridge_found": False}, indent=2))


if __name__ == "__main__":
    main()
