#!/usr/bin/env python3
"""Validate RHUH-GBM patient metadata without accessing MRI or segmentations."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


CLINICAL_SHA256 = "32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728"
GDC_MANIFEST_SHA256 = "5a70dc03e33ac651d1afc020e505351142b15228ad0583f9f12b72a19e4acd49"
PATIENT_RE = re.compile(r"^RHUH-\d{4}$")


class RhuhMetadataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise RhuhMetadataError(f"official CSV SHA-256 mismatch: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(clinical_path: Path, gdc_manifest_path: Path) -> dict:
    clinical = _read(clinical_path, CLINICAL_SHA256)
    gdc = _read(gdc_manifest_path, GDC_MANIFEST_SHA256)
    clinical_ids = [row.get("Patient ID", "").strip() for row in clinical]
    gdc_ids = [row.get("Participant ID", "").strip() for row in gdc]
    if len(clinical) != 40 or len(set(clinical_ids)) != 40 or any(PATIENT_RE.fullmatch(value) is None for value in clinical_ids):
        raise RhuhMetadataError("clinical patient contract changed")
    counts = Counter(gdc_ids)
    if len(gdc) != 600 or len(counts) != 40 or set(counts.values()) != {15}:
        raise RhuhMetadataError("GDC 3-timepoint x 5-modality series contract changed")
    if set(clinical_ids) != set(gdc_ids):
        raise RhuhMetadataError("clinical/GDC patient ID sets differ")
    if {row.get("File Type") for row in gdc} != {"DICOM"}:
        raise RhuhMetadataError("GDC manifest unexpectedly contains non-DICOM data")
    serialized = json.dumps({"clinical": clinical, "gdc": gdc})
    if re.search(r"BraTS-GLI|UCSF-|UCSD-PTGBM", serialized, re.IGNORECASE):
        raise RhuhMetadataError("unexpected cross-dataset ID bridge appeared")
    return {
        "schema": "aflow.brats.rhuh_gbm.metadata_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clinical_rows": 40,
        "clinical_unique_patients": 40,
        "gdc_dicom_series_rows": 600,
        "gdc_unique_patients": 40,
        "gdc_series_per_patient": 15,
        "patient_id_sets_equal": True,
        "patient_id_min": min(clinical_ids),
        "patient_id_max": max(clinical_ids),
        "cross_dataset_identifier_bridge_found": False,
        "public_nifti_or_segmentation_inventory": False,
        "nifti_or_segmentation_voxel_access": False,
        "model_inference": False,
        "sources": {
            "clinical_path": str(clinical_path.resolve()), "clinical_sha256": CLINICAL_SHA256,
            "gdc_manifest_path": str(gdc_manifest_path.resolve()), "gdc_manifest_sha256": GDC_MANIFEST_SHA256,
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--gdc-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    value = audit(args.clinical.resolve(), args.gdc_manifest.resolve())
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({key: value[key] for key in ("clinical_unique_patients", "gdc_dicom_series_rows", "gdc_series_per_patient", "cross_dataset_identifier_bridge_found", "public_nifti_or_segmentation_inventory")}, indent=2))


if __name__ == "__main__":
    main()
