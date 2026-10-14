#!/usr/bin/env python3
"""Freeze the public 95-glioma BraTS-Africa cohort before payload access."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/reviews/2026-09-06_brats_africa_source_audit"
INVENTORY = RECORD / "OFFICIAL_INVENTORY.json"
EXPECTED_INVENTORY_SHA256 = "16f6eee4fe258e9e75a69a2bda75c7f3f5a00365ab833308bf1e1943fbcee45b"
OUTPUT = RECORD / "FROZEN_GLIOMA95_COHORT.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INVENTORY) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("official inventory SHA mismatch")
    inventory = json.loads(INVENTORY.read_text())
    selected = [row for row in inventory["files"] if "/95_Glioma/" in row["path"]]
    excluded = [row for row in inventory["files"] if "/51_OtherNeoplasms/" in row["path"]]
    if len(selected) != 475 or len(excluded) != 255:
        raise ValueError("publisher category cardinality mismatch")
    pattern = re.compile(r"^(BraTS-SSA-\d{5}-000)-(?P<kind>t1c|seg)\.nii\.gz$")
    cases: dict[str, dict] = {}
    for row in selected:
        match = pattern.match(row["basename"])
        if not match:
            continue
        case = match.group(1)
        kind = match.group("kind")
        if kind in cases.setdefault(case, {}):
            raise ValueError(f"duplicate {kind}: {case}")
        cases[case][kind] = {"path": row["path"], "bytes": int(row["size"]), "type": row["type"], "target_type": row["target_type"]}
    if len(cases) != 95 or any(set(value) != {"t1c", "seg"} for value in cases.values()):
        raise ValueError("95-case T1CE/seg pairing failed")
    rows = [{"case_id": case, "subject_id": case.rsplit("-", 1)[0], "t1ce": cases[case]["t1c"], "seg": cases[case]["seg"]} for case in sorted(cases)]
    if len({row["subject_id"] for row in rows}) != 95:
        raise ValueError("subjects are not one-to-one with cases")
    result = {
        "schema": "aflow.brats_africa.glioma95_frozen_cohort",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_package_payload_nifti_or_label_access",
        "official": {"collection_doi": "10.7937/V8H6-8X67", "package_id": 948, "inventory_sha256": EXPECTED_INVENTORY_SHA256, "license": "CC BY 4.0"},
        "selection": {"publisher_directory": "95_Glioma", "cases": 95, "subjects": 95, "files": 190, "bytes": sum(row["t1ce"]["bytes"] + row["seg"]["bytes"] for row in rows), "case_ids_sha256": hashlib.sha256(("\n".join(row["case_id"] for row in rows) + "\n").encode()).hexdigest(), "rows": rows},
        "excluded": {"publisher_directory": "51_OtherNeoplasms", "cases": 51, "files": 255, "reason": "not the adult diffuse glioma task; exclusion fixed by publisher category before pixels/models"},
        "evidence_role": "public post-challenge external cohort; original 2023 train60/validation15/test20 labels are now all released; not hidden blind",
        "institution": "six Nigerian diagnostic centers; institution-disjoint from the seven named BraTS2024 post-treatment contributors; no cross-dataset patient ID bridge",
        "access": {"package_payload": False, "nifti_arrays": False, "label_values": False, "model_inference": False, "gpu": False, "old_cohort": False},
        "freezer": str(Path(__file__).resolve().relative_to(ROOT)),
        "freezer_sha256": sha(Path(__file__).resolve())
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "cases": 95, "subjects": 95, "files": 190, "bytes": result["selection"]["bytes"], "case_ids_sha256": result["selection"]["case_ids_sha256"], "output_sha256": sha(OUTPUT)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
