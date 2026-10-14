"""Frozen paths and hashes for the BraTS-Africa95 baseline panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-06_brats_africa95_frozen_baselines"
SOURCE_REVIEW = ROOT / "archive/reviews/2026-09-06_brats_africa_source_audit"
COHORT = SOURCE_REVIEW / "FROZEN_GLIOMA95_COHORT.json"
INVENTORY = SOURCE_REVIEW / "OFFICIAL_INVENTORY.json"
SOURCE = Path("/tmp/aflow-brats-africa95-source-20260906")
CACHE = Path("/tmp/aflow-brats-africa95-cache-20260906")
SOURCE_PREACCESS = RECORD / "SOURCE_PREACCESS.json"
SOURCE_DOWNLOAD = RECORD / "SOURCE_DOWNLOAD.json"
FIRST_PROBE = RECORD / "FIRST_CASE_PROBE_V2.json"
CACHE_RECEIPT = RECORD / "CACHE_RECEIPT.json"
PREACCESS = RECORD / "MODEL_PREACCESS.json"

PREREGISTERED_IDEA_SHA = "20d7a9d778097e224f7063aced0c17f2be1592e6384030a01b7b5202b107c262"
IDEA_SHA = "9557f759a6fce29785c202eaf82c4979a1cdb2abfd88803c2f9c2a7023c173dc"
PREREG_SHA = "ced348b400a1283dfb33fdc54315a2abdaaf0c291a614277580eff168b580b79"
COHORT_SHA = "a2ce66d546b43e3bc03242a51827a155c7601956b1af52eefddb4fe99a72c449"
INVENTORY_SHA = "16f6eee4fe258e9e75a69a2bda75c7f3f5a00365ab833308bf1e1943fbcee45b"
CASES = 95
FILES = 190
BYTES_PER_COPY = 277_630_584
BOOTSTRAP_SEED = 2026090601
BOOTSTRAP_SAMPLES = 10_000
AXES = ((), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))

FULL972 = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth"
GCSAM184 = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth"
FULL972_SHA = "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
GCSAM184_SHA = "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"frozen SHA mismatch: {path}")


def documents() -> None:
    checked(RECORD / "IDEA.md", IDEA_SHA)
    checked(RECORD / "PREREGISTRATION.json", PREREG_SHA)
    checked(COHORT, COHORT_SHA)
    checked(INVENTORY, INVENTORY_SHA)


def cohort_rows() -> list[dict]:
    documents()
    value = json.loads(COHORT.read_text())
    rows = value["selection"]["rows"]
    if value.get("status") != "frozen_before_package_payload_nifti_or_label_access" or len(rows) != CASES:
        raise ValueError("frozen cohort contract mismatch")
    if value["selection"]["files"] != FILES or value["selection"]["bytes"] != BYTES_PER_COPY:
        raise ValueError("frozen cohort file/byte mismatch")
    return rows


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
