"""Frozen paths, hashes, and constants for the WAW-GliomAI152 panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-06_waw_gliomai152_frozen_panel"
REVIEW = ROOT / "archive/reviews/2026-09-06_waw_gliomai_source_audit"
SOURCE_ROOT = ROOT / "workspace/BraTS/waw_gliomai_source_20260906"
CACHE = SOURCE_ROOT / "cache128"
COHORT = REVIEW / "FROZEN_COHORT.json"
SOURCE_RECEIPT = RECORD / "SOURCE_RECEIPT.json"
CACHE_RECEIPT = RECORD / "CACHE_RECEIPT.json"
FIRST_PROBE = RECORD / "FIRST_CASE_PROBE.json"
MODEL_PREACCESS = RECORD / "MODEL_PREACCESS_V2.json"
PREREGISTERED_IDEA_SHA = "5ccf61e564d321cc32fd4c92d193f1802020aae0d1c55e189d3067d0387fb0db"
IDEA_SHA = "f7a950e6af947591d0457de15692bb3b59947248054f6c2d8293b737f6e62dd0"
PREREG_SHA = "5dd8156a7740b1b115e7e54ca20e727a47e7913360015bd7033ff9a58e721448"
COHORT_SHA = "25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a"
SOURCE_PREACCESS_SHA = "860f9327a06ec527e0211a13aae7cdf43b6a30a7bd0f14b7b55e7706e2aacc74"
SOURCE_RECEIPT_SHA = "834e3d410be61282536a705c3bd49d91ec3734bda88afabe35283705d2597db4"
CASES = 152
BOOTSTRAP_SEED = 20261452
BOOTSTRAP_SAMPLES = 10000
AXES = ((), (-1,), (-2,), (-2, -1), (-3,), (-3, -1), (-3, -2), (-3, -2, -1))
FULL = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth"
GCSAM = ROOT / "workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth"
FULL_SHA = "f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b"
GCSAM_SHA = "acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA mismatch: {path}")


def docs() -> None:
    checked(RECORD / "IDEA.md", IDEA_SHA)
    checked(RECORD / "PREREG.md", PREREG_SHA)
    checked(RECORD / "SOURCE_PREACCESS.json", SOURCE_PREACCESS_SHA)
    checked(COHORT, COHORT_SHA)
    checked(SOURCE_RECEIPT, SOURCE_RECEIPT_SHA)


def cases() -> list[dict]:
    docs()
    cohort = json.loads(COHORT.read_text())
    rows = cohort["cases"]
    if (
        cohort["status"] != "frozen_before_coreg_image_archive_access"
        or cohort["case_count"] != CASES
        or cohort["subject_count"] != CASES
        or len(rows) != CASES
    ):
        raise ValueError("cohort contract")
    return rows


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
