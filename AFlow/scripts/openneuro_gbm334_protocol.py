"""Frozen constants and contracts for the OpenNeuro-GBM334 panel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-06_openneuro_gbm334_frozen_panel"
REVIEW = ROOT / "archive/reviews/2026-09-06_openneuro_ds007045_source_audit"
INVENTORY = REVIEW / "SOURCE_INVENTORY.json"
PREREGISTERED_IDEA_SHA = "bfda8f45c1c5b981a13676335ba14672320283a6eea8171ac61a733808376e4f"
IDEA_SHA = "2853d8682bccd9d89b195c58781729cd2bad6ded8f8836bd79a85a792c0b573e"
PREREG_SHA = "9e269e850b7cbbfab7aea7f49e60280efebd971112bcbc18f54f241f9fa24265"
INVENTORY_SHA = "4bba11fb31913b18f44428087d9daf9dcd8f49590577ea11b715ddbcbbbf66eb"
MANIFEST_SHA = "8342256890d3b6a8ebf408ff081a84867f76fd816da0e97e76f98b41e1cce6cb"
CASES = 334
BOOTSTRAP_SEED = 20261640
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
    checked(INVENTORY, INVENTORY_SHA)


def cases() -> list[dict]:
    docs()
    inventory = json.loads(INVENTORY.read_text())
    rows = inventory["selected"]["cases"]
    if (
        inventory["status"] != "passed_metadata_only_payload_not_accessed_model_not_accessed"
        or inventory["snapshot_id"] != "ds007045:1.0.1"
        or inventory["participant_count_frozen_intersection"] != CASES
        or inventory["selected"]["manifest_sha256"] != MANIFEST_SHA
        or len(rows) != CASES
        or len({row["subject_id"] for row in rows}) != CASES
    ):
        raise ValueError("source inventory contract")
    return rows


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
