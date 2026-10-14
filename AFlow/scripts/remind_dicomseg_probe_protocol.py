"""Frozen constants/helpers for the ReMIND-002 DICOM SEG source probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-05_remind_dicomseg_source_probe"
OLD_RECORD = ROOT / "archive/research_records/2026-09-05_remind_t1ce_source_probe"
WORK = Path("/tmp/aflow-remind-dicomseg-probe-20260905")
DEPENDENCY = Path("/tmp/aflow-remind-dicomseg-deps/pydicom-3.0.1")
WHEEL = Path("/tmp/aflow-remind-dicomseg-deps/pydicom-3.0.1-py3-none-any.whl")
IDEA_SHA = "4fcc7f13425931b1a21c0ca976034b37fbc4c48e57a9f4b3d8c1c869887ef61f"
PREREG_SHA = "ddfefd074b4ce04e53f6d44d5fa8bc563dc85fcc514b67e1cf55c17a586b9212"
AMENDMENT_SHA = "326839aa18d0c01822fe693cf8e447ac4bc0e4348affa90f2d625daf1b8ad769"
COHORT_SHA = "8118827aa001dead98cc8cc1f6ee5a2fa9b797a6a8797329ba22f4766d669536"
OLD_RESULT_SHA = "6cd6906a4da72e6ab32781392cc16a886dd554efd1123cf1a80fdfae27816da6"
OLD_DOWNLOAD_SHA = "c68342445c57f55c3cafcf06cb330c939a1d0cbdf222ed5cacc6ebab25278e91"
WHEEL_SHA = "db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41"
DEPENDENCY_MANIFEST_SHA = "49e2b56e2fac41da819f22e8801eb5b7afa5437f97f5eb48ddf3af43574c9946"
ASSETS = {
    "t1ce": {
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.309430968157051488200208000420039817667",
        "study_uid": "1.3.6.1.4.1.14519.5.2.1.322752919633264035254585915433270238948",
        "description": "3D_AX_T1_postcontrast", "instances": 192, "bytes": 21660158,
    },
    "tumor": {
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.80716149485382078399551532220853537216",
        "study_uid": "1.3.6.1.4.1.14519.5.2.1.322752919633264035254585915433270238948",
        "description": "tumor seg - MR ref: 3D_AX_T1_postcontrast", "instances": 1, "bytes": 17148,
    },
    "cerebrum": {
        "series_uid": "1.3.6.1.4.1.14519.5.2.1.323423363124946855158015352096264740086",
        "study_uid": "1.3.6.1.4.1.14519.5.2.1.322752919633264035254585915433270238948",
        "description": "cerebrum seg - MR ref: 3D_AX_T1_postcontrast", "instances": 1, "bytes": 4834884,
    },
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_manifest_sha() -> str:
    rows = []
    for path in sorted(item for item in DEPENDENCY.rglob("*")
                       if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"):
        rows.append({"path": str(path.relative_to(DEPENDENCY)), "bytes": path.stat().st_size, "sha256": sha(path)})
    raw = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def docs() -> None:
    frozen = {
        RECORD / "IDEA.md": IDEA_SHA,
        RECORD / "PREREG.md": PREREG_SHA,
        RECORD / "AMENDMENT_01.md": AMENDMENT_SHA,
        OLD_RECORD / "COHORT_CANDIDATES.json": COHORT_SHA,
        OLD_RECORD / "RESULT.json": OLD_RESULT_SHA,
        OLD_RECORD / "DOWNLOAD_RECEIPT.json": OLD_DOWNLOAD_SHA,
        WHEEL: WHEEL_SHA,
    }
    for path, expected in frozen.items():
        if sha(path) != expected:
            raise ValueError(f"frozen ReMIND DICOM SEG input changed: {path}")
    if dependency_manifest_sha() != DEPENDENCY_MANIFEST_SHA:
        raise ValueError("extracted pydicom dependency manifest changed")


def write_new(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
