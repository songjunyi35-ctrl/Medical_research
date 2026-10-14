"""Download frozen ReMIND-002 T1ce once and both DICOM SEG series twice."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import requests

from scripts import remind_dicomseg_probe_protocol as p

OLD_DICOM = Path("/tmp/aflow-remind-probe-20260905/dicom")


def download_series(role: str, round_name: str) -> dict:
    asset = p.ASSETS[role]
    directory = p.WORK / f"{role}_{round_name}"
    directory.mkdir()
    zip_path = directory / f"{role}.zip"
    url = "https://nbia.cancerimagingarchive.net/nbia-api/services/v1/getImage"
    with requests.get(url, params={"SeriesInstanceUID": asset["series_uid"]}, stream=True, timeout=180) as reply:
        reply.raise_for_status()
        headers = {key: value for key, value in reply.headers.items()
                   if key.lower() in ("content-type", "content-length", "content-disposition", "etag")}
        with zip_path.open("xb") as handle:
            for block in reply.iter_content(8 << 20):
                if block:
                    handle.write(block)
    extract = directory / "extract"
    extract.mkdir()
    with zipfile.ZipFile(zip_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if any(Path(item.filename).is_absolute() or ".." in Path(item.filename).parts for item in members):
            raise ValueError("unsafe NBIA ZIP path")
        dcm_members = [item for item in members if item.filename.lower().endswith(".dcm")]
        other = [item for item in members if not item.filename.lower().endswith(".dcm")]
        if len(dcm_members) != asset["instances"] or [item.filename for item in other] != ["LICENSE"]:
            raise ValueError(f"unexpected {role} ZIP member contract")
        archive.extractall(extract)
    dcm_paths = sorted(extract.rglob("*.dcm"))
    manifest = [{"name": path.name, "bytes": path.stat().st_size, "sha256": p.sha(path)} for path in dcm_paths]
    if sum(row["bytes"] for row in manifest) != asset["bytes"]:
        raise ValueError(f"{role} DICOM bytes differ from official series metadata")
    return {
        "role": role,
        "round": round_name,
        "series_uid": asset["series_uid"],
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": p.sha(zip_path),
        "response_headers": headers,
        "dicom_dir": str(extract),
        "dicom_instances": len(manifest),
        "dicom_bytes": sum(row["bytes"] for row in manifest),
        "dicom_manifest": manifest,
        "license": {"bytes": other[0].file_size, "sha256": p.sha(extract / "LICENSE")},
    }


def manifest(paths: list[Path]) -> list[dict]:
    return [{"name": path.name, "bytes": path.stat().st_size, "sha256": p.sha(path)} for path in sorted(paths)]


def main() -> None:
    from scripts.freeze_remind_dicomseg_probe import check_implementation

    check_implementation()
    preaccess_path = p.RECORD / "PROBE_PREACCESS.json"
    preaccess = json.loads(preaccess_path.read_text())
    if preaccess.get("status") != "frozen_before_dicomseg_download_or_array_decode":
        raise ValueError("wrong ReMIND DICOM SEG preaccess")
    if p.WORK.exists():
        raise FileExistsError("ReMIND DICOM SEG probe work directory exists")
    if not OLD_DICOM.exists():
        raise FileNotFoundError("old fixed-case DICOM evidence is absent")
    if shutil.disk_usage(p.WORK.parent).free < 2 * 1024 ** 3:
        raise RuntimeError("less than 2GiB /tmp reserve")
    p.WORK.mkdir()
    t1ce = download_series("t1ce", "b")
    old = manifest(list(OLD_DICOM.rglob("*.dcm")))
    t1ce_repeat_exact = old == t1ce["dicom_manifest"] and len(old) == 192
    tumor_a = download_series("tumor", "a")
    tumor_b = download_series("tumor", "b")
    cerebrum_a = download_series("cerebrum", "a")
    cerebrum_b = download_series("cerebrum", "b")
    tumor_exact = tumor_a["dicom_manifest"] == tumor_b["dicom_manifest"]
    cerebrum_exact = cerebrum_a["dicom_manifest"] == cerebrum_b["dicom_manifest"]
    report = {
        "schema": "aflow.remind_dicomseg_source_probe.download",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_download" if t1ce_repeat_exact and tumor_exact and cerebrum_exact else "failed_download",
        "t1ce": t1ce,
        "tumor_a": tumor_a,
        "tumor_b": tumor_b,
        "cerebrum_a": cerebrum_a,
        "cerebrum_b": cerebrum_b,
        "gates": {"t1ce_dicom_manifest_repeat_exact": t1ce_repeat_exact,
                  "tumor_dicom_repeat_exact": tumor_exact,
                  "cerebrum_dicom_repeat_exact": cerebrum_exact},
        "all_download_gates_passed": t1ce_repeat_exact and tumor_exact and cerebrum_exact,
        "preaccess_sha256": p.sha(preaccess_path),
        "prior_pixel_access_preserved": True,
        "arrays_decoded": False,
        "other_candidate_pixels": False,
        "model_inference": False,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "DOWNLOAD_RECEIPT.json", report)
    print(json.dumps({"status": report["status"], "gates": report["gates"],
                      "zip_bytes": {role: report[role]["zip_bytes"] for role in (
                          "t1ce", "tumor_a", "tumor_b", "cerebrum_a", "cerebrum_b")}}, indent=2))
    if not report["all_download_gates_passed"]:
        raise RuntimeError("ReMIND DICOM SEG download gates failed")


if __name__ == "__main__":
    main()
