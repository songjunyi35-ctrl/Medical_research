"""Download twice and audit the frozen UTSW three-case label variants."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import parse_qs, urlparse

import nibabel as nib
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-05_utsw_label_variant_bridge"
SOURCE_REVIEW = ROOT / "archive/reviews/2026-09-05_utsw_glioma_source_audit"
INVENTORY = SOURCE_REVIEW / "OFFICIAL_INVENTORY_AMENDMENT_01.json"
PAGE = "https://www.cancerimagingarchive.net/collection/utsw-glioma/"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PACKAGE_ID = 1265
SUBJECTS = ("BT0001", "BT0002", "BT0003")
NAMES = (
    "brain_t1ce.nii.gz",
    "brain_t1ce_ants.nii.gz",
    "rtumorseg_manual_correction.nii.gz",
    "tumorseg_FeTS.nii.gz",
    "tumorseg_manual_correction.nii.gz",
)
IMAGES = ("brain_t1ce.nii.gz", "brain_t1ce_ants.nii.gz")
MANUALS = ("rtumorseg_manual_correction.nii.gz", "tumorseg_manual_correction.nii.gz")
WORK = Path("/tmp/aflow-utsw-label-variant-20260905")
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"
CONFIG = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf"
EXPECTED = {
    "inventory": "5f210c0a1e0c33f7b71a56d9bbce8b52a62f24e2a0b53655aa3a098dbde3f871",
    "idea": "f3e2d69ed75893f91f8fd86b4d1692b5fe1b5760cd9f2921cc727e674b3edcd5",
    "prereg": "2667210f9afaa1d09d874a41a5b6fa9661cb95133f85622237c1114e5f4fb8d7",
    "ascp": "f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe",
    "key": "9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645",
    "config": "edfcd31fd75b65f0c10b2bf0d06b983ca15f5ffb6adb8eba52be4817df08fadf",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def frozen_inventory() -> dict[str, dict]:
    data = json.loads(INVENTORY.read_text())
    selected = {
        item["path"]: item
        for item in data["files"]
        if item["path"].split("/")[-2] in SUBJECTS and item["basename"] in NAMES
    }
    if len(selected) != 15 or sum(int(item["size"]) for item in selected.values()) != 28_880_493:
        raise RuntimeError("frozen15 inventory mismatch")
    serialized = "".join(f"{path}\t{selected[path]['size']}\n" for path in sorted(selected))
    if hashlib.sha256(serialized.encode()).hexdigest() != "0f7fd05936e9b216ae744961d45d33a602d4ea7ba5e7fcaa5d6a54c8bd2c9a3c":
        raise RuntimeError("frozen path-size hash mismatch")
    return selected


def authorize() -> tuple[requests.Session, str]:
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    link = html.unescape(next(value for value in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in value))
    state = parse_qs(urlparse(link).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if int(context["id"]) != PACKAGE_ID or context["type"] != "external_download_package":
        raise RuntimeError("unexpected public package context")
    config = session.get(f"{BASE}/config.js", timeout=60)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(
        f"{BASE}/auth/authorize_public_link",
        params={"response_type": "code", "client_id": client, "redirect_uri": redirect, "state": state},
        allow_redirects=False,
        timeout=60,
    )
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(
        f"{BASE}/auth/token",
        json={
            "code": query["code"][0], "state": query.get("state", [state])[0],
            "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect,
        },
        timeout=60,
    )
    token.raise_for_status()
    return session, token.json()["access_token"]


def transfer_copy(copy_name: str, selected: dict[str, dict]) -> dict:
    session, bearer = authorize()
    headers = {"Authorization": "Bearer " + bearer}
    copy_root = WORK / copy_name
    copy_root.mkdir(parents=True, exist_ok=False)
    receipt = {"copy": copy_name, "subjects": {}}
    for subject in SUBJECTS:
        paths = [f"/UTSW-Glioma/{subject}/{name}" for name in NAMES]
        spec_reply = session.post(
            f"{BASE}/api/v5/packages/{PACKAGE_ID}/transfer_spec/download",
            headers=headers,
            params={"transfer_type": "connect", "type": "received"},
            json={"paths": [{"path": path, "type": "file"} for path in paths]},
            timeout=60,
        )
        if spec_reply.status_code != 201:
            raise RuntimeError(f"transfer spec HTTP {spec_reply.status_code}: {subject}")
        spec = spec_reply.json()
        sources = [item["source"] for item in spec.get("paths", [])]
        if len(sources) != 5 or {Path(source).name for source in sources} != set(NAMES):
            raise RuntimeError(f"transfer spec path expansion mismatch: {subject}")
        if any(source_path not in selected for source_path in paths):
            raise RuntimeError("non-frozen requested path")
        destination = copy_root / subject
        destination.mkdir()
        log_dir = WORK / "logs" / copy_name / subject
        log_dir.mkdir(parents=True)
        command = [
            str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
            "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY), "-l", "50m",
            "--overwrite=never", "-v", "-L", str(log_dir), "-f", str(CONFIG), *sources, str(destination),
        ]
        if "tags" in spec:
            command[1:1] = ["--tags64=" + base64.b64encode(json.dumps(spec["tags"]).encode()).decode()]
        environment = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
        transfer = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
        if transfer.returncode != 0:
            message = (transfer.stdout + transfer.stderr).replace(spec["token"], "[REDACTED]")[-1500:]
            raise RuntimeError(f"ascp failed {copy_name}/{subject}: {message}")
        files = [path for path in destination.rglob("*") if path.is_file()]
        by_name = {path.name: path for path in files}
        if set(by_name) != set(NAMES) or len(files) != 5:
            raise RuntimeError(f"downloaded file set mismatch: {copy_name}/{subject}: {sorted(by_name)}")
        subject_rows = {}
        for name in NAMES:
            logical = f"/UTSW-Glioma/{subject}/{name}"
            path = by_name[name]
            if path.stat().st_size != int(selected[logical]["size"]):
                raise RuntimeError(f"official listed size mismatch: {copy_name}/{subject}/{name}")
            subject_rows[name] = {"bytes": path.stat().st_size, "sha256": sha(path), "path": str(path)}
        receipt["subjects"][subject] = subject_rows
    return receipt


def array_validity(array: np.ndarray, is_label: bool) -> dict:
    finite = bool(np.isfinite(array).all())
    row = {
        "finite": finite,
        "shape": list(array.shape),
        "nonzero_voxels": int(np.count_nonzero(array)) if finite else None,
    }
    if is_label and finite:
        integer_exact = bool(np.array_equal(array, np.rint(array)))
        values = np.unique(array)
        row.update({
            "integer_exact": integer_exact,
            "values": [float(value) for value in values.tolist()],
            "includes_zero": bool(np.any(values == 0)),
            "includes_positive": bool(np.any(values > 0)),
        })
    return row


def geometry_row(image: nib.spatialimages.SpatialImage, label: nib.spatialimages.SpatialImage) -> dict:
    affine_max_abs = float(np.max(np.abs(np.asarray(image.affine) - np.asarray(label.affine))))
    return {
        "shape_exact": tuple(image.shape) == tuple(label.shape),
        "affine_max_abs": affine_max_abs,
        "passed": tuple(image.shape) == tuple(label.shape) and affine_max_abs <= 1e-6,
    }


def analyze(copy_receipt: dict) -> dict:
    cases = {}
    pair_passes = {f"{image}__{label}": True for image in IMAGES for label in MANUALS}
    validity_passed = True
    for subject in SUBJECTS:
        paths = {name: Path(copy_receipt["subjects"][subject][name]["path"]) for name in NAMES}
        objects = {name: nib.load(path, mmap=False) for name, path in paths.items()}
        arrays = {name: np.asanyarray(image.dataobj) for name, image in objects.items()}
        validity = {}
        for name in NAMES:
            is_label = name not in IMAGES
            validity[name] = array_validity(arrays[name], is_label=is_label)
            row = validity[name]
            good = row["finite"] and row["nonzero_voxels"] > 0
            if is_label:
                good = good and row["integer_exact"] and row["includes_zero"] and row["includes_positive"]
            validity_passed = validity_passed and good
        geometry = {}
        for image in IMAGES:
            for label in MANUALS:
                key = f"{image}__{label}"
                geometry[key] = geometry_row(objects[image], objects[label])
                pair_passes[key] = pair_passes[key] and geometry[key]["passed"]
        comparisons = {}
        fets = arrays["tumorseg_FeTS.nii.gz"] > 0
        for label in MANUALS:
            manual = arrays[label] > 0
            compatible = tuple(manual.shape) == tuple(fets.shape) and float(np.max(np.abs(objects[label].affine - objects["tumorseg_FeTS.nii.gz"].affine))) <= 1e-6
            if compatible:
                denom = int(manual.sum()) + int(fets.sum())
                comparisons[label] = {"geometry_compatible": True, "binary_dice_vs_fets": 2.0 * int(np.logical_and(manual, fets).sum()) / denom if denom else 1.0}
            else:
                comparisons[label] = {"geometry_compatible": False, "binary_dice_vs_fets": None}
        cases[subject] = {"validity": validity, "geometry": geometry, "descriptive_only": comparisons}
    winning = sorted(key for key, passed in pair_passes.items() if passed)
    return {"cases": cases, "all_case_pair_passes": pair_passes, "winning_pairs": winning, "validity_passed": validity_passed}


def main() -> None:
    required = {
        "inventory": INVENTORY,
        "idea": RECORD / "IDEA.md",
        "prereg": RECORD / "PREREG.md",
        "ascp": ASCP,
        "key": KEY,
        "config": CONFIG,
    }
    actual = {name: sha(path) for name, path in required.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen dependency hash mismatch: {actual}")
    selected = frozen_inventory()
    if WORK.exists():
        raise FileExistsError(WORK)
    if any((RECORD / name).exists() for name in ("PREACCESS.json", "TRANSFER_RECEIPT.json", "RAW_QC.json")):
        raise FileExistsError("formal access already started")
    write_new(RECORD / "PREACCESS.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_utsw_payload",
        "subjects": list(SUBJECTS), "names": list(NAMES), "expected_single_copy_bytes": 28_880_493,
        "dependency_sha256": actual,
        "image_voxels_accessed": False, "label_voxels_accessed": False,
        "model_inference": False, "old_cohort_accessed": False,
    })
    WORK.mkdir()
    copy_a = transfer_copy("copy_a", selected)
    copy_b = transfer_copy("copy_b", selected)
    exact = True
    pairs = {}
    for subject in SUBJECTS:
        pairs[subject] = {}
        for name in NAMES:
            a = copy_a["subjects"][subject][name]
            b = copy_b["subjects"][subject][name]
            passed = a["bytes"] == b["bytes"] and a["sha256"] == b["sha256"]
            exact = exact and passed
            pairs[subject][name] = {"bytes": a["bytes"], "sha256": a["sha256"], "copy_exact": passed}
    transfer_receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if exact else "failed",
        "subjects": pairs, "files_exact": sum(row["copy_exact"] for values in pairs.values() for row in values.values()),
        "expected_files": 15, "copy_a_bytes": sum(row["bytes"] for values in copy_a["subjects"].values() for row in values.values()),
        "copy_b_bytes": sum(row["bytes"] for values in copy_b["subjects"].values() for row in values.values()),
        "image_voxels_accessed": False, "label_voxels_accessed": False, "model_inference": False,
    }
    write_new(RECORD / "TRANSFER_RECEIPT.json", transfer_receipt)
    if not exact:
        raise RuntimeError("dual transfer exact gate failed before pixel access")
    raw = analyze(copy_a)
    raw.update({
        "schema": "aflow.utsw_label_variant_bridge.raw_qc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_unique_geometry_pair" if raw["validity_passed"] and len(raw["winning_pairs"]) == 1 else "failed_unresolved_geometry_pair",
        "transfer_receipt_sha256": sha(RECORD / "TRANSFER_RECEIPT.json"),
        "image_voxels_accessed": True, "label_voxels_accessed": True,
        "model_inference": False, "old_cohort_accessed": False,
    })
    write_new(RECORD / "RAW_QC.json", raw)
    print(json.dumps({
        "status": raw["status"], "winning_pairs": raw["winning_pairs"],
        "validity_passed": raw["validity_passed"], "dual_transfer_exact": exact,
        "transfer_sha256": sha(RECORD / "TRANSFER_RECEIPT.json"), "raw_sha256": sha(RECORD / "RAW_QC.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
