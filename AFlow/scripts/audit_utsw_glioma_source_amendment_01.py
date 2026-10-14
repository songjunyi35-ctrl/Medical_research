"""Complete the UTSW inventory with official iteration-token pagination."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive/reviews/2026-09-05_utsw_glioma_source_audit"
PAGE = "https://www.cancerimagingarchive.net/collection/utsw-glioma/"
METADATA = "https://www.cancerimagingarchive.net/wp-content/uploads/UTSW_Glioma_Metadata-2-1.tsv"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PACKAGE_ID = 1265
COMMON = {
    "brain_fl_ants.nii.gz", "brain_flair.nii.gz", "brain_t1.nii.gz", "brain_t1_ants.nii.gz",
    "brain_t1ce.nii.gz", "brain_t1ce_ants.nii.gz", "brain_t2.nii.gz", "brain_t2_ants.nii.gz",
    "tumorseg_FeTS.nii.gz",
}
MANUAL_EXTRA = {"rtumorseg_manual_correction.nii.gz", "tumorseg_manual_correction.nii.gz"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authorize() -> tuple[str, bytes, bytes, dict]:
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    metadata = session.get(METADATA, timeout=60)
    metadata.raise_for_status()
    link = html.unescape(next(value for value in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in value))
    state = parse_qs(urlparse(link).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if context.get("type") != "external_download_package" or int(context.get("id")) != PACKAGE_ID:
        raise RuntimeError("unexpected package context")
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
    token_reply = session.post(
        f"{BASE}/auth/token",
        json={
            "code": query["code"][0], "state": query.get("state", [state])[0],
            "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect,
        },
        timeout=60,
    )
    token_reply.raise_for_status()
    bearer = token_reply.json()["access_token"]
    package_reply = session.get(
        f"{BASE}/api/v5/packages/{PACKAGE_ID}", headers={"Authorization": "Bearer " + bearer}, timeout=60
    )
    package_reply.raise_for_status()
    return bearer, page.content, metadata.content, package_reply.json()


def list_directory(path: str, bearer: str) -> tuple[list[dict], int]:
    session = requests.Session()
    headers = {"Authorization": "Bearer " + bearer}
    token = None
    seen_tokens: set[str] = set()
    items: list[dict] = []
    pages = 0
    while True:
        params = {"iteration_token": token} if token else None
        reply = session.post(
            f"{BASE}/api/v5/packages/{PACKAGE_ID}/files/received/page",
            headers=headers,
            params=params,
            json={"path": path, "filters": {}},
            timeout=60,
        )
        reply.raise_for_status()
        body = reply.json()
        page_items = body.get("items", [])
        if body.get("item_count") != len(page_items):
            raise RuntimeError(f"page item_count mismatch: {path}")
        items.extend(page_items)
        pages += 1
        token = reply.headers.get("x-aspera-next-iteration-token")
        if not token:
            break
        if token in seen_tokens:
            raise RuntimeError(f"repeated iteration token: {path}")
        seen_tokens.add(token)
    paths = [item["path"] for item in items]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"duplicate item path: {path}")
    return items, pages


def compact(item: dict) -> dict:
    target = item.get("target") or {}
    return {
        "path": item.get("path"),
        "basename": item.get("basename"),
        "type": item.get("type"),
        "size": item.get("size") if item.get("size") is not None else target.get("size"),
        "target_type": target.get("type"),
        "checksum": item.get("checksum") or target.get("checksum"),
        "md5": item.get("md5") or target.get("md5"),
        "sha256": item.get("sha256") or target.get("sha256"),
    }


def main() -> None:
    bearer, page_bytes, metadata_bytes, package = authorize()
    rows = list(csv.DictReader(metadata_bytes.decode().splitlines(), delimiter="\t"))
    by_id = {row["Subject ID"].strip(): row for row in rows}
    if len(rows) != 625 or len(by_id) != 625:
        raise RuntimeError("metadata cohort identity failed")
    manual = {subject for subject, row in by_id.items() if row["Manually Refined Segmentation"].strip() == "Yes"}
    if len(manual) != 362:
        raise RuntimeError("manual subset count failed")

    root_items, root_pages = list_directory("/", bearer)
    if len(root_items) != 1 or root_items[0].get("path") != "/UTSW-Glioma" or root_items[0].get("type") != "directory":
        raise RuntimeError("unexpected package root")
    subjects, subject_pages = list_directory("/UTSW-Glioma", bearer)
    subject_ids = [item.get("basename") for item in subjects]
    if set(subject_ids) != set(by_id) or len(subject_ids) != 625:
        raise RuntimeError("package subject directories do not equal TSV IDs")
    if any(item.get("type") != "directory" for item in subjects):
        raise RuntimeError("non-directory subject item")

    def read_subject(subject: str) -> tuple[str, list[dict], int]:
        values, pages = list_directory(f"/UTSW-Glioma/{subject}", bearer)
        return subject, values, pages

    listings: dict[str, list[dict]] = {}
    per_subject_pages = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for number, (subject, values, pages) in enumerate(pool.map(read_subject, sorted(by_id)), start=1):
            listings[subject] = values
            per_subject_pages[subject] = pages
            if number % 50 == 0 or number == 625:
                print(json.dumps({"subjects_listed": number, "expected": 625}), flush=True)

    failures = {}
    compact_files = []
    for subject in sorted(by_id):
        values = listings[subject]
        basenames = {item.get("basename") for item in values}
        expected = COMMON | (MANUAL_EXTRA if subject in manual else set())
        bad_type = [item.get("path") for item in values if item.get("type") != "symbolic_link" or (item.get("target") or {}).get("type") != "file"]
        bad_size = [item.get("path") for item in values if int(item.get("size") or (item.get("target") or {}).get("size") or 0) <= 0]
        if basenames != expected or len(values) != len(expected) or bad_type or bad_size or per_subject_pages[subject] != 1:
            failures[subject] = {
                "missing": sorted(expected - basenames), "extra": sorted(basenames - expected),
                "count": len(values), "bad_type": bad_type, "bad_size": bad_size,
                "pages": per_subject_pages[subject],
            }
        compact_files.extend(compact(item) for item in values)
    compact_files.sort(key=lambda item: item["path"])
    official_total = int(package.get("total_files"))
    if official_total != 6349 or len(compact_files) != 6349:
        failures["__total__"] = {"package_total_files": official_total, "listed_files": len(compact_files)}

    inventory_path = OUT / "OFFICIAL_INVENTORY_AMENDMENT_01.json"
    report_path = OUT / "SOURCE_AUDIT_AMENDMENT_01.json"
    if inventory_path.exists() or report_path.exists():
        raise FileExistsError("amendment output already exists")
    inventory = {
        "schema": "aflow.utsw_glioma.official_inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": PACKAGE_ID,
        "files": compact_files,
    }
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    report = {
        "schema": "aflow.utsw_glioma.source_audit_amendment_01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_identity_inventory_label_variant_unresolved" if not failures else "failed_full_inventory_gate",
        "supersedes_interpretation_not_evidence": "SOURCE_AUDIT.json",
        "implementation_failure_preserved": True,
        "official": {
            "package_id": PACKAGE_ID, "license": "CC BY 4.0",
            "page_sha256": sha(page_bytes), "metadata_sha256": sha(metadata_bytes),
            "package_total_files": official_total,
            "package_title": package.get("title"),
            "pagination_browsing": package.get("pagination_browsing"),
        },
        "cohort": {
            "subjects": len(by_id), "manual_refined": len(manual), "automatic_only": len(by_id) - len(manual),
            "manual_subject_ids_sha256": sha(("\n".join(sorted(manual)) + "\n").encode()),
        },
        "inventory": {
            "root_pages": root_pages, "subject_directory_pages": subject_pages,
            "listed_files": len(compact_files), "listed_bytes": sum(int(item["size"]) for item in compact_files),
            "inventory_path": str(inventory_path.relative_to(ROOT)),
            "inventory_sha256": sha(inventory_path.read_bytes()),
            "failures": failures,
            "expected_formula": "362*11 + 263*9 = 6349",
        },
        "label_variant": {
            "status": "unresolved_without_pixel_access",
            "candidates": sorted(MANUAL_EXTRA),
            "future_requirement": "separately preregister fixed-case geometry and label-semantic bridge before selecting a manual mask",
        },
        "access": {
            "image_voxels_accessed": False, "label_voxels_accessed": False,
            "model_inference": False, "old_cohort_accessed": False, "package_payload_downloaded": False,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "subjects": len(by_id), "manual": len(manual),
        "files": len(compact_files), "bytes": report["inventory"]["listed_bytes"],
        "failures": len(failures), "inventory_sha256": report["inventory"]["inventory_sha256"],
        "report_sha256": sha(report_path.read_bytes()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
