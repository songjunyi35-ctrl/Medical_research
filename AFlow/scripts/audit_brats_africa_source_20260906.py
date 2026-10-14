#!/usr/bin/env python3
"""Recursively inventory public TCIA BraTS-Africa package 948 without payload access."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import parse_qs, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive/reviews/2026-09-06_brats_africa_source_audit"
PAGE = "https://www.cancerimagingarchive.net/collection/brats-africa/"
DATAINFO = "https://www.cancerimagingarchive.net/wp-content/uploads/BraTS-Africa_TCIA_datainfo_v2.xlsx"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PACKAGE_ID = 948


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def authorize() -> tuple[str, bytes, bytes, dict]:
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    datainfo = session.get(DATAINFO, timeout=60)
    datainfo.raise_for_status()
    links = [html.unescape(x) for x in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in x]
    contexts = []
    for link in links:
        state = parse_qs(urlparse(link).query)["context"][0]
        context = json.loads(base64.b64decode(state))
        if int(context.get("id", -1)) == PACKAGE_ID:
            contexts.append((state, context))
    if not contexts:
        raise RuntimeError("public package948 context not found")
    state, context = contexts[0]
    if context.get("type") != "external_download_package":
        raise RuntimeError("unexpected public-link context type")
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
        json={"code": query["code"][0], "state": query.get("state", [state])[0], "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect},
        timeout=60,
    )
    token.raise_for_status()
    bearer = token.json()["access_token"]
    package = session.get(f"{BASE}/api/v5/packages/{PACKAGE_ID}", headers={"Authorization": "Bearer " + bearer}, timeout=60)
    package.raise_for_status()
    return bearer, page.content, datainfo.content, package.json()


def list_directory(path: str, bearer: str) -> tuple[list[dict], int]:
    session = requests.Session()
    headers = {"Authorization": "Bearer " + bearer}
    token = None
    seen = set()
    items = []
    pages = 0
    while True:
        reply = None
        last_error = None
        for attempt in range(3):
            try:
                reply = session.post(
                    f"{BASE}/api/v5/packages/{PACKAGE_ID}/files/received/page",
                    headers=headers,
                    params={"iteration_token": token} if token else None,
                    json={"path": path, "filters": {}},
                    timeout=60,
                )
                reply.raise_for_status()
                break
            except requests.RequestException as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        if reply is None or not reply.ok:
            raise RuntimeError(f"directory listing failed after 3 attempts: {path}") from last_error
        body = reply.json()
        page_items = body.get("items", [])
        if body.get("item_count") != len(page_items):
            raise RuntimeError(f"page item count mismatch: {path}")
        items.extend(page_items)
        pages += 1
        token = reply.headers.get("x-aspera-next-iteration-token")
        if not token:
            break
        if token in seen:
            raise RuntimeError(f"repeated iteration token: {path}")
        seen.add(token)
    paths = [x.get("path") for x in items]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"duplicate child path: {path}")
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
    inventory_path = OUT / "OFFICIAL_INVENTORY.json"
    report_path = OUT / "SOURCE_AUDIT.json"
    datainfo_path = OUT / "source_metadata" / "BraTS-Africa_TCIA_datainfo_v2.xlsx"
    if inventory_path.exists() or report_path.exists() or datainfo_path.exists():
        raise FileExistsError("refusing to overwrite source audit")
    bearer, page_bytes, datainfo_bytes, package = authorize()
    pending = ["/"]
    listed_directories = set()
    all_items: dict[str, dict] = {}
    pages_by_directory = {}
    while pending:
        current = sorted(set(pending) - listed_directories)
        pending = []
        if not current:
            break
        with ThreadPoolExecutor(max_workers=min(4, len(current))) as pool:
            results = list(pool.map(lambda path: (path, *list_directory(path, bearer)), current))
        for path, children, pages in results:
            listed_directories.add(path)
            pages_by_directory[path] = pages
            for child in children:
                child_path = str(child.get("path"))
                if child_path in all_items and all_items[child_path] != compact(child):
                    raise RuntimeError("conflicting repeated inventory item")
                all_items[child_path] = compact(child)
                if child.get("type") == "directory":
                    pending.append(child_path)
        print(json.dumps({"directories_listed": len(listed_directories), "items_seen": len(all_items), "queued": len(pending)}), flush=True)
        if len(listed_directories) > 1000 or len(all_items) > 10000:
            raise RuntimeError("unexpected inventory scale")

    files = sorted((x for x in all_items.values() if x["type"] != "directory"), key=lambda x: x["path"])
    directories = sorted((x for x in all_items.values() if x["type"] == "directory"), key=lambda x: x["path"])
    bad_files = [x["path"] for x in files if int(x.get("size") or 0) <= 0 or x.get("target_type") not in (None, "file")]
    case_pattern = re.compile(r"BraTS-SSA-(?P<case>\d{5})-(?P<suffix>.+)\.nii\.gz$", re.I)
    by_case: dict[str, list[dict]] = {}
    unmatched = []
    for item in files:
        match = case_pattern.search(str(item["basename"]))
        if match:
            by_case.setdefault(match.group("case"), []).append(item)
        else:
            unmatched.append(item["path"])
    inventory = {
        "schema": "aflow.brats_africa.official_inventory",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": PACKAGE_ID,
        "files": files,
        "directories": directories,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    datainfo_path.parent.mkdir(parents=True, exist_ok=True)
    datainfo_path.write_bytes(datainfo_bytes)
    case_summary = {}
    for case, items in sorted(by_case.items()):
        names = [str(x["basename"]).lower() for x in items]
        case_summary[case] = {
            "files": len(items),
            "t1ce": [x["path"] for x in items if any(token in str(x["basename"]).lower() for token in ("-t1c.", "-t1ce.", "-t1gd."))],
            "seg": [x["path"] for x in items if any(token in str(x["basename"]).lower() for token in ("-seg.", "segmentation"))],
            "basenames": names,
        }
    paired = [case for case, value in case_summary.items() if len(value["t1ce"]) == 1 and len(value["seg"]) == 1]
    report = {
        "schema": "aflow.brats_africa.source_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "inventory_complete_pairing_to_be_interpreted",
        "official": {
            "package_id": PACKAGE_ID,
            "package_title": package.get("title"),
            "package_total_files": package.get("total_files"),
            "package_total_bytes": package.get("total_bytes") or package.get("total_size"),
            "tcia_page_sha256": sha(page_bytes),
            "datainfo_sha256": sha(datainfo_bytes),
            "license": "CC BY 4.0",
        },
        "inventory": {
            "directories_listed": len(listed_directories),
            "directory_entries": len(directories),
            "files": len(files),
            "bytes": sum(int(x.get("size") or 0) for x in files),
            "bad_files": bad_files,
            "pages_by_directory": pages_by_directory,
            "inventory_path": str(inventory_path.relative_to(ROOT)),
            "inventory_sha256": sha(inventory_path.read_bytes()),
        },
        "case_pattern": {"matched_cases": len(by_case), "paired_unique_t1ce_seg_cases": len(paired), "unmatched_paths": unmatched, "cases": case_summary},
        "access": {"package_payload": False, "image_voxels": False, "label_voxels": False, "model_inference": False, "gpu": False, "old_cohort": False},
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": sha(Path(__file__).read_bytes()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "package": report["official"], "inventory": {k: report["inventory"][k] for k in ("directories_listed", "directory_entries", "files", "bytes", "bad_files", "inventory_sha256")}, "matched_cases": len(by_case), "paired": len(paired), "unmatched": len(unmatched), "report_sha256": sha(report_path.read_bytes())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
