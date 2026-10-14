"""Audit official UTSW-Glioma metadata and Faspex inventory without pixels."""
from __future__ import annotations

import base64
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


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def browse_all(session: requests.Session, headers: dict[str, str]) -> list[dict]:
    pending = ["/"]
    files: list[dict] = []
    visited: set[str] = set()
    while pending:
        path = pending.pop(0)
        if path in visited:
            raise RuntimeError(f"directory cycle: {path}")
        visited.add(path)
        reply = session.post(
            f"{BASE}/api/v5/packages/{PACKAGE_ID}/files/received/page",
            headers=headers,
            json={"path": path, "filters": {}},
            timeout=60,
        )
        reply.raise_for_status()
        body = reply.json()
        items = body.get("items", [])
        total = body.get("total_count", len(items))
        if len(items) != total:
            raise RuntimeError(f"pagination unresolved for {path}: {len(items)} != {total}")
        for item in items:
            kind = item.get("type")
            if kind == "directory":
                pending.append(item["path"])
            elif kind == "file" or (
                kind == "symbolic_link" and item.get("target", {}).get("type") == "file"
            ):
                files.append(
                    {key: item.get(key) for key in (
                        "path", "basename", "type", "size", "mtime", "checksum", "sha256", "md5"
                    )}
                )
            else:
                raise RuntimeError(f"unsupported item type {kind!r} at {path}")
        print(json.dumps({"visited_directories": len(visited), "pending": len(pending), "files": len(files)}), flush=True)
    return sorted(files, key=lambda item: item["path"])


def main() -> None:
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    metadata = session.get(METADATA, timeout=60)
    metadata.raise_for_status()

    links = re.findall(r'https://faspex[^"<>\s]+', page.text)
    public_url = html.unescape(next(link for link in links if "context=" in link))
    state = parse_qs(urlparse(public_url).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if context.get("type") != "external_download_package" or int(context.get("id")) != PACKAGE_ID:
        raise RuntimeError(f"unexpected official package context: {context}")

    rows = list(csv.DictReader(metadata.text.splitlines(), delimiter="\t"))
    ids = [row["Subject ID"].strip() for row in rows]
    manual_ids = sorted(row["Subject ID"].strip() for row in rows if row["Manually Refined Segmentation"].strip() == "Yes")
    flags = sorted({row["Manually Refined Segmentation"].strip() for row in rows})
    if len(rows) != 625 or len(ids) != len(set(ids)) or len(manual_ids) != 362 or flags != ["No", "Yes"]:
        raise RuntimeError("official TSV cohort/manual-subset gate failed")
    if any(row["Operation Status"].strip() != "PRE" for row in rows):
        raise RuntimeError("non-preoperative row in official metadata")
    for sequence in ("T1", "T1GD", "T2", "T2FLAIR"):
        if any(row[sequence].strip() != "Available" for row in rows):
            raise RuntimeError(f"missing required sequence: {sequence}")

    config = session.get(f"{BASE}/config.js", timeout=60)
    config.raise_for_status()
    client_id = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(
        f"{BASE}/auth/authorize_public_link",
        params={"response_type": "code", "client_id": client_id, "redirect_uri": redirect, "state": state},
        allow_redirects=False,
        timeout=60,
    )
    auth.raise_for_status()
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(
        f"{BASE}/auth/token",
        json={
            "code": query["code"][0],
            "state": query.get("state", [state])[0],
            "grant_type": "authorization_code",
            "client_id": client_id,
            "redirect_uri": redirect,
        },
        timeout=60,
    )
    token.raise_for_status()
    bearer = token.json()["access_token"]
    headers = {"Authorization": f"Bearer {bearer}"}
    package_reply = session.get(f"{BASE}/api/v5/packages/{PACKAGE_ID}", headers=headers, timeout=60)
    package_reply.raise_for_status()
    package = package_reply.json()
    files = browse_all(session, headers)

    # Match paths case-insensitively; no voxel payload is opened or downloaded.
    path_rows = [(item["path"], item["path"].lower()) for item in files]
    matches = {}
    for subject in manual_ids:
        subject_paths = [path for path, lower in path_rows if subject.lower() in lower]
        t1gd = [path for path in subject_paths if re.search(r"(^|[/_.-])t1gd([/_.-]|$)", path.lower())]
        segmentations = [path for path in subject_paths if "seg" in Path(path).name.lower()]
        matches[subject] = {"all": subject_paths, "t1gd": t1gd, "segmentation": segmentations}

    unmatched = {
        subject: {key: value for key, value in values.items() if key != "all"}
        for subject, values in matches.items()
        if len(values["t1gd"]) != 1 or len(values["segmentation"]) != 1
    }
    report = {
        "schema": "aflow.utsw_glioma.source_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_metadata_and_inventory" if not unmatched else "failed_manual_subset_file_pairing",
        "official": {
            "collection_url": PAGE,
            "package_id": PACKAGE_ID,
            "context_type": context["type"],
            "page_sha256": sha(page.content),
            "metadata_url": METADATA,
            "metadata_sha256": sha(metadata.content),
            "license": "CC BY 4.0",
            "package": {key: package.get(key) for key in (
                "id", "title", "total_bytes", "total_files", "files_on_server",
                "pagination_browsing", "prevent_http_download", "download_availability"
            )},
        },
        "cohort": {
            "metadata_rows": len(rows),
            "unique_subjects": len(set(ids)),
            "manual_refined_subjects": len(manual_ids),
            "automatic_only_subjects": len(rows) - len(manual_ids),
            "operation_status_values": sorted({row["Operation Status"].strip() for row in rows}),
            "manual_flag_values": flags,
            "manual_subject_ids_sha256": sha(("\n".join(manual_ids) + "\n").encode()),
        },
        "inventory": {
            "file_count": len(files),
            "total_bytes": sum(int(item["size"]) for item in files),
            "paths_sha256": sha(("\n".join(item["path"] for item in files) + "\n").encode()),
            "manual_pairing_unmatched_count": len(unmatched),
            "manual_pairing_unmatched": unmatched,
        },
        "access": {
            "image_voxels_accessed": False,
            "label_voxels_accessed": False,
            "model_inference": False,
            "old_cohort_accessed": False,
            "package_payload_downloaded": False,
        },
        "interpretation": "Source eligibility only; no model evaluation is authorized. Preoperative UTSW must remain separate from post-treatment rankings.",
    }
    destination = OUT / "SOURCE_AUDIT.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "subjects": len(rows),
        "manual": len(manual_ids),
        "files": len(files),
        "bytes": report["inventory"]["total_bytes"],
        "unmatched": len(unmatched),
        "report_sha256": sha(destination.read_bytes()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
