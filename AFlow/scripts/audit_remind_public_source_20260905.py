"""Audit official ReMIND series and NRRD-package metadata without pixel access."""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive/reviews/2026-09-05_remind_source_audit/REMIND_PUBLIC_SOURCE.json"
PAGE = "https://www.cancerimagingarchive.net/collection/remind/"
SERIES = "https://nbia.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=ReMIND&format=json"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"


def main() -> None:
    session = requests.Session()
    page = session.get(PAGE, timeout=30)
    page.raise_for_status()
    links = [html.unescape(value) for value in re.findall(r'https://faspex[^"<>\s]+', page.text)]
    candidates = {}
    for link in links:
        values = parse_qs(urlparse(link).query).get("context", [])
        if not values:
            continue
        state_value = values[0]
        try:
            decoded = json.loads(base64.b64decode(state_value))
        except (ValueError, json.JSONDecodeError):
            continue
        if str(decoded.get("id")) == "755":
            candidates[state_value] = decoded
    if len(candidates) != 1:
        raise ValueError("expected exactly one unique official ReMIND package755 public link")
    state, context = next(iter(candidates.items()))
    if context.get("id") != "755" or context.get("type") != "external_download_package":
        raise ValueError("ReMIND official public link is not package755")
    config = session.get(BASE + "/config.js", timeout=30)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    authorization = session.get(
        BASE + "/auth/authorize_public_link",
        params={"response_type": "code", "client_id": client, "redirect_uri": BASE + "/token", "state": state},
        allow_redirects=False,
        timeout=30,
    )
    if authorization.status_code != 302:
        raise RuntimeError("ReMIND public authorization failed")
    query = parse_qs(urlparse(authorization.headers["Location"]).query)
    token = session.post(
        BASE + "/auth/token",
        json={"code": query["code"][0], "state": query.get("state", [state])[0],
              "grant_type": "authorization_code", "client_id": client, "redirect_uri": BASE + "/token"},
        timeout=30,
    )
    token.raise_for_status()
    bearer = token.json()["access_token"]
    headers = {"Authorization": "Bearer " + bearer}
    package = session.get(BASE + "/api/v5/packages/755", headers=headers, timeout=30)
    package.raise_for_status()
    package_value = package.json()
    pending = ["/"]
    files = []
    while pending:
        path = pending.pop(0)
        reply = session.post(
            BASE + "/api/v5/packages/755/files/received/page",
            headers=headers,
            json={"path": path, "filters": {}},
            timeout=30,
        )
        reply.raise_for_status()
        value = reply.json()
        items = value.get("items", [])
        if len(items) != value.get("total_count", len(items)):
            raise RuntimeError("ReMIND package listing is paginated/incomplete")
        for item in items:
            if item["type"] == "directory":
                pending.append(item["path"])
            elif item["type"] == "file" or (
                item["type"] == "symbolic_link" and item.get("target", {}).get("type") == "file"
            ):
                files.append({key: item.get(key) for key in (
                    "path", "basename", "type", "size", "mtime", "checksum", "sha256", "md5"
                )})
            else:
                raise ValueError("unsupported ReMIND package entry type")
    series_reply = session.get(SERIES, timeout=60)
    series_reply.raise_for_status()
    series_rows = series_reply.json()
    selected = [row for row in series_rows if row.get("StudyDesc") == "Preop"
                and row.get("Modality") == "MR"
                and "postcontrast" in row.get("SeriesDescription", "").lower()]
    identifiers = sorted({match.group(0) for row in files
                          for match in [re.search(r"ReMIND-\d{3}", row["path"], re.I)] if match})
    report = {
        "schema": "aflow.remind.official_source_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "official_metadata_available_no_pixels_accessed",
        "collection": {"doi": "10.7937/3RAG-D070", "subjects": 114, "license": "CC BY 4.0"},
        "series_api": {
            "url": SERIES,
            "response_bytes": len(series_reply.content),
            "response_sha256": hashlib.sha256(series_reply.content).hexdigest(),
            "rows": len(series_rows),
            "patients": len({row["PatientID"] for row in series_rows}),
            "modality_counts": dict(Counter(row["Modality"] for row in series_rows)),
            "preop_t1_postcontrast": {
                "rows": len(selected),
                "patients": len({row["PatientID"] for row in selected}),
                "bytes": sum(int(row["FileSize"]) for row in selected),
                "images": sum(int(row["ImageCount"]) for row in selected),
                "series": [{key: row.get(key) for key in (
                    "PatientID", "SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription",
                    "ImageCount", "FileSize", "LicenseName", "LicenseURI"
                )} for row in sorted(selected, key=lambda item: item["PatientID"])],
            },
        },
        "segmentation_package": {
            "id": 755,
            "title": package_value.get("title"),
            "total_bytes_field": package_value.get("total_bytes"),
            "total_files_field": package_value.get("total_files"),
            "files_on_server": package_value.get("files_on_server"),
            "listed_files": len(files),
            "listed_bytes": sum(int(row["size"]) for row in files),
            "patient_ids_in_paths": identifiers,
            "patient_ids_in_paths_count": len(identifiers),
            "extension_counts": dict(Counter(Path(row["basename"]).suffix.lower() for row in files)),
            "files": sorted(files, key=lambda row: row["path"]),
        },
        "access": {
            "official_web": True,
            "official_series_metadata": True,
            "official_package_inventory": True,
            "dicom_pixels": False,
            "nrrd_bytes": False,
            "segmentation_voxels": False,
            "model_inference": False,
            "new_metrics": False,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with OUT.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"],
        "series_rows": report["series_api"]["rows"],
        "preop_t1_postcontrast": report["series_api"]["preop_t1_postcontrast"],
        "segmentation_package": {key: value for key, value in report["segmentation_package"].items() if key != "files"},
        "access": report["access"],
    }, indent=2))


if __name__ == "__main__":
    main()
