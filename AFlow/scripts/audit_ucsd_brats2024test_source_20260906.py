"""Read-only source audit of TCIA UCSD-PTGBM package 1221 (BraTS-GLI 2024 Test Data).

Metadata only: recursive paginated file inventory, optional small .sums checksum
download, and patient-overlap audit against the project's 1,350-case BraTS union
and the previously accessed UCSD main release (136 subjects). No image or label
pixel download, no model access.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/"
PACKAGE_ID = 1221
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
OUT = ROOT / "archive/reviews/2026-09-06_ucsd_brats2024test_source_audit"
CLINICAL = ROOT / "archive/reviews/2026-09-05_30h_chain_audit/source_metadata/UCSD_PTGBM-clinical-information_v3_2026-12-Mar.xlsx"
PRIOR_INVENTORY = ROOT / "archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/OFFICIAL_ASSET_INVENTORY.json"
TRAIN_MANIFEST = ROOT / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/train_cases.jsonl"
TEST_MANIFEST = ROOT / "data/datasets/brats2024_random_subject_80_20_seed20260722_v1/sealed_test/test_cases.jsonl"
ARCHIVED_TRAIN_SHA = "aa2e55915e2a4bc7c2af2e95d6e08f2199397694fe4a92ea66999b534a22f63d"
ARCHIVED_TEST_SHA = "7b91690c63a5eb359be105b3cc7c4ad563d41fca2b2c84e521251b7ed006b8d2"


def authorize(session: requests.Session):
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    states = []
    for link in re.findall(r'https://faspex[^"<>\s]+', page.text):
        state = parse_qs(urlparse(html.unescape(link)).query).get("context", [None])[0]
        if state:
            ctx = json.loads(base64.b64decode(state))
            if str(ctx.get("id")) == str(PACKAGE_ID):
                states.append(state)
    if not states or len(set(states)) != 1:
        raise RuntimeError(f"expected exactly one faspex link for package {PACKAGE_ID}, found {len(set(states))}")
    state = states[0]
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
    headers = {"Authorization": "Bearer " + token.json()["access_token"]}
    meta = session.get(f"{BASE}/api/v5/packages/{PACKAGE_ID}", headers=headers, timeout=60)
    meta.raise_for_status()
    return headers, page.content, meta.json()


def list_directory(session, headers, path):
    items, pages, token = [], 0, None
    while True:
        params = {"iteration_token": token} if token else None
        reply = session.post(
            f"{BASE}/api/v5/packages/{PACKAGE_ID}/files/received/page",
            headers=headers, params=params, json={"path": path, "filters": {}}, timeout=60,
        )
        reply.raise_for_status()
        body = reply.json()
        page_items = body.get("items", [])
        if body.get("item_count") is not None and body.get("item_count") != len(page_items):
            raise RuntimeError(f"page item_count mismatch at {path}")
        items.extend(page_items)
        pages += 1
        token = reply.headers.get("x-aspera-next-iteration-token")
        if not token:
            return items, pages


def walk_all(session, headers):
    inventory, dir_pages = [], {}
    queue = ["/"]
    while queue:
        path = queue.pop(0)
        items, pages = list_directory(session, headers, path)
        dir_pages[path] = pages
        for item in items:
            row = {k: item.get(k) for k in ("path", "basename", "type", "size")}
            inventory.append(row)
            if item.get("type") == "directory":
                queue.append(item["path"])
    return inventory, dir_pages


def clinical_mapping():
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(CLINICAL) as z:
        shared = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        rows = ET.fromstring(z.read("xl/worksheets/sheet2.xml")).findall("m:sheetData/m:row", ns)
        pairs = []
        for row in rows[1:]:
            values = {}
            for cell in row.findall("m:c", ns):
                col = re.sub(r"\d", "", cell.get("r", ""))
                if col not in ("A", "B"):
                    continue
                value = cell.find("m:v", ns)
                text = value.text if value is not None else ""
                values[col] = shared[int(text)] if cell.get("t") == "s" and text else text
            if values.get("A"):
                pairs.append((values["A"], values.get("B", "")))
    mapping = dict(pairs)
    if len(pairs) != len(mapping):
        raise RuntimeError("duplicate clinical rows")
    return mapping


def union_subjects():
    for path, expected in ((TRAIN_MANIFEST, ARCHIVED_TRAIN_SHA), (TEST_MANIFEST, ARCHIVED_TEST_SHA)):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"manifest changed since archived audit: {path}")
    subjects = set()
    for path in (TRAIN_MANIFEST, TEST_MANIFEST):
        for line in path.read_text().splitlines():
            subjects.add(json.loads(line)["subject_id"])
    return subjects


def main():
    OUT.mkdir(exist_ok=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1.0, allowed_methods=frozenset(("GET", "POST")))))
    headers, page_bytes, package_meta = authorize(session)
    inventory, dir_pages = walk_all(session, headers)
    files = [row for row in inventory if row["type"] == "file"]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": PACKAGE_ID,
        "collection_page_sha256": hashlib.sha256(page_bytes).hexdigest(),
        "package_meta": {k: package_meta.get(k) for k in ("id", "title", "created_at", "total_bytes", "file_count") if k in package_meta},
        "directories": sum(1 for r in inventory if r["type"] == "directory"),
        "files": len(files),
        "file_bytes_total": sum(int(r["size"] or 0) for r in files),
        "dir_pages": dir_pages,
        "image_or_label_pixels_downloaded": False,
        "model_access": False,
    }
    inv_path = OUT / "PACKAGE1221_INVENTORY.json"
    with inv_path.open("x") as f:
        json.dump({"package_id": PACKAGE_ID, "inventory": inventory}, f, indent=1, sort_keys=True)
        f.write("\n")
    report["inventory_sha256"] = hashlib.sha256(inv_path.read_bytes()).hexdigest()

    # per-case structure
    case_files = {}
    for row in files:
        parts = [p for p in row["path"].split("/") if p]
        if len(parts) >= 2:
            case_files.setdefault(parts[-2], []).append(row["basename"])
    subject_pattern = re.compile(r"(UCSD-PTGBM-\d{4})")
    brats_pattern = re.compile(r"(BraTS-GLI-\d{5})")
    ucsd_subjects = set()
    brats_subjects_in_names = set()
    for row in files:
        m = subject_pattern.search(row["path"])
        if m:
            ucsd_subjects.add(m.group(1))
        b = brats_pattern.search(row["path"])
        if b:
            brats_subjects_in_names.add(b.group(1))
    report["case_directories"] = len(case_files)
    report["ucsd_subject_ids_seen"] = sorted(ucsd_subjects)
    report["brats_subject_ids_seen"] = sorted(brats_subjects_in_names)
    report["basename_kinds"] = sorted({re.sub(r"UCSD-PTGBM-\d{4}_\d{2}_?|BraTS-GLI-\d{5}-\d{3}[-_]?", "", r["basename"]) for r in files})[:40]

    # checksum metadata (.sums) download only
    sums_rows = [r for r in files if r["basename"].endswith(".sums")]
    report["sums_files"] = sums_rows
    if len(sums_rows) == 1 and int(sums_rows[0]["size"]) < 5_000_000:
        spec_reply = session.post(
            f"{BASE}/api/v5/packages/{PACKAGE_ID}/transfer_spec/download",
            headers=headers, params={"type": "received", "transfer_type": "connect"},
            json={"paths": [{"path": sums_rows[0]["path"], "type": "file"}]}, timeout=60,
        )
        spec_reply.raise_for_status()
        spec = spec_reply.json()
        sources = [row["source"] for row in spec["paths"]]
        env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
        cmd = [str(SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"), "--mode=recv",
               "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
               "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]),
               "-i", str(SDK / "ibm-public-client-key.pem"), "-L", str(SDK), "--overwrite=never", "-l", "20m",
               "--tags64=" + base64.b64encode(json.dumps(spec.get("tags", {})).encode()).decode(),
               *sources, str(OUT)]
        transfer = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        report["sums_transfer_returncode"] = transfer.returncode
        sums_path = OUT / sums_rows[0]["basename"]
        if sums_path.is_file():
            report["official_sums"] = {"bytes": sums_path.stat().st_size,
                                       "sha256": hashlib.sha256(sums_path.read_bytes()).hexdigest()}

    # patient bridge audit
    mapping = clinical_mapping()  # UCSD-PTGBM-XXXX_YY -> BraTS case id
    union = union_subjects()
    prior = json.loads(PRIOR_INVENTORY.read_text())
    selected_subjects = {"UCSD-PTGBM-" + row["ucsd_subject_id"].split("-")[1] for row in prior["selection"]["selected"]}
    excluded_brats_subjects = {"-".join(c.split("-")[:3]) for c in prior["excluded"]["case_ids"]}
    brats_to_ucsd_subject = {}
    for a, b in mapping.items():
        if b:
            brats_to_ucsd_subject.setdefault("-".join(b.split("-")[:3]), set()).add(a.split("_")[0])
    excluded_ucsd_subjects = set()
    for bs in excluded_brats_subjects:
        excluded_ucsd_subjects.update(brats_to_ucsd_subject.get(bs, set()))
    main_release_subjects = selected_subjects | excluded_ucsd_subjects
    report["prior_main_release_subjects"] = len(main_release_subjects)

    # package subjects resolved to clinical rows
    pkg_subjects = ucsd_subjects if ucsd_subjects else {
        s for bs in brats_subjects_in_names for s in brats_to_ucsd_subject.get(bs, set())}
    per_subject = {}
    for subj in sorted(pkg_subjects):
        rows = {a: b for a, b in mapping.items() if a.startswith(subj + "_")}
        brats_ids = {"-".join(b.split("-")[:3]) for b in rows.values() if b}
        per_subject[subj] = {
            "clinical_rows": len(rows),
            "brats_subjects": sorted(brats_ids),
            "union_overlap": sorted(brats_ids & union),
            "in_prior_main_release": subj in main_release_subjects,
        }
    report["package_subjects"] = len(pkg_subjects)
    report["subjects_with_no_clinical_rows"] = sorted(s for s, v in per_subject.items() if v["clinical_rows"] == 0)
    report["subjects_with_union_overlap"] = sorted(s for s, v in per_subject.items() if v["union_overlap"])
    report["subjects_in_prior_main_release"] = sorted(s for s, v in per_subject.items() if v["in_prior_main_release"])
    report["per_subject"] = per_subject
    report["union_subjects"] = len(union)
    report["clinical_rows_total"] = len(mapping)

    out_path = OUT / "SOURCE_AUDIT.json"
    with out_path.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    concise = {k: v for k, v in report.items() if k not in ("per_subject", "dir_pages", "sums_files", "ucsd_subject_ids_seen", "brats_subject_ids_seen")}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
