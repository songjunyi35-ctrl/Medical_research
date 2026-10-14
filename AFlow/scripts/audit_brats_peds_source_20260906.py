"""Read-only inventory audit of the TCIA BraTS-PEDs packages (no pixel download).

Lists every faspex package linked from the collection page, walks the file tree
of each, and reports per-case file kinds so a future frozen cohort can be fixed
before any pixel access. Downloads nothing except directory listings.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/brats-peds/"
OUT = ROOT / "archive/reviews/2026-09-06_brats_peds_source_audit"


def authorize(session, state):
    config = session.get(f"{BASE}/config.js", timeout=60)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(f"{BASE}/auth/authorize_public_link",
                       params={"response_type": "code", "client_id": client, "redirect_uri": redirect, "state": state},
                       allow_redirects=False, timeout=60)
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(f"{BASE}/auth/token",
                         json={"code": query["code"][0], "state": query.get("state", [state])[0],
                               "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect},
                         timeout=60)
    token.raise_for_status()
    return {"Authorization": "Bearer " + token.json()["access_token"]}


def list_directory(session, headers, package_id, path):
    items, token = [], None
    while True:
        params = {"iteration_token": token} if token else None
        reply = session.post(f"{BASE}/api/v5/packages/{package_id}/files/received/page",
                             headers=headers, params=params, json={"path": path, "filters": {}}, timeout=60)
        reply.raise_for_status()
        body = reply.json()
        items.extend(body.get("items", []))
        token = reply.headers.get("x-aspera-next-iteration-token")
        if not token:
            return items


def walk(session, headers, package_id, max_dirs=4000):
    inventory = []
    queue = ["/"]
    seen = 0
    while queue and seen < max_dirs:
        path = queue.pop(0)
        seen += 1
        for item in list_directory(session, headers, package_id, path):
            row = {k: item.get(k) for k in ("path", "basename", "type", "size")}
            inventory.append(row)
            if item.get("type") == "directory":
                queue.append(item["path"])
    return inventory, not queue


def main():
    OUT.mkdir(exist_ok=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1.0, allowed_methods=frozenset(("GET", "POST")))))
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    packages = {}
    for link in re.findall(r'https://faspex[^"<>\s]+', page.text):
        state = parse_qs(urlparse(html.unescape(link)).query).get("context", [None])[0]
        if state:
            ctx = json.loads(base64.b64decode(state))
            packages[str(ctx.get("id"))] = state
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "collection_page_sha256": hashlib.sha256(page.content).hexdigest(),
              "package_ids": sorted(packages), "packages": {},
              "pixels_downloaded": False, "model_access": False}
    for pid, state in sorted(packages.items()):
        headers = authorize(session, state)
        meta = session.get(f"{BASE}/api/v5/packages/{pid}", headers=headers, timeout=60)
        meta.raise_for_status()
        inventory, complete = walk(session, headers, pid)
        files = [r for r in inventory if r["type"] in ("file", "symbolic_link")]
        kinds = collections.Counter()
        cases = collections.defaultdict(set)
        for r in files:
            parts = [p for p in r["path"].split("/") if p]
            case = parts[-2] if len(parts) >= 2 else "?"
            kind = re.sub(r".*?(-|_)(?=[A-Za-z]+\.nii)", "", r["basename"]) if r["basename"].endswith(".nii.gz") else r["basename"]
            kind = re.sub(r"^.*?((t1c|t1n|t2w|t2f|seg|t1ce|t1|t2|flair|tumor_seg)[\w-]*\.nii\.gz)$", r"\1", r["basename"], flags=re.I)
            kinds[kind if kind != r["basename"] or not r["basename"].endswith('.nii.gz') else r["basename"]] += 1
            cases[case].add(kind)
        seg_cases = [c for c, ks in cases.items() if any('seg' in k.lower() for k in ks)]
        report["packages"][pid] = {
            "title": meta.json().get("title"),
            "walk_complete": complete,
            "rows": len(inventory),
            "files": len(files),
            "bytes": sum(int(r["size"] or 0) for r in files),
            "directories": sum(1 for r in inventory if r["type"] == "directory"),
            "basename_kind_counts": dict(kinds.most_common(30)),
            "case_directories": len(cases),
            "cases_with_seg": len(seg_cases),
            "sums_files": [r for r in files if r["basename"].endswith(".sums")],
            "sample_paths": [r["path"] for r in files[:8]],
        }
        inv_path = OUT / f"PACKAGE{pid}_INVENTORY.json"
        with inv_path.open("x") as f:
            json.dump({"package_id": pid, "inventory": inventory}, f, indent=1, sort_keys=True)
            f.write("\n")
        report["packages"][pid]["inventory_sha256"] = hashlib.sha256(inv_path.read_bytes()).hexdigest()
    out = OUT / "SOURCE_AUDIT.json"
    with out.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    concise = {pid: {k: v for k, v in info.items() if k not in ("basename_kind_counts", "sample_paths", "sums_files")}
               for pid, info in report["packages"].items()}
    print(json.dumps({"package_ids": report["package_ids"], "summary": concise}, indent=2))
    for pid, info in report["packages"].items():
        print(pid, json.dumps(info["basename_kind_counts"], indent=1)[:800])
        print(pid, "sample:", info["sample_paths"][:4])


if __name__ == "__main__":
    main()
