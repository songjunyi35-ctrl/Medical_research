#!/usr/bin/env python3
"""Dual-download the frozen 95 BraTS-Africa T1CE/seg pairs in 100-path batches."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import parse_qs, urlparse

import requests

from scripts import brats_africa95_protocol as p


BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/brats-africa/"
PACKAGE_ID = 948
BATCH_SIZE = 100
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"
CONFIG = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf"
ASCP_SHA = "f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe"
KEY_SHA = "9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645"
CONFIG_SHA = "edfcd31fd75b65f0c10b2bf0d06b983ca15f5ffb6adb8eba52be4817df08fadf"


def authenticate() -> tuple[requests.Session, dict[str, str]]:
    session = requests.Session()
    page = session.get(PAGE, timeout=60); page.raise_for_status()
    states = []
    for link in (html.unescape(x) for x in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in x):
        state = parse_qs(urlparse(link).query)["context"][0]
        context = json.loads(base64.b64decode(state))
        if int(context.get("id", -1)) == PACKAGE_ID:
            states.append(state)
    if len(set(states)) != 1:
        raise RuntimeError("unique public package948 context not found")
    state = states[0]
    config = session.get(f"{BASE}/config.js", timeout=60); config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(f"{BASE}/auth/authorize_public_link", params={"response_type": "code", "client_id": client, "redirect_uri": redirect, "state": state}, allow_redirects=False, timeout=60)
    if auth.status_code != 302:
        raise RuntimeError("public authorization failed")
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(f"{BASE}/auth/token", json={"code": query["code"][0], "state": query.get("state", [state])[0], "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect}, timeout=60)
    token.raise_for_status()
    return session, {"Authorization": "Bearer " + token.json()["access_token"]}


def expected(rows: list[dict]) -> dict[str, int]:
    result = {}
    suffix = "/BraTS-Africa/95_Glioma/"
    for row in rows:
        for kind in ("t1ce", "seg"):
            source = row[kind]["path"]
            if suffix not in source:
                raise ValueError("source path outside frozen publisher directory")
            result[source.split(suffix, 1)[1]] = int(row[kind]["bytes"])
    if len(result) != p.FILES or sum(result.values()) != p.BYTES_PER_COPY:
        raise ValueError("expected file set mismatch")
    return result


def chunks(values: list[str]) -> list[list[str]]:
    return [values[i:i + BATCH_SIZE] for i in range(0, len(values), BATCH_SIZE)]


def transfer_copy(name: str, rows: list[dict], sizes: dict[str, int]) -> dict:
    session, headers = authenticate()
    requested = sorted(row[kind]["path"] for row in rows for kind in ("t1ce", "seg"))
    root = p.SOURCE / name; root.mkdir(parents=True, exist_ok=False)
    logs = p.SOURCE / "logs" / name; logs.mkdir(parents=True, exist_ok=False)
    batch_records = []
    suffix = "/BraTS-Africa/95_Glioma/"
    for number, batch in enumerate(chunks(requested)):
        reply = session.post(f"{BASE}/api/v5/packages/{PACKAGE_ID}/transfer_spec/download", headers=headers, params={"transfer_type": "connect", "type": "received"}, json={"paths": [{"path": x, "type": "file"} for x in batch]}, timeout=120)
        if reply.status_code != 201:
            raise RuntimeError(f"transfer spec batch{number} HTTP{reply.status_code}")
        spec = reply.json(); sources = [x["source"] for x in spec.get("paths", [])]
        relative = [x.split(suffix, 1)[1] if suffix in x else "" for x in sources]
        if len(sources) != len(batch) or set(relative) != {x.split(suffix, 1)[1] for x in batch}:
            raise RuntimeError("transfer spec path mismatch")
        prefixes = {x.split(suffix, 1)[0] + "/BraTS-Africa/95_Glioma" for x in sources}
        if len(prefixes) != 1:
            raise RuntimeError("transfer source root mismatch")
        log = logs / f"batch_{number:02d}"; log.mkdir()
        command = [str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"], "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY), "-l", "200m", "--overwrite=never", "--src-base=" + prefixes.pop(), "-v", "-L", str(log), "-f", str(CONFIG), *sources, str(root)]
        if "tags" in spec:
            command[1:1] = ["--tags64=" + base64.b64encode(json.dumps(spec["tags"]).encode()).decode()]
        env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
        run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
        batch_records.append({"batch": number, "requested": len(batch), "returncode": run.returncode, "message": (run.stdout + run.stderr)[-1200:].replace(spec["token"], "[REDACTED]")})
        if run.returncode != 0:
            raise RuntimeError(f"ascp batch{number} returncode{run.returncode}")
        print(f"completed {name} batch {number + 1}/{len(chunks(requested))}", flush=True)
    files = []
    for relative in sorted(sizes):
        path = root / relative
        item = {"relative_path": relative, "path": str(path), "expected_bytes": sizes[relative], "exists": path.is_file()}
        if path.is_file():
            item.update(bytes=path.stat().st_size, sha256=p.sha(path), size_exact=path.stat().st_size == sizes[relative])
        files.append(item)
    return {"copy": name, "batches": batch_records, "files": files, "all_sizes_exact": len(files) == p.FILES and all(x.get("size_exact") for x in files)}


def main() -> None:
    p.documents(); rows = p.cohort_rows()
    p.checked(ASCP, ASCP_SHA); p.checked(KEY, KEY_SHA); p.checked(CONFIG, CONFIG_SHA)
    if not p.SOURCE_PREACCESS.exists():
        raise FileNotFoundError("source preaccess receipt must be frozen first")
    pre = json.loads(p.SOURCE_PREACCESS.read_text())
    if pre.get("status") != "frozen_before_source_payload_and_model_inference" or pre.get("access", {}).get("package_payload") is not False:
        raise ValueError("invalid preaccess receipt")
    if p.SOURCE.exists() or p.SOURCE_DOWNLOAD.exists():
        raise FileExistsError("source download already started")
    sizes = expected(rows); p.SOURCE.mkdir()
    copies = [transfer_copy(name, rows, sizes) for name in ("copy_a", "copy_b")]
    maps = [{x["relative_path"]: x for x in copy["files"]} for copy in copies]
    comparisons = []
    for relative in sorted(sizes):
        a, b = maps[0][relative], maps[1][relative]
        exact = bool(a.get("exists") and b.get("exists") and a.get("size_exact") and b.get("size_exact") and a.get("sha256") and a.get("sha256") == b.get("sha256") and a.get("bytes") == b.get("bytes"))
        comparisons.append({"relative_path": relative, "bytes": a.get("bytes"), "sha256": a.get("sha256"), "copy_exact": exact})
    passed = all(copy["all_sizes_exact"] for copy in copies) and all(x["copy_exact"] for x in comparisons)
    report = {"schema": "aflow.brats_africa95.source_download", "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if passed else "failed", "batch_size_max": BATCH_SIZE, "batches_per_copy": 2, "files_per_copy": p.FILES, "bytes_per_copy": p.BYTES_PER_COPY, "copies": copies, "comparisons": comparisons, "files_copy_exact": sum(x["copy_exact"] for x in comparisons), "cohort_sha256": p.COHORT_SHA, "source_preaccess_sha256": p.sha(p.SOURCE_PREACCESS), "voxel_arrays_accessed": False, "label_values_accessed": False, "model_inference": False, "runner_sha256": p.sha(Path(__file__))}
    p.write_new(p.SOURCE_DOWNLOAD, report)
    print(json.dumps({"status": report["status"], "files_copy_exact": report["files_copy_exact"], "report_sha256": p.sha(p.SOURCE_DOWNLOAD)}, indent=2))
    if not passed:
        raise RuntimeError("dual source gate failed")


if __name__ == "__main__":
    main()
