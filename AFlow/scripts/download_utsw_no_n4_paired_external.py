"""Dual-download the frozen UTSW362 image/label pairs without voxel access."""
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
from urllib.parse import parse_qs, urlparse

import requests

from scripts import utsw_no_n4_paired_external_protocol as p


BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/utsw-glioma/"
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"
CONFIG = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf"
ASCP_SHA = "f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe"
KEY_SHA = "9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645"
CONFIG_SHA = "edfcd31fd75b65f0c10b2bf0d06b983ca15f5ffb6adb8eba52be4817df08fadf"


def authenticate() -> tuple[requests.Session, dict[str, str]]:
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    link = html.unescape(next(value for value in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in value))
    state = parse_qs(urlparse(link).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if int(context["id"]) != 1265 or context["type"] != "external_download_package":
        raise RuntimeError("unexpected UTSW public package")
    config = session.get(f"{BASE}/config.js", timeout=60)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(f"{BASE}/auth/authorize_public_link", params={
        "response_type": "code", "client_id": client, "redirect_uri": redirect, "state": state,
    }, allow_redirects=False, timeout=60)
    if auth.status_code != 302:
        raise RuntimeError("UTSW public authorization failed")
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(f"{BASE}/auth/token", json={
        "code": query["code"][0], "state": query.get("state", [state])[0],
        "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect,
    }, timeout=60)
    token.raise_for_status()
    return session, {"Authorization": "Bearer " + token.json()["access_token"]}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_paths(rows: list[dict]) -> dict[str, int]:
    values = {}
    for row in rows:
        values[f"{row['subject_id']}/brain_t1ce.nii.gz"] = row["image_bytes"]
        values[f"{row['subject_id']}/rtumorseg_manual_correction.nii.gz"] = row["label_bytes"]
    if len(values) != p.SOURCE_FILES or sum(values.values()) != p.SOURCE_BYTES:
        raise ValueError("UTSW expected relative source mismatch")
    return values


def transfer_copy(copy_name: str, rows: list[dict], expected: dict[str, int]) -> dict:
    session, headers = authenticate()
    requested = [
        path for row in rows for path in (row["image_official_path"], row["label_official_path"])
    ]
    reply = session.post(
        f"{BASE}/api/v5/packages/1265/transfer_spec/download",
        headers=headers,
        params={"transfer_type": "connect", "type": "received"},
        json={"paths": [{"path": path, "type": "file"} for path in requested]},
        timeout=120,
    )
    if reply.status_code != 201:
        raise RuntimeError(f"UTSW transfer spec failed: {copy_name}/{reply.status_code}")
    spec = reply.json()
    sources = [item["source"] for item in spec.get("paths", [])]
    if len(sources) != p.SOURCE_FILES:
        raise RuntimeError("UTSW transfer spec file count mismatch")
    suffix = "/UTSW-Glioma/"
    relative = [source.split(suffix, 1)[1] if suffix in source else "" for source in sources]
    if set(relative) != set(expected):
        raise RuntimeError("UTSW transfer spec path set mismatch")
    prefixes = {source.split(suffix, 1)[0] + "/UTSW-Glioma" for source in sources}
    if len(prefixes) != 1:
        raise RuntimeError("UTSW transfer spec source root mismatch")
    destination = p.SOURCE / copy_name
    destination.mkdir(parents=True, exist_ok=False)
    log_dir = p.SOURCE / "logs" / copy_name
    log_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
        "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY), "-l", "200m",
        "--overwrite=never", "--src-base=" + prefixes.pop(), "-v", "-L", str(log_dir), "-f", str(CONFIG),
        *sources, str(destination),
    ]
    if "tags" in spec:
        command[1:1] = ["--tags64=" + base64.b64encode(json.dumps(spec["tags"]).encode()).decode()]
    environment = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    transfer = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=1800)
    files = []
    for name in sorted(expected):
        path = destination / name
        item = {"relative_path": name, "path": str(path), "expected_bytes": expected[name], "exists": path.is_file()}
        if path.is_file():
            item.update(bytes=path.stat().st_size, sha256=file_sha(path))
            item["size_exact"] = item["bytes"] == item["expected_bytes"]
        files.append(item)
    passed = transfer.returncode == 0 and len(files) == p.SOURCE_FILES and all(item.get("size_exact") for item in files)
    return {
        "copy": copy_name, "returncode": transfer.returncode,
        "message": (transfer.stdout + transfer.stderr)[-2000:].replace(spec["token"], "[REDACTED]"),
        "files": files, "all_sizes_exact": passed,
    }


def main() -> None:
    from scripts.freeze_utsw_no_n4_paired_external import check_implementation

    check_implementation()
    preaccess = p.verify_preaccess()
    if not p.four_hour_gate():
        raise RuntimeError("UTSW four-hour gate expired before source download")
    p.checked(ASCP, ASCP_SHA); p.checked(KEY, KEY_SHA); p.checked(CONFIG, CONFIG_SHA)
    smoke = json.loads((p.RECORD / "GPU_SMOKE_RECEIPT.json").read_text())
    if smoke.get("status") != "passed" or smoke.get("preaccess_sha256") != p.sha(p.PREACCESS):
        raise ValueError("UTSW paired synthetic smoke did not pass")
    if p.SOURCE.exists():
        raise FileExistsError(p.SOURCE)
    rows = p.cohort()
    expected = expected_paths(rows)
    p.SOURCE.mkdir()
    copies = [transfer_copy(name, rows, expected) for name in ("copy_a", "copy_b")]
    by_copy = [{item["relative_path"]: item for item in copy["files"]} for copy in copies]
    comparisons = []
    for name in sorted(expected):
        a, b = by_copy[0][name], by_copy[1][name]
        comparisons.append({
            "relative_path": name, "bytes": a.get("bytes"), "sha256": a.get("sha256"),
            "copy_exact": a.get("bytes") == b.get("bytes") and a.get("sha256") == b.get("sha256"),
        })
    passed = all(copy["all_sizes_exact"] for copy in copies) and all(item["copy_exact"] for item in comparisons)
    report = {
        "schema": "aflow.utsw_no_n4_paired_external.source_download",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "files_per_copy": p.SOURCE_FILES, "bytes_per_copy": p.SOURCE_BYTES,
        "copies": copies, "comparisons": comparisons,
        "files_copy_exact": sum(item["copy_exact"] for item in comparisons),
        "preaccess_sha256": p.sha(p.PREACCESS),
        "voxel_arrays_accessed": False, "model_inference": False,
        "new_subjects_downloaded": 359,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "SOURCE_DOWNLOAD.json", report)
    print(json.dumps({
        "status": report["status"], "files_copy_exact": report["files_copy_exact"],
        "expected": p.SOURCE_FILES, "report_sha256": p.sha(p.RECORD / "SOURCE_DOWNLOAD.json"),
    }, indent=2))
    if not passed:
        raise RuntimeError("UTSW dual source download gate failed before voxel access")


if __name__ == "__main__":
    main()
