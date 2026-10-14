"""Verify one Faspex transfer can preserve two already-accessed subject paths."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import parse_qs, urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/utsw-glioma/"
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"
CONFIG = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf"
DESTINATION = Path("/tmp/aflow-utsw-src-base-probe-20260905")
REQUESTED = (
    "/UTSW-Glioma/BT0001/brain_t1ce.nii.gz",
    "/UTSW-Glioma/BT0002/brain_t1ce.nii.gz",
)
EXPECTED = {
    "BT0001/brain_t1ce.nii.gz": "dd8cc29f568461607fbf034a37c309586a7fb9b8af754e5369a95fdd04a8051e",
    "BT0002/brain_t1ce.nii.gz": "fc7cda13d959308fb70ccf0b3dbec807ccfa90440dfbfb55ac3d9077829a913d",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    session = requests.Session()
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    link = html.unescape(next(value for value in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in value))
    state = parse_qs(urlparse(link).query)["context"][0]
    config_js = session.get(f"{BASE}/config.js", timeout=60)
    config_js.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config_js.text).group(1)
    redirect = f"{BASE}/token"
    auth = session.get(f"{BASE}/auth/authorize_public_link", params={
        "response_type": "code", "client_id": client, "redirect_uri": redirect, "state": state,
    }, allow_redirects=False, timeout=60)
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token_reply = session.post(f"{BASE}/auth/token", json={
        "code": query["code"][0], "state": query.get("state", [state])[0],
        "grant_type": "authorization_code", "client_id": client, "redirect_uri": redirect,
    }, timeout=60)
    token_reply.raise_for_status()
    bearer = token_reply.json()["access_token"]
    spec_reply = session.post(
        f"{BASE}/api/v5/packages/1265/transfer_spec/download",
        headers={"Authorization": "Bearer " + bearer},
        params={"transfer_type": "connect", "type": "received"},
        json={"paths": [{"path": path, "type": "file"} for path in REQUESTED]},
        timeout=60,
    )
    if spec_reply.status_code != 201:
        raise RuntimeError(f"transfer spec HTTP {spec_reply.status_code}")
    spec = spec_reply.json()
    sources = [row["source"] for row in spec["paths"]]
    suffix = "/UTSW-Glioma/"
    prefixes = {source.split(suffix, 1)[0] + "/UTSW-Glioma" for source in sources if suffix in source}
    if len(sources) != 2 or len(prefixes) != 1:
        raise RuntimeError("unexpected multi-path source roots")
    source_base = prefixes.pop()
    DESTINATION.mkdir()
    log_dir = DESTINATION / "logs"
    log_dir.mkdir()
    command = [
        str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
        "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY), "-l", "50m",
        "--overwrite=never", "--src-base=" + source_base, "-v", "-L", str(log_dir), "-f", str(CONFIG),
        *sources, str(DESTINATION),
    ]
    if "tags" in spec:
        command[1:1] = ["--tags64=" + base64.b64encode(json.dumps(spec["tags"]).encode()).decode()]
    environment = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    transfer = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    if transfer.returncode != 0:
        message = (transfer.stdout + transfer.stderr).replace(spec["token"], "[REDACTED]")[-1800:]
        raise RuntimeError("ascp failed: " + message)
    actual = {}
    for relative, expected in EXPECTED.items():
        path = DESTINATION / relative
        if not path.is_file():
            raise RuntimeError(f"missing preserved path {relative}")
        actual[relative] = sha(path)
        if actual[relative] != expected:
            raise RuntimeError(f"content mismatch {relative}")
    report = {
        "status": "passed",
        "requested": list(REQUESTED),
        "preserved_relative_paths": sorted(actual),
        "sha256": actual,
        "single_transfer": True,
        "new_subjects_accessed": False,
        "label_voxels_accessed": False,
        "model_inference": False,
    }
    (DESTINATION / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
