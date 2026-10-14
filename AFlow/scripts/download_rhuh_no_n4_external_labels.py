"""Download only the frozen RHUH80 segmentations after checkpoint-bound preaccess."""
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

from scripts import rhuh_no_n4_external_protocol as p

BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"
ASCP_SHA = "f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe"
KEY_SHA = "9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645"


def authenticate() -> tuple[requests.Session, dict[str, str]]:
    session = requests.Session()
    page = session.get("https://www.cancerimagingarchive.net/collection/rhuh-gbm/", timeout=30)
    page.raise_for_status()
    link = html.unescape(next(value for value in re.findall(r'https://faspex[^"<>\s]+', page.text) if "context=" in value))
    state = parse_qs(urlparse(link).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if context.get("id") != "684" or context.get("type") != "external_download_package":
        raise ValueError("official public link is not RHUH package684")
    config = session.get(BASE + "/config.js", timeout=30)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    params = {"response_type": "code", "client_id": client, "redirect_uri": BASE + "/token", "state": state}
    authorization = session.get(BASE + "/auth/authorize_public_link", params=params, allow_redirects=False, timeout=30)
    if authorization.status_code != 302:
        raise RuntimeError("RHUH public authorization code failed")
    query = parse_qs(urlparse(authorization.headers["Location"]).query)
    token = session.post(
        BASE + "/auth/token",
        json={"code": query["code"][0], "state": query.get("state", [state])[0],
              "grant_type": "authorization_code", "client_id": client, "redirect_uri": BASE + "/token"},
        timeout=30,
    )
    token.raise_for_status()
    return session, {"Authorization": "Bearer " + token.json()["access_token"]}


def main() -> None:
    from scripts.freeze_rhuh_no_n4_external import check_implementation

    check_implementation()
    preaccess = p.verify_preaccess()
    if not p.two_hour_gate():
        raise RuntimeError("two-hour gate expired before RHUH label download")
    p.checked(ASCP, ASCP_SHA)
    p.checked(KEY, KEY_SHA)
    inventory = json.loads((p.REVIEW / "RHUH_LABEL_INVENTORY_READINESS.json").read_text())
    rows = inventory["rows"]
    if len(rows) != 80 or not all(row["inventory_exact"] for row in rows):
        raise ValueError("RHUH label inventory is not exact")
    output = p.SOURCE_WORK / "raw_labels"
    output.mkdir(parents=True, exist_ok=False)
    session, headers = authenticate()
    reply = session.post(
        BASE + "/api/v5/packages/684/transfer_spec/download",
        headers=headers,
        params={"type": "received", "transfer_type": "connect"},
        json={"paths": [{"path": row["official_path"], "type": "symbolic_link"} for row in rows]},
        timeout=30,
    )
    reply.raise_for_status()
    spec = reply.json()
    sources = [item["source"] for item in spec["paths"]]
    if len(sources) != 80 or {Path(source).name for source in sources} != {row["basename"] for row in rows}:
        raise ValueError("transfer spec differs from frozen RHUH label paths")
    env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    command = [
        str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
        "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY),
        "-L", str(SDK), "--overwrite=never", "-l", "20m",
        "--tags64=" + base64.b64encode(json.dumps(spec.get("tags", {})).encode()).decode(),
        *sources, str(output),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    files = []
    for row in rows:
        path = output / row["basename"]
        item = {"case_id": row["case_id"], "path": str(path), "exists": path.is_file()}
        if path.is_file():
            item.update(bytes=path.stat().st_size, md5=__import__("hashlib").md5(path.read_bytes()).hexdigest(), sha256=p.sha(path))
            item["official_match"] = item["bytes"] == row["bytes"] and item["md5"] == row["official_md5"]
        files.append(item)
    report = {
        "schema": "aflow.rhuh_no_n4_external.label_download",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": result.returncode,
        "message": (result.stdout + result.stderr)[-2000:].replace(spec["token"], "[REDACTED]"),
        "files": files,
        "all_official_match": result.returncode == 0 and all(item.get("official_match") for item in files),
        "preaccess_sha256": p.sha(p.PREACCESS),
        "inventory_sha256": p.LABEL_INVENTORY_SHA,
        "label_voxel_access": False,
        "model_inference": False,
        "runner_sha256": p.sha(Path(__file__)),
    }
    p.write_new(p.RECORD / "LABEL_DOWNLOAD.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("files", "message")}, indent=2))
    if not report["all_official_match"]:
        raise RuntimeError("RHUH official label transfer failed")


if __name__ == "__main__":
    main()
