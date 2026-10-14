"""Download only ReMIND package755's checksum manifest; never request NRRD."""
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

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/reviews/2026-09-05_remind_source_audit"
SOURCE = RECORD / "REMIND_PUBLIC_SOURCE.json"
SOURCE_SHA = "b2f281023dda7ba2fc95767a00f7af67e66a945f158e8f5b1c9bb9ccce2f2b9a"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"


def sha(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate() -> tuple[requests.Session, dict[str, str]]:
    session = requests.Session()
    page = session.get("https://www.cancerimagingarchive.net/collection/remind/", timeout=30)
    page.raise_for_status()
    states = {}
    for link in (html.unescape(value) for value in re.findall(r'https://faspex[^"<>\s]+', page.text)):
        values = parse_qs(urlparse(link).query).get("context", [])
        if not values:
            continue
        try:
            context = json.loads(base64.b64decode(values[0]))
        except (ValueError, json.JSONDecodeError):
            continue
        if str(context.get("id")) == "755":
            states[values[0]] = context
    if len(states) != 1:
        raise ValueError("expected one unique ReMIND package755 context")
    state, context = next(iter(states.items()))
    if context.get("type") != "external_download_package":
        raise ValueError("wrong ReMIND public-link type")
    config = session.get(BASE + "/config.js", timeout=30)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    authorization = session.get(
        BASE + "/auth/authorize_public_link",
        params={"response_type": "code", "client_id": client, "redirect_uri": BASE + "/token", "state": state},
        allow_redirects=False, timeout=30,
    )
    if authorization.status_code != 302:
        raise RuntimeError("ReMIND public auth failed")
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
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("ReMIND source audit changed")
    source = json.loads(SOURCE.read_text())
    rows = source["segmentation_package"]["files"]
    candidates = [row for row in rows if row["path"] == "/ReMIND_NRRD_Segs_September_2023.sums"]
    if len(candidates) != 1 or candidates[0]["size"] != 45831 or candidates[0]["type"] != "file":
        raise ValueError("official ReMIND sums identity mismatch")
    output = RECORD / "source_metadata/ReMIND_NRRD_Segs_September_2023.sums"
    if output.exists():
        raise FileExistsError("ReMIND sums already downloaded")
    session, headers = authenticate()
    reply = session.post(
        BASE + "/api/v5/packages/755/transfer_spec/download",
        headers=headers,
        params={"type": "received", "transfer_type": "connect"},
        json={"paths": [{"path": candidates[0]["path"], "type": "file"}]},
        timeout=30,
    )
    reply.raise_for_status()
    spec = reply.json()
    sources = [row["source"] for row in spec["paths"]]
    if len(sources) != 1 or not sources[0].endswith("ReMIND_NRRD_Segs_September_2023.sums"):
        raise ValueError("transfer spec requested a non-sums file")
    env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    command = [
        str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
        "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY),
        "-L", str(SDK), "--overwrite=never", "-l", "20m", sources[0], str(output.parent),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    if not output.is_file():
        raise RuntimeError("ReMIND sums transfer produced no file")
    parsed = {}
    for line in output.read_text().splitlines():
        checksum, path = line.split(maxsplit=1)
        path = path.lstrip("*")
        if not re.fullmatch(r"[0-9a-f]{32}", checksum) or path in parsed:
            raise ValueError("invalid/duplicate ReMIND checksum row")
        parsed[path] = checksum
    listed_nrrd = {row["path"].lstrip("/") for row in rows if row["basename"].endswith(".nrrd")}
    sums_nrrd = {path for path in parsed if path.endswith(".nrrd")}
    report = {
        "schema": "aflow.remind.sums_download",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "official_sums_downloaded_metadata_only",
        "file": {"path": str(output.relative_to(ROOT)), "bytes": output.stat().st_size, "sha256": sha(output)},
        "rows": len(parsed),
        "nrrd_rows": len(sums_nrrd),
        "browse_nrrd_rows": len(listed_nrrd),
        "sums_minus_browse_count": len(sums_nrrd - listed_nrrd),
        "browse_minus_sums_count": len(listed_nrrd - sums_nrrd),
        "sums_minus_browse_sample": sorted(sums_nrrd - listed_nrrd)[:20],
        "browse_minus_sums_sample": sorted(listed_nrrd - sums_nrrd)[:20],
        "transfer_returncode": result.returncode,
        "source_audit_sha256": SOURCE_SHA,
        "nrrd_downloaded": False,
        "segmentation_voxel_access": False,
        "model_inference": False,
        "runner_sha256": sha(Path(__file__)),
    }
    with (RECORD / "REMIND_SUMS_DOWNLOAD.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
