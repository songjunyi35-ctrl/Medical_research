"""Read official public RHUH package metadata; never download imaging or log tokens."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
PAGE = "https://www.cancerimagingarchive.net/collection/rhuh-gbm/"


def main():
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "imaging_downloaded": False, "label_voxels_accessed": False,
              "model_inference": False, "probes": []}
    session = requests.Session()
    page = session.get(PAGE, timeout=30)
    page.raise_for_status()
    report["collection_page_sha256"] = hashlib.sha256(page.content).hexdigest()
    links = re.findall(r'https://faspex[^"<>\s]+', page.text)
    public_url = html.unescape(next(x for x in links if "context=" in x))
    state = parse_qs(urlparse(public_url).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    if str(context.get("id")) != "684" or context.get("type") != "external_download_package":
        raise RuntimeError("unexpected official package identity")
    report["package_id"] = 684
    config = session.get(BASE + "/config.js", timeout=30)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    redirect = BASE + "/token"
    # Official frontend tc() replaces /api/v5 with /auth.
    auth = session.get(BASE + "/auth/authorize_public_link", params={
        "response_type": "code", "client_id": client,
        "redirect_uri": redirect, "state": state}, allow_redirects=False, timeout=30)
    report["authorize_http"] = auth.status_code
    query = parse_qs(urlparse(auth.headers.get("Location", "")).query)
    if "code" not in query:
        report["status"] = "blocked_before_public_authorization_code"
    else:
        token = session.post(BASE + "/auth/token", json={
            "code": query["code"][0], "state": query.get("state", [state])[0],
            "grant_type": "authorization_code", "client_id": client,
            "redirect_uri": redirect}, timeout=30)
        report["token_http"] = token.status_code
        data = token.json()
        bearer = data.get("access_token")
        if not bearer:
            report["status"] = "blocked_before_public_bearer"
        else:
            payload = bearer.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            report["public_scope"] = claims.get("scope")
            headers = {"Authorization": "Bearer " + bearer}
            for path in ("/packages/684", "/configuration"):
                reply = session.get(BASE + "/api/v5" + path, headers=headers, timeout=30)
                item = {"path": path, "http_status": reply.status_code,
                        "response_sha256": hashlib.sha256(reply.content).hexdigest()}
                try:
                    body = reply.json()
                except ValueError:
                    body = {}
                # Error bodies contain no temporary transfer or OAuth credentials.
                if reply.status_code >= 400:
                    item["error"] = body
                else:
                    item["top_level_keys"] = sorted(body) if isinstance(body, dict) else []
                    if path == "/packages/684":
                        report["package"] = {key: body.get(key) for key in (
                            "id", "title", "total_bytes", "total_files", "files_on_server",
                            "pagination_browsing", "prevent_http_download", "download_availability")}
                    else:
                        report["http_gateway_url"] = body.get("http_gateway_url")
                report["probes"].append(item)
            if report["probes"][0]["http_status"] == 200:
                reply = session.post(BASE + "/api/v5/packages/684/files/received/page", headers=headers,
                                     json={"path": "/", "filters": {}}, timeout=30)
                report["root_browse_http"] = reply.status_code
                report["root_browse"] = reply.json()
                def browse(path):
                    reader = requests.Session()
                    reader.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5, allowed_methods=frozenset(("POST",)))))
                    response = reader.post(BASE + "/api/v5/packages/684/files/received/page", headers=headers,
                                            json={"path": path, "filters": {}}, timeout=30)
                    response.raise_for_status()
                    value = response.json()
                    if len(value.get("items", [])) != value.get("total_count", len(value.get("items", []))):
                        raise RuntimeError("incomplete file listing; pagination must be resolved")
                    return value["items"]
                prior = OUT / "RHUH_PUBLIC_SOURCE_AUDIT_V7.json"
                report["reused_official_inventory_sha256"] = hashlib.sha256(prior.read_bytes()).hexdigest()
                pending = []
                files = json.loads(prior.read_text())["official_files"]
                while pending:
                    if len(pending) > 200:
                        raise RuntimeError("unexpected directory count")
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        listings = list(pool.map(browse, pending))
                    pending = []
                    for listing in listings:
                        for item in listing:
                            if item["type"] == "directory":
                                pending.append(item["path"])
                            elif item["type"] == "file" or (item["type"] == "symbolic_link" and item.get("target", {}).get("type") == "file"):
                                files.append({k: item.get(k) for k in ("path", "basename", "type", "size", "mtime", "checksum", "sha256", "md5")})
                            else:
                                raise RuntimeError("unsupported official file type")
                    print(json.dumps({"browse_pending_directories":len(pending), "files_listed":len(files)}), flush=True)
                report["official_files"] = sorted(files, key=lambda x: x["path"])
                report["official_file_count"] = len(files)
                report["official_file_bytes"] = sum(int(x["size"]) for x in files)
                spec_reply = session.post(BASE + "/api/v5/packages/684/transfer_spec/download", headers=headers,
                    params={"transfer_type": "connect", "type": "received"},
                    json={"paths": [{"path": "/RHUH-GBM-nii-v1.sums", "type": "file"}]}, timeout=30)
                report["sums_transfer_spec_http"] = spec_reply.status_code
                spec = spec_reply.json()
                report["sums_transfer_spec_keys"] = sorted(spec)
                if spec_reply.status_code >= 400:
                    report["sums_transfer_error"] = spec
                report["sums_transfer_transport"] = {key: spec.get(key) for key in (
                    "remote_host", "remote_user", "ssh_port", "fasp_port", "http_fallback", "http_fallback_port", "paths")}
                if spec_reply.status_code == 201:
                    sdk = Path("/tmp/aflow-rhuh-sdk-FuoemL")
                    destination = OUT / "source_metadata"
                    destination.mkdir(exist_ok=True)
                    sources = [item["source"] for item in spec["paths"]]
                    if len(sources) != 1 or not sources[0].endswith("/RHUH-GBM-nii-v1.sums"):
                        raise RuntimeError("refusing any non-sums download")
                    # Official transfer/spec.schema.yaml maps token to ASPERA_SCP_TOKEN.
                    env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
                    binary = sdk / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
                    command = [str(binary), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
                               "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]),
                               "-i", str(sdk / "ibm-public-client-key.pem"), "-l", "20m", "--overwrite=never", "-v", "-L", str(sdk),
                               "-f", str(sdk / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/etc/aspera.conf"), sources[0], str(destination)]
                    if "tags" in spec:
                        command[1:1] = ["--tags64=" + base64.b64encode(json.dumps(spec["tags"]).encode()).decode()]
                    try:
                        transfer = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
                        report["sums_transfer_returncode"] = transfer.returncode
                        report["sums_transfer_message"] = (transfer.stdout + transfer.stderr)[-1800:].replace(spec["token"], "[REDACTED]")
                    except subprocess.TimeoutExpired:
                        report["sums_transfer_status"] = "timeout45s"
                    sums = destination / "RHUH-GBM-nii-v1.sums"
                    if sums.is_file():
                        report["official_sums"] = {"path":str(sums.relative_to(ROOT)), "bytes":sums.stat().st_size,
                                                   "sha256":hashlib.sha256(sums.read_bytes()).hexdigest()}
            report["status"] = ("public_package_metadata_available" if report["probes"][0]["http_status"] == 200
                                else "blocked_before_official_inventory")
    destination = OUT / "RHUH_PUBLIC_SOURCE_AUDIT_V9.json"
    with destination.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("official_files", "root_browse", "probes", "sample_leaf_items")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
