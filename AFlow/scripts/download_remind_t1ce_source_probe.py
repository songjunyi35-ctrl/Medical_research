"""Download only frozen ReMIND-002 DICOM and two NRRDs for source probing."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlparse
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "archive/research_records/2026-09-05_remind_t1ce_source_probe"
WORK = Path("/tmp/aflow-remind-probe-20260905")
PREACCESS_SHA = "cffc281338e43b3e25e854b2f229276e7c7a39b2b891537524c67875d276452b"
COHORT_SHA = "8118827aa001dead98cc8cc1f6ee5a2fa9b797a6a8797329ba22f4766d669536"
AMENDMENT_SHA = "38d47cd213ba6d4623e648f4f91e003b02508ce708f247db3978e7629c09253e"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")
ASCP = SDK / "ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"
KEY = SDK / "ibm-public-client-key.pem"
ASCP_SHA = "f59e7d6afae896f320084a5d441caaece48a1303e5afd014cf885a74bb329efe"
KEY_SHA = "9983bd387eebf27190c754de76a0ae09d5c39597ee224c3974a6bdb8faec8645"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
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
        raise ValueError("wrong ReMIND package context")
    config = session.get(BASE + "/config.js", timeout=30)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'", config.text).group(1)
    auth = session.get(
        BASE + "/auth/authorize_public_link",
        params={"response_type": "code", "client_id": client, "redirect_uri": BASE + "/token", "state": state},
        allow_redirects=False, timeout=30,
    )
    if auth.status_code != 302:
        raise RuntimeError("ReMIND public authorization failed")
    query = parse_qs(urlparse(auth.headers["Location"]).query)
    token = session.post(
        BASE + "/auth/token",
        json={"code": query["code"][0], "state": query.get("state", [state])[0],
              "grant_type": "authorization_code", "client_id": client, "redirect_uri": BASE + "/token"},
        timeout=30,
    )
    token.raise_for_status()
    return session, {"Authorization": "Bearer " + token.json()["access_token"]}


def download_nrrds(paths: list[str], destination: Path) -> dict:
    session, headers = authenticate()
    reply = session.post(
        BASE + "/api/v5/packages/755/transfer_spec/download",
        headers=headers,
        params={"type": "received", "transfer_type": "connect"},
        json={"paths": [{"path": path, "type": "symbolic_link"} for path in paths]},
        timeout=30,
    )
    reply.raise_for_status()
    spec = reply.json()
    sources = [item["source"] for item in spec["paths"]]
    if len(sources) != 2 or {Path(source).name for source in sources} != {Path(path).name for path in paths}:
        raise ValueError("ReMIND NRRD transfer spec differs from frozen paths")
    destination.mkdir()
    env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    command = [
        str(ASCP), "--mode=recv", "--host=" + spec["remote_host"], "--user=" + spec["remote_user"],
        "-P", str(spec["ssh_port"]), "-O", str(spec["fasp_port"]), "-i", str(KEY),
        "-L", str(SDK), "--overwrite=never", "-l", "20m", *sources, str(destination),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    files = []
    for path in paths:
        local = destination / Path(path).name
        files.append({"official_path": path, "path": str(local), "bytes": local.stat().st_size if local.exists() else None,
                      "sha256": sha(local) if local.exists() else None})
    return {"returncode": result.returncode, "files": files,
            "message": (result.stdout + result.stderr)[-1000:].replace(spec["token"], "[REDACTED]")}


def main() -> None:
    from scripts.freeze_remind_t1ce_source_probe import check_implementation

    check_implementation()
    for path, expected in ((RECORD / "PROBE_PREACCESS.json", PREACCESS_SHA),
                           (RECORD / "COHORT_CANDIDATES.json", COHORT_SHA),
                           (RECORD / "AMENDMENT_01.md", AMENDMENT_SHA),
                           (ASCP, ASCP_SHA), (KEY, KEY_SHA)):
        if sha(path) != expected:
            raise ValueError(f"frozen ReMIND probe input changed: {path}")
    if WORK.exists():
        raise FileExistsError("ReMIND probe work directory already exists")
    if shutil.disk_usage(WORK.parent).free < 2 * 1024 ** 3:
        raise RuntimeError("less than 2GiB /tmp reserve")
    WORK.mkdir()
    preaccess = json.loads((RECORD / "PROBE_PREACCESS.json").read_text())
    probe = preaccess["probe"]
    if probe["case_id"] != "ReMIND-002":
        raise ValueError("wrong fixed ReMIND probe")
    series = probe["t1ce"]
    zip_path = WORK / "ReMIND-002-t1post.zip"
    url = "https://nbia.cancerimagingarchive.net/nbia-api/services/v1/getImage"
    with requests.get(url, params={"SeriesInstanceUID": series["SeriesInstanceUID"]}, stream=True, timeout=180) as reply:
        reply.raise_for_status()
        headers = {key: value for key, value in reply.headers.items()
                   if key.lower() in ("content-type", "content-length", "content-disposition", "etag")}
        with zip_path.open("xb") as handle:
            for block in reply.iter_content(8 << 20):
                if block:
                    handle.write(block)
    dicom = WORK / "dicom"
    dicom.mkdir()
    with zipfile.ZipFile(zip_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if any(Path(item.filename).is_absolute() or ".." in Path(item.filename).parts for item in members):
            raise ValueError("unsafe NBIA zip member")
        archive.extractall(dicom)
    paths = [probe["manual_tumor"]["official_path"], probe["automatic_cerebrum"]["official_path"]]
    first = download_nrrds(paths, WORK / "nrrd_a")
    second = download_nrrds(paths, WORK / "nrrd_b")
    by_name_a = {Path(item["official_path"]).name: item for item in first["files"]}
    by_name_b = {Path(item["official_path"]).name: item for item in second["files"]}
    expected_sizes = {Path(probe["manual_tumor"]["official_path"]).name: probe["manual_tumor"]["bytes"],
                      Path(probe["automatic_cerebrum"]["official_path"]).name: probe["automatic_cerebrum"]["bytes"]}
    repeat = {}
    for name in sorted(by_name_a):
        a, b = by_name_a[name], by_name_b[name]
        repeat[name] = {
            "expected_browse_bytes": expected_sizes[name],
            "bytes_a": a["bytes"], "bytes_b": b["bytes"],
            "sha256_a": a["sha256"], "sha256_b": b["sha256"],
            "passed": a["bytes"] == b["bytes"] == expected_sizes[name] and a["sha256"] == b["sha256"] and a["bytes"] > 0,
        }
    uncompressed_bytes = sum(item.file_size for item in members)
    report = {
        "schema": "aflow.remind_t1ce_source_probe.download",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "downloaded_for_source_probe",
        "probe": "ReMIND-002",
        "dicom": {"zip_path": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha(zip_path),
                  "zip_members": len(members), "uncompressed_bytes": uncompressed_bytes,
                  "expected_uncompressed_bytes": series["FileSize"], "response_headers": headers, "extract_dir": str(dicom)},
        "nrrd_transfer_a": first,
        "nrrd_transfer_b": second,
        "nrrd_repeat": repeat,
        "all_download_gates_passed": len(members) == 192 and uncompressed_bytes == series["FileSize"]
                                     and first["returncode"] == second["returncode"] == 0
                                     and all(item["passed"] for item in repeat.values()),
        "publisher_checksum_caveat": "official sums contains empty-file MD5 for all356 rows; not used as content hash",
        "preaccess_sha256": PREACCESS_SHA,
        "cohort_sha256": COHORT_SHA,
        "amendment_sha256": AMENDMENT_SHA,
        "other_candidate_pixels": False,
        "model_inference": False,
        "runner_sha256": sha(Path(__file__)),
    }
    with (RECORD / "DOWNLOAD_RECEIPT.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("nrrd_transfer_a", "nrrd_transfer_b")}, indent=2))
    if not report["all_download_gates_passed"]:
        raise RuntimeError("ReMIND probe download gates failed")


if __name__ == "__main__":
    main()
