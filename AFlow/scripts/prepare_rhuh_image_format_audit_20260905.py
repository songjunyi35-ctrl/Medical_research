"""Freeze RHUH80 identity and audit image format only, before any label access."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
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
RECORD = ROOT / "archive/reviews/2026-09-05_30h_chain_audit"
WORK = ROOT / "workspace/BraTS/rhuh_gbm_source_20260905"
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
SDK = Path("/tmp/aflow-rhuh-sdk-FuoemL")


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def freeze():
    source = RECORD / "RHUH_PUBLIC_SOURCE_AUDIT_V7.json"
    sums = RECORD / "source_metadata/RHUH-GBM-nii-v1.sums"
    assert digest(source) == "7b50b80e1d317241037fbf974db1361e08e3ba9360f54bfcad1d46bd73abef09"
    assert digest(sums) == "7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130"
    file_hashes = {}
    for line in sums.read_text().splitlines():
        md5, relative = line.split(maxsplit=1)
        assert re.fullmatch(r"[0-9a-f]{32}", md5)
        relative = relative.lstrip("*")
        assert relative not in file_hashes
        file_hashes[relative] = md5
    files = [row for row in json.loads(source.read_text())["official_files"] if row["path"].endswith(".nii.gz")]
    assert len(files) == len(file_hashes) == 720
    assert {row["path"].lstrip("/") for row in files} == set(file_hashes)
    by_study = {}
    for row in files:
        match = re.fullmatch(r"/RHUH-GBM_nii_v1/(RHUH-\d{4})/([012])/([^/]+)", row["path"])
        assert match is not None
        subject, tp, name = match.groups()
        if tp == "0":
            continue
        key = subject + "_" + tp
        kind = "image" if name.endswith("_t1ce.nii.gz") else "label" if "segmentation" in name else None
        if kind:
            entry = by_study.setdefault(key, {"case_id": key, "subject_id": subject, "timepoint":int(tp)})
            assert kind not in entry
            entry[kind] = {"official_path":row["path"], "basename": name, "bytes":row["size"],
                           "official_md5":file_hashes[row["path"].lstrip("/")]}
    rows = [by_study[key] for key in sorted(by_study)]
    assert len(rows) == 80 and all("image" in row and "label" in row for row in rows)
    subjects = {row["subject_id"] for row in rows}
    clinical = ROOT / "archive/reviews/2026-09-02_rhuh_gbm_external_feasibility/source_metadata/clinical_data_TCIA_RHUH-GBM.csv"
    assert digest(clinical) == "32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728"
    with clinical.open() as handle:
        assert subjects == {row["Patient ID"] for row in csv.DictReader(handle)}
    receipt = {"status":"frozen_before_any_rhuh_image_or_label_voxel_download", "created_utc":datetime.now(timezone.utc).isoformat(),
        "cases":80, "subjects":40, "timepoints":[1,2], "selected":rows,
        "source_inventory_sha256":digest(source), "official_sums_sha256":digest(sums),
        "source_paper":"https://arxiv.org/html/2305.00005v2",
        "brats_source_paper":"https://arxiv.org/html/2405.18368v1",
        "institution_provenance":"RHUH Spain is absent from the seven USA/Germany BraTS2024 contributing institutions; no patient ID bridge exists, so namespace disjointness alone is not a patient-identity proof",
        "image_bytes":sum(row["image"]["bytes"] for row in rows), "label_bytes":sum(row["label"]["bytes"] for row in rows),
        "authorized_current_action":"download only these80 T1ce files from official package, verify official MD5, compute SHA256 and format QC; no segmentation download, no model, no intensity transform selection",
        "input_format_gates":{"finite":True,"nonnegative":True,"nonempty_positive_support":True,"dimensions":3,"strict_positive_spacing":True},
        "stop_rule":"any image format failure blocks unchanged historical positive-only N4 pipeline; no case removal and no result-dependent preprocessing",
        "access":{"rhuh_images":False,"rhuh_labels":False,"model_inference":False,"new_evaluation_metrics":False},
        "runner_sha256":digest(Path(__file__))}
    save_new(RECORD / "RHUH_IMAGE_QC_PREACCESS.json", receipt)
    print(json.dumps({k:receipt[k] for k in ("status","cases","subjects","image_bytes","label_bytes")},indent=2))


def authenticate():
    session = requests.Session()
    page = session.get("https://www.cancerimagingarchive.net/collection/rhuh-gbm/", timeout=30)
    page.raise_for_status()
    link = html.unescape(next(x for x in re.findall(r'https://faspex[^"<>\s]+',page.text) if "context=" in x))
    state = parse_qs(urlparse(link).query)["context"][0]
    context = json.loads(base64.b64decode(state))
    assert context["id"] == "684" and context["type"] == "external_download_package"
    config = session.get(BASE+"/config.js",timeout=30)
    config.raise_for_status()
    client = re.search(r"client_id:\s*'([^']+)'",config.text).group(1)
    params = {"response_type":"code","client_id":client,"redirect_uri":BASE+"/token","state":state}
    reply = session.get(BASE+"/auth/authorize_public_link",params=params,allow_redirects=False,timeout=30)
    assert reply.status_code == 302
    query = parse_qs(urlparse(reply.headers["Location"]).query)
    token = session.post(BASE+"/auth/token",json={"code":query["code"][0],"state":query.get("state",[state])[0],
        "grant_type":"authorization_code","client_id":client,"redirect_uri":BASE+"/token"},timeout=30)
    token.raise_for_status()
    return session, {"Authorization":"Bearer "+token.json()["access_token"]}


def download():
    receipt_path = RECORD / "RHUH_IMAGE_QC_PREACCESS.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["runner_sha256"] == digest(Path(__file__))
    raw = WORK / "raw_images"
    raw.mkdir(parents=True,exist_ok=False)
    session, headers = authenticate()
    spec_reply = session.post(BASE+"/api/v5/packages/684/transfer_spec/download",headers=headers,
        params={"type":"received","transfer_type":"connect"},json={"paths":[{"path":row["image"]["official_path"],"type":"symbolic_link"} for row in receipt["selected"]]},timeout=30)
    spec_reply.raise_for_status()
    spec = spec_reply.json()
    sources = [row["source"] for row in spec["paths"]]
    assert len(sources) == 80 and all(x.endswith("_t1ce.nii.gz") for x in sources)
    assert {Path(x).name for x in sources} == {row["image"]["basename"] for row in receipt["selected"]}
    env = dict(os.environ, ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie",""))
    cmd = [str(SDK/"ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"),"--mode=recv","--host="+spec["remote_host"],
        "--user="+spec["remote_user"],"-P",str(spec["ssh_port"]),"-O",str(spec["fasp_port"]),"-i",str(SDK/"ibm-public-client-key.pem"),
        "-L",str(SDK),"--overwrite=never","-l","100m","--tags64="+base64.b64encode(json.dumps(spec.get("tags",{})).encode()).decode(),*sources,str(raw)]
    result = subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
    report = {"created_utc":datetime.now(timezone.utc).isoformat(),"preaccess_sha256":digest(receipt_path),"returncode":result.returncode,
        "message":(result.stdout+result.stderr)[-2000:].replace(spec["token"],"[REDACTED]"),"label_download":False,"model_inference":False,"files":[]}
    for row in receipt["selected"]:
        path = raw / row["image"]["basename"]
        item = {"case_id":row["case_id"],"path":str(path),"exists":path.is_file()}
        if path.is_file():
            item.update(bytes=path.stat().st_size,md5=digest(path,"md5"),sha256=digest(path))
            item["official_match"] = item["md5"] == row["image"]["official_md5"] and item["bytes"] == row["image"]["bytes"]
        report["files"].append(item)
    report["all_official_match"] = all(item.get("official_match") for item in report["files"])
    save_new(RECORD/"RHUH_IMAGE_DOWNLOAD.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2))
    if not report["all_official_match"]:
        raise RuntimeError("image download failed; no format QC authorized")


def qc():
    import nibabel as nib
    import numpy as np
    receipt = json.loads((RECORD/"RHUH_IMAGE_QC_PREACCESS.json").read_text())
    download_receipt = json.loads((RECORD/"RHUH_IMAGE_DOWNLOAD.json").read_text())
    assert receipt["runner_sha256"] == digest(Path(__file__)) and download_receipt["all_official_match"]
    rows=[]
    for item in download_receipt["files"]:
        path=Path(item["path"])
        assert digest(path)==item["sha256"]
        nii=nib.load(str(path))
        arr=np.asanyarray(nii.dataobj)
        finite=bool(np.isfinite(arr).all())
        negative=int(np.count_nonzero(arr<0))
        positive=int(np.count_nonzero(arr>0))
        spacing=[float(x) for x in nib.affines.voxel_sizes(nii.affine)]
        ok=finite and negative==0 and positive>0 and arr.ndim==3 and all(x>0 for x in spacing)
        rows.append({"case_id":item["case_id"],"shape":list(nii.shape),"dtype":str(nii.get_data_dtype()),
            "axis_codes":list(nib.aff2axcodes(nii.affine)),"spacing":spacing,"min":float(arr.min()),"max":float(arr.max()),
            "negative_voxels":negative,"positive_voxels":positive,"finite":finite,"format_gate_passed":ok})
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"cases":80,"subjects":40,"per_case":rows,
        "passed":all(row["format_gate_passed"] for row in rows),"negative_image_count":sum(row["negative_voxels"]>0 for row in rows),
        "labels_downloaded":False,"label_voxel_access":False,"model_inference":False,"new_evaluation_metrics":False,
        "preaccess_sha256":digest(RECORD/"RHUH_IMAGE_QC_PREACCESS.json"),"download_receipt_sha256":digest(RECORD/"RHUH_IMAGE_DOWNLOAD.json")}
    save_new(RECORD/"RHUH_IMAGE_FORMAT_QC.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="per_case"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","download","qc"))
    {"freeze":freeze,"download":download,"qc":qc}[parser.parse_args().action]()
