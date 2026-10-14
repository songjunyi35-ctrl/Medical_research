"""Direct TCIA provenance audit for all already-evaluated75 cases; no model calls."""
from __future__ import annotations
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs,urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/"archive/reviews/2026-09-05_30h_chain_audit"
WORK=ROOT/"workspace/BraTS/ucsd_official_source_verification_20260905"
BASE="https://faspex.cancerimagingarchive.net/aspera/faspex"
SDK=Path("/tmp/aflow-rhuh-sdk-FuoemL")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def session():
    s=requests.Session();s.mount("https://",HTTPAdapter(max_retries=Retry(total=3,backoff_factor=.5,allowed_methods=frozenset(("GET","POST")))))
    return s


def write_new(path,data):
    with path.open("x") as f:json.dump(data,f,indent=2,sort_keys=True);f.write("\n")


def main():
    old_path=ROOT/"archive/research_records/2026-09-02_ucsd_ptgbm_external_confirmation/OFFICIAL_ASSET_INVENTORY.json"
    assert sha(old_path)=="6cb1f5a5eb740c06c6e8b01e57c7e6aab6dd8a7472f02d8acdd42540e2d16c38"
    old=json.loads(old_path.read_text())["selection"]["selected"]
    assert len(old)==75
    clinical=json.loads((RECORD/"UCSD_OFFICIAL_CLINICAL_MAPPING_AUDIT.json").read_text());assert clinical["status"]=="passed"
    s=session();page=s.get("https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/",timeout=30);page.raise_for_status()
    states=[]
    for url in re.findall(r'https://faspex[^"<>\s]+',page.text):
        state=parse_qs(urlparse(html.unescape(url)).query).get("context",[None])[0]
        if state and str(json.loads(base64.b64decode(state)).get("id"))=="1019":states.append(state)
    assert states and len(set(states))==1
    state=states[0];config=s.get(BASE+"/config.js",timeout=30);config.raise_for_status()
    client=re.search(r"client_id:\s*'([^']+)'",config.text).group(1)
    auth=s.get(BASE+"/auth/authorize_public_link",params={"response_type":"code","client_id":client,"redirect_uri":BASE+"/token","state":state},allow_redirects=False,timeout=30)
    query=parse_qs(urlparse(auth.headers["Location"]).query)
    response=s.post(BASE+"/auth/token",json={"code":query["code"][0],"state":query.get("state",[state])[0],"grant_type":"authorization_code","client_id":client,"redirect_uri":BASE+"/token"},timeout=30)
    response.raise_for_status();headers={"Authorization":"Bearer "+response.json()["access_token"]}
    def inventory(row):
        m=re.fullmatch(r"sub-(\d{4})_ses-(\d{2})",row["ucsd_case_id"]);assert m
        official_id="UCSD-PTGBM-"+m[1]+"_"+m[2]
        reply=session().post(BASE+"/api/v5/packages/1019/files/received/page",headers=headers,
            json={"path":"/UCSD-PTGBM/"+official_id,"filters":{}},timeout=30)
        reply.raise_for_status();items=reply.json()["items"]
        assert len(items)<100  # Each case contains <100 files; never accept a truncated page.
        found={}
        for kind,suffix in (("image","_T1post.nii.gz"),("label","_BraTS_tumor_seg.nii.gz")):
            matches=[x for x in items if x["basename"]==official_id+suffix];assert len(matches)==1
            found[kind]={k:matches[0][k] for k in ("path","basename","size","type")}
        return {"ucsd_case_id":row["ucsd_case_id"],"official_id":official_id,"brats_case_id":row["brats_case_id"],"files":found}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inventory,old))
    total=sum(int(item["size"]) for row in rows for item in row["files"].values())
    assert total<1_000_000_000
    if shutil.disk_usage(ROOT).free<total+2*1024**3:raise RuntimeError("source audit disk reserve insufficient")
    preaccess={"created_utc":datetime.now(timezone.utc).isoformat(),"purpose":"official source identity of exactly the frozen already-evaluated UCSD75; no prediction or parameter selection",
        "existing_selection_sha256":sha(old_path),"cases":75,"subjects":53,"official_package_id":1019,"challenge_package1221_used":False,
        "official_bytes":total,"rows":rows,"runner_sha256":sha(Path(__file__)),"model_inference":False,
        "new_patients":False,"old_report_and_cache_unchanged":True}
    write_new(RECORD/"UCSD_ALL75_DIRECT_PREACCESS.json",preaccess)
    raw=WORK/"raw";raw.mkdir(parents=True,exist_ok=False)
    items=[item for row in rows for item in row["files"].values()]
    reply=s.post(BASE+"/api/v5/packages/1019/transfer_spec/download",headers=headers,params={"type":"received","transfer_type":"connect"},
        json={"paths":[{"path":item["path"],"type":item["type"]} for item in items]},timeout=30)
    reply.raise_for_status();spec=reply.json();sources=[x["source"] for x in spec["paths"]]
    assert len(sources)==150 and {Path(x).name for x in sources}=={x["basename"] for x in items}
    env=dict(os.environ,ASPERA_SCP_TOKEN=spec["token"],ASPERA_SCP_COOKIE=spec.get("cookie",""))
    cmd=[str(SDK/"ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"),"--mode=recv","--host="+spec["remote_host"],"--user="+spec["remote_user"],
        "-P",str(spec["ssh_port"]),"-O",str(spec["fasp_port"]),"-i",str(SDK/"ibm-public-client-key.pem"),"-L",str(SDK),"--overwrite=never","-l","100m",
        "--tags64="+base64.b64encode(json.dumps(spec.get("tags",{})).encode()).decode(),*sources,str(raw)]
    print(f"frozen150 official files, bytes={total}; downloading source proof only",flush=True)
    transfer=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
    receipt={"returncode":transfer.returncode,"message":(transfer.stdout+transfer.stderr)[-1200:].replace(spec["token"],"[REDACTED]"),"files":[]}
    for item in items:
        path=raw/item["basename"]
        receipt["files"].append({"path":str(path),"exists":path.is_file(),"bytes":path.stat().st_size if path.is_file() else 0,
                                 "sha256":sha(path) if path.is_file() else None,"official_size":item["size"]})
    receipt["all_files_present_and_size_matched"]=all(x["exists"] and x["bytes"]==x["official_size"] for x in receipt["files"])
    write_new(RECORD/"UCSD_ALL75_DIRECT_DOWNLOAD.json",receipt)
    if not receipt["all_files_present_and_size_matched"]:raise RuntimeError("download failed, preserve partials")
    import nibabel as nib
    import numpy as np
    mirrors=json.loads((ROOT/"workspace/BraTS/ucsd_ptgbm_external_confirmation_20260902/raw/DOWNLOAD_RECEIPT.json").read_text())
    by_case={row["ucsd_case_id"]:row for row in mirrors["records"]}
    comparisons=[]
    for row in rows:
        output={"case_id":row["ucsd_case_id"],"files":{}}
        for kind in ("image","label"):
            official_path=raw/row["files"][kind]["basename"]
            mirror=by_case[row["ucsd_case_id"]]["files"][kind]
            assert sha(mirror["path"])==mirror["sha256"]
            a=nib.load(str(official_path));b=nib.load(mirror["path"])
            x=np.asanyarray(a.dataobj);y=np.asanyarray(b.dataobj)
            same_geometry=x.shape==y.shape and bool(np.allclose(a.affine,b.affine,rtol=0,atol=1e-5))
            result={"official_sha256":sha(official_path),"mirror_sha256":mirror["sha256"],"official_dtype":str(a.get_data_dtype()),"mirror_dtype":str(b.get_data_dtype()),
                    "official_shape":list(x.shape),"mirror_shape":list(y.shape),"same_geometry":same_geometry,
                    "array_equal":bool(np.array_equal(x,y)),"official_range":[float(x.min()),float(x.max())],"mirror_range":[float(y.min()),float(y.max())]}
            if same_geometry:
                result["unequal_voxels"]=int(np.count_nonzero(x!=y))
                if kind=="image":
                    selected=(x!=0)|(y!=0);result["source_nonzero_to_mirror_zero"]=int(np.count_nonzero((x!=0)&(y==0)))
                    result["source_zero_to_mirror_nonzero"]=int(np.count_nonzero((x==0)&(y!=0)))
                    result["nonzero_correlation"]=float(np.corrcoef(x[selected],y[selected])[0,1])
            output["files"][kind]=result
        comparisons.append(output)
        if len(comparisons)%10==0:print(f"source identity compared {len(comparisons)}/75",flush=True)
    report={"status":"completed_source_identity_audit","cases":75,"subjects":53,"comparisons":comparisons,
        "official_image_dtype_counts":dict(Counter(r["files"]["image"]["official_dtype"] for r in comparisons)),
        "mirror_image_dtype_counts":dict(Counter(r["files"]["image"]["mirror_dtype"] for r in comparisons)),
        "image_arrays_equal_count":sum(r["files"]["image"]["array_equal"] for r in comparisons),
        "label_arrays_equal_count":sum(r["files"]["label"]["array_equal"] for r in comparisons),
        "all_geometry_equal":all(v["same_geometry"] for r in comparisons for v in r["files"].values()),
        "preaccess_sha256":sha(RECORD/"UCSD_ALL75_DIRECT_PREACCESS.json"),"download_receipt_sha256":sha(RECORD/"UCSD_ALL75_DIRECT_DOWNLOAD.json"),
        "clinical_mapping_sha256":sha(RECORD/"UCSD_OFFICIAL_CLINICAL_MAPPING_AUDIT.json"),"original_model_reports_modified":False,
        "model_inference":False,"prediction_metric_recomputation":False,"parameter_or_case_selection":False,
        "interpretation":"quantify mirror-vs-official representation; no original-float source benchmark was run and existing uint8 benchmark must be labeled as such"}
    write_new(RECORD/"UCSD_ALL75_DIRECT_SOURCE_COMPARISON.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="comparisons"},indent=2))


if __name__=="__main__":main()
