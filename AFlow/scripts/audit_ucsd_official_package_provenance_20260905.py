"""Inspect TCIA UCSD main package1019; only checksum metadata may be downloaded."""
from __future__ import annotations
import base64
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import parse_qs,urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/"archive/reviews/2026-09-05_30h_chain_audit"
BASE="https://faspex.cancerimagingarchive.net/aspera/faspex"
SDK=Path("/tmp/aflow-rhuh-sdk-FuoemL")


def main():
    s=requests.Session()
    s.mount("https://",HTTPAdapter(max_retries=Retry(total=3,backoff_factor=0.5,allowed_methods=frozenset(("GET","POST")))))
    page=s.get("https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/",timeout=30);page.raise_for_status()
    states=[]
    for link in re.findall(r'https://faspex[^"<>\s]+',page.text):
        state=parse_qs(urlparse(html.unescape(link)).query).get("context",[None])[0]
        if state:
            ctx=json.loads(base64.b64decode(state))
            if str(ctx.get("id"))=="1019":states.append(state)
    assert states and len(set(states))==1
    state=states[0]
    config=s.get(BASE+"/config.js",timeout=30);config.raise_for_status()
    client=re.search(r"client_id:\s*'([^']+)'",config.text).group(1)
    auth=s.get(BASE+"/auth/authorize_public_link",params={"response_type":"code","client_id":client,"redirect_uri":BASE+"/token","state":state},allow_redirects=False,timeout=30)
    query=parse_qs(urlparse(auth.headers["Location"]).query)
    token=s.post(BASE+"/auth/token",json={"code":query["code"][0],"state":query.get("state",[state])[0],"grant_type":"authorization_code","client_id":client,"redirect_uri":BASE+"/token"},timeout=30)
    token.raise_for_status();headers={"Authorization":"Bearer "+token.json()["access_token"]}
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"official_package_id":1019,
            "excluded_challenge_test_package":1221,"new_image_or_label_download":False,"model_inference":False,
            "source_page_sha256":hashlib.sha256(page.content).hexdigest(),"listings":[]}
    def browse(path):
        reply=s.post(BASE+"/api/v5/packages/1019/files/received/page",headers=headers,json={"path":path,"filters":{}},timeout=30)
        reply.raise_for_status();data=reply.json()
        report["listings"].append({"path":path,"data":data})
        return data["items"]
    items=browse("/")
    sums=[x for x in items if x["basename"].endswith(".sums")]
    if not sums:
        for folder in [x for x in items if x["type"]=="directory"]:
            sums.extend(x for x in browse(folder["path"]) if x["basename"].endswith(".sums"))
    report["selected_case006_items"]=browse("/UCSD-PTGBM/UCSD-PTGBM-0006_01")
    report["checksum_files"]=sums
    proof_items=[x for x in report["selected_case006_items"] if x["basename"] in (
        "UCSD-PTGBM-0006_01_T1post.nii.gz","UCSD-PTGBM-0006_01_BraTS_tumor_seg.nii.gz")]
    assert len(proof_items)==2 and sum(int(x["size"]) for x in proof_items)<6_000_000
    proof_dir=ROOT/"workspace/BraTS/ucsd_official_existing_case_proof_20260905"
    proof_dir.mkdir(exist_ok=False)
    preaccess={"purpose":"source identity audit of the first already-evaluated external case, not model evaluation or tuning",
        "official_package_id":1019,"case":"sub-0006_ses-01","files":[{k:x[k] for k in ("path","basename","size","type")} for x in proof_items],
        "no_model_inference":True,"selection_rule":"first case in archived75 selection order","prior_clinical_mapping_verified":True}
    with (RECORD/"UCSD_OFFICIAL_CASE_PROOF_PREACCESS.json").open("x") as f:json.dump(preaccess,f,indent=2,sort_keys=True);f.write("\n")
    reply=s.post(BASE+"/api/v5/packages/1019/transfer_spec/download",headers=headers,params={"type":"received","transfer_type":"connect"},
        json={"paths":[{"path":x["path"],"type":x["type"]} for x in proof_items]},timeout=30)
    reply.raise_for_status();spec=reply.json()
    sources=[x["source"] for x in spec["paths"]]
    assert len(sources)==2 and {Path(x).name for x in sources}=={x["basename"] for x in proof_items}
    env=dict(os.environ,ASPERA_SCP_TOKEN=spec["token"],ASPERA_SCP_COOKIE=spec.get("cookie",""))
    cmd=[str(SDK/"ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"),"--mode=recv","--host="+spec["remote_host"],"--user="+spec["remote_user"],
        "-P",str(spec["ssh_port"]),"-O",str(spec["fasp_port"]),"-i",str(SDK/"ibm-public-client-key.pem"),"-L",str(SDK),"--overwrite=never","-l","20m",
        "--tags64="+base64.b64encode(json.dumps(spec.get("tags",{})).encode()).decode(),*sources,str(proof_dir)]
    transfer=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60)
    report["existing_case_proof_transfer_returncode"]=transfer.returncode
    report["new_image_or_label_download"]=True
    report["new_case_or_model_evaluation"]=False
    report["existing_case_official_files"]=[]
    for item in proof_items:
        path=proof_dir/item["basename"]
        report["existing_case_official_files"].append({"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                                     "matches_official_size":path.stat().st_size==item["size"]})
    if len(sums)==1:
        item=sums[0]
        assert int(item["size"])<5_000_000
        response=s.post(BASE+"/api/v5/packages/1019/transfer_spec/download",headers=headers,params={"type":"received","transfer_type":"connect"},
                        json={"paths":[{"path":item["path"],"type":item["type"]}]},timeout=30)
        response.raise_for_status();spec=response.json()
        sources=[row["source"] for row in spec["paths"]]
        assert len(sources)==1 and Path(sources[0]).name==item["basename"] and sources[0].endswith(".sums")
        output=RECORD/"source_metadata"
        env=dict(os.environ,ASPERA_SCP_TOKEN=spec["token"],ASPERA_SCP_COOKIE=spec.get("cookie",""))
        cmd=[str(SDK/"ibm-aspera-transfer-sdk-linux-amd64-1.1.9/bin/ascp"),"--mode=recv","--host="+spec["remote_host"],"--user="+spec["remote_user"],
             "-P",str(spec["ssh_port"]),"-O",str(spec["fasp_port"]),"-i",str(SDK/"ibm-public-client-key.pem"),"-L",str(SDK),"--overwrite=never","-l","20m",
             "--tags64="+base64.b64encode(json.dumps(spec.get("tags",{})).encode()).decode(),sources[0],str(output)]
        transfer=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60)
        report["transfer_returncode"]=transfer.returncode
        report["transfer_message"]=(transfer.stdout+transfer.stderr)[-1200:].replace(spec["token"],"[REDACTED]")
        checksum_path=output/item["basename"]
        if checksum_path.is_file():
            report["official_sums"]={"path":str(checksum_path.relative_to(ROOT)),"bytes":checksum_path.stat().st_size,
                                     "sha256":hashlib.sha256(checksum_path.read_bytes()).hexdigest()}
    with (RECORD/"UCSD_OFFICIAL_SOURCE_AUDIT_V3.json").open("x") as f:json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("listings","selected_case006_items")},indent=2))


if __name__=="__main__":main()
