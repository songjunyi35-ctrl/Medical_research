"""Download frozen UTSW362 in fixed 100-path Aspera batches."""
import base64
from datetime import datetime,timezone
import json,os
from pathlib import Path
import subprocess
from scripts import download_utsw_no_n4_paired_external as shared
from scripts import utsw_frozen_baseline_external_protocol as p

BATCH_SIZE=100

def chunks(items):return [items[i:i+BATCH_SIZE] for i in range(0,len(items),BATCH_SIZE)]

def transfer_copy(name,rows,expected):
    session,headers=shared.authenticate();requested=sorted(path for row in rows for path in (row["image_official_path"],row["label_official_path"]));batches=[]
    root=p.SOURCE/name;root.mkdir(parents=True,exist_ok=False);logs=p.SOURCE/"logs"/name;logs.mkdir(parents=True,exist_ok=False)
    for number,batch in enumerate(chunks(requested)):
        reply=session.post(shared.BASE+"/api/v5/packages/1265/transfer_spec/download",headers=headers,
            params={"transfer_type":"connect","type":"received"},json={"paths":[{"path":x,"type":"file"} for x in batch]},timeout=120)
        if reply.status_code!=201:raise RuntimeError(f"spec batch{number} HTTP{reply.status_code}")
        spec=reply.json();sources=[x["source"] for x in spec.get("paths",[])];suffix="/UTSW-Glioma/";relative=[x.split(suffix,1)[1] if suffix in x else "" for x in sources]
        expected_relative={x.split(suffix,1)[1] for x in batch}
        if len(sources)!=len(batch) or set(relative)!=expected_relative:raise RuntimeError("batch spec mismatch")
        prefixes={x.split(suffix,1)[0]+"/UTSW-Glioma" for x in sources}
        if len(prefixes)!=1:raise RuntimeError("batch root mismatch")
        log=logs/f"batch_{number:02d}";log.mkdir();command=[str(shared.ASCP),"--mode=recv","--host="+spec["remote_host"],"--user="+spec["remote_user"],
            "-P",str(spec["ssh_port"]),"-O",str(spec["fasp_port"]),"-i",str(shared.KEY),"-l","200m","--overwrite=never",
            "--src-base="+prefixes.pop(),"-v","-L",str(log),"-f",str(shared.CONFIG),*sources,str(root)]
        if "tags" in spec:command[1:1]=["--tags64="+base64.b64encode(json.dumps(spec["tags"]).encode()).decode()]
        env=dict(os.environ,ASPERA_SCP_TOKEN=spec["token"],ASPERA_SCP_COOKIE=spec.get("cookie",""));run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=900)
        batches.append({"batch":number,"requested":len(batch),"returncode":run.returncode,"message":(run.stdout+run.stderr)[-1200:].replace(spec["token"],"[REDACTED]")})
        if run.returncode!=0:raise RuntimeError(f"ascp batch{number} returncode{run.returncode}")
        print(f"completed {name} batch {number+1}/{len(chunks(requested))}",flush=True)
    files=[]
    for rel in sorted(expected):
        path=root/rel;item={"relative_path":rel,"path":str(path),"expected_bytes":expected[rel],"exists":path.is_file()}
        if path.is_file():item.update(bytes=path.stat().st_size,sha256=shared.file_sha(path),size_exact=path.stat().st_size==expected[rel])
        files.append(item)
    return {"copy":name,"batches":batches,"files":files,"all_sizes_exact":len(files)==724 and all(x.get("size_exact") for x in files)}

def main():
    from scripts.freeze_utsw_frozen_baseline_external import check_implementation
    check_implementation();p.verify_preaccess();smoke=json.loads((p.RECORD/"GPU_SMOKE_RECEIPT.json").read_text())
    if smoke.get("status")!="passed" or smoke.get("preaccess_sha256")!=p.sha(p.PREACCESS):raise ValueError("smoke")
    if not p.three_hour_gate():raise RuntimeError("time gate")
    shared.p.checked(shared.ASCP,shared.ASCP_SHA);shared.p.checked(shared.KEY,shared.KEY_SHA);shared.p.checked(shared.CONFIG,shared.CONFIG_SHA)
    if p.SOURCE.exists() or p.SOURCE_DOWNLOAD.exists():raise FileExistsError("v2 source already started")
    rows=p.cohort();expected=shared.expected_paths(rows);p.SOURCE.mkdir();copies=[transfer_copy(x,rows,expected) for x in ("copy_a","copy_b")];maps=[{x["relative_path"]:x for x in c["files"]} for c in copies];comparisons=[]
    for rel in sorted(expected):
        a,b=maps[0][rel],maps[1][rel];exact=bool(a.get("exists") and b.get("exists") and a.get("size_exact") and b.get("size_exact") and
            a.get("sha256") and b.get("sha256") and a["bytes"]==b["bytes"] and a["sha256"]==b["sha256"])
        comparisons.append({"relative_path":rel,"bytes":a.get("bytes"),"sha256":a.get("sha256"),"copy_exact":exact})
    passed=all(c["all_sizes_exact"] for c in copies) and all(x["copy_exact"] for x in comparisons)
    report={"schema":"aflow.utsw_frozen_baseline_external.source_download_amendment_01","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"passed" if passed else "failed","batch_size_max":100,"batches_per_copy":8,"files_per_copy":724,"bytes_per_copy":p.SOURCE_BYTES,
        "copies":copies,"comparisons":comparisons,"files_copy_exact":sum(x["copy_exact"] for x in comparisons),"preaccess_sha256":p.sha(p.PREACCESS),
        "voxel_arrays_accessed":False,"model_inference":False,"new_subjects_downloaded":359,"runner_sha256":p.sha(Path(__file__))}
    p.write_new(p.SOURCE_DOWNLOAD,report);print(json.dumps({"status":report["status"],"files_exact":report["files_copy_exact"],"sha256":p.sha(p.SOURCE_DOWNLOAD)},indent=2))
    if not passed:raise RuntimeError("v2 download failed")
if __name__=="__main__":main()
