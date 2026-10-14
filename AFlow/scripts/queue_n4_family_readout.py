"""Wait for all ten frozen family models, then perform the one-time E/F readout."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import no_n4_clean_protocol as shared

OTSU=ROOT/"archive/research_records/2026-09-05_n4_otsu_clean_replication"


def main():
    deadline=datetime.fromisoformat("2026-09-06T14:02:39+00:00")
    queued={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"waiting_without_cuda_allocation",
        "requirements":["original8models_complete","otsu2models_complete","both_gpus_idle","all_cache_bytes_rehashed","both_preaccess_receipts_frozen"],
        "only_cohorts":"E42/17,F42/19","old_cohort_model_access":False,"raw_or_protocol_tuning":False,
        "wrapper_sha256":shared.sha(Path(__file__)),"archiver_sha256":shared.sha(ROOT/"scripts/archive_n4_family_stage_a.py")}
    shared.write_new(shared.RECORD/"FAMILY_READOUT_QUEUE.json",queued)
    print("queued family readout: waiting for all10models; no CUDA or inner access",flush=True)
    done_paths=(shared.RECORD/"FORMAL_RUNNER_DONE.json",OTSU/"FORMAL_RUNNER_DONE.json")
    while not all(path.exists() for path in done_paths):
        if datetime.now(timezone.utc)>=deadline:raise RuntimeError("window ended before family completion")
        time.sleep(30)
    assert all(json.loads(path.read_text())["status"]=="complete" for path in done_paths)
    if (deadline-datetime.now(timezone.utc)).total_seconds()<1800:raise RuntimeError("insufficient readout/archive reserve")
    idle_count=0
    while idle_count<3:
        reply=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True)
        rows=[tuple(int(v.strip()) for v in line.split(",")) for line in reply.stdout.splitlines() if line.strip()]
        idle=len(rows)==2 and all(memory<=512 and utilization==0 for _,memory,utilization in rows)
        idle_count=idle_count+1 if idle else 0
        if datetime.now(timezone.utc)>=deadline:raise RuntimeError("window ended at idle gate")
        if idle_count<3:time.sleep(10)
    for record in (shared.RECORD,OTSU):
        if (record/"INNER_STARTED.json").exists() or (record/"RAW_INNER.json").exists():raise FileExistsError("family already read out")
    caches=[shared.CONTROL_CACHE]
    for record in (shared.RECORD,OTSU):
        receipt=json.loads((record/"CACHE_RECEIPT.json").read_text());assert receipt["status"]=="passed"
        caches.append(Path(receipt["candidate_cache"]))
    checks=[]
    for cache in caches:
        index=json.loads((cache/"index.json").read_text())
        for filename,key in (("images.f32","images_sha256"),("labels.u8","labels_sha256")):
            digest=shared.sha(cache/filename)
            if digest!=index[key]:raise RuntimeError(f"cache bytes changed: {cache}/{filename}")
            checks.append({"path":str(cache/filename),"sha256":digest})
    shared.write_new(shared.RECORD/"FAMILY_CACHE_PREACCESS_REHASH.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"files":checks,"all_exact":True,
        "gpu_idle_rows":rows,"inner_model_access":False,"byte_hashing_only":True})
    commands=[
        ["scripts/freeze_n4_otsu_clean.py","inner"],
        ["scripts/freeze_no_n4_clean.py","inner"],
        ["scripts/evaluate_no_n4_clean.py","--device","cuda:0"],
        ["scripts/evaluate_n4_otsu_clean.py","--device","cuda:0"],
        ["scripts/archive_n4_family_stage_a.py","no_n4"],
        ["scripts/archive_n4_family_stage_a.py","otsu"],
    ]
    for command in commands:
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    shared.write_new(shared.RECORD/"FAMILY_READOUT_DONE.json",{"finished_utc":datetime.now(timezone.utc).isoformat(),"status":"raw_and_results_archived_pending_canonical_interpretation",
        "no_n4_raw_sha256":shared.sha(shared.RECORD/"RAW_INNER.json"),"otsu_raw_sha256":shared.sha(OTSU/"RAW_INNER.json"),"old_cohort_evaluation":False})
    print("family readout/archival completed; main agent must update canonical interpretation and decide conditional final stage",flush=True)


if __name__=="__main__":main()
