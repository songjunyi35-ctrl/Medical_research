"""Supervise the four two-GPU waves; never launch evaluation automatically."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import no_n4_clean_protocol as p
from scripts.freeze_no_n4_clean import check_implementation,main as freeze


def main(phase):
    check_implementation();p.docs()
    if phase=="formal":
        smoke=json.loads((p.RECORD/"GPU_SMOKE_RECEIPT.json").read_text())
        assert smoke["all_pairing_gates_passed"] and smoke["status"]=="passed_smoke_pairing"
    logfile_dir=p.WORK/phase/"driver_logs";logfile_dir.mkdir(parents=True,exist_ok=False)
    deadline=datetime.fromisoformat("2026-09-06T14:02:39+00:00")
    receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,"deadline_utc":deadline.isoformat(),
        "implementation_sha256":p.sha(p.RECORD/"IMPLEMENTATION.json"),"cache_receipt_sha256":p.sha(p.RECORD/"CACHE_RECEIPT.json"),
        "waves":[[f"e_{r}",f"f_{r}"] for r in ("history","ema","canonical","no_n4")],"automatic_evaluation":False}
    p.write_new(p.RECORD/(phase.upper()+"_LAUNCH_RECEIPT.json"),receipt)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",
             MPLCONFIGDIR="/tmp/aflow_matplotlib_cache")
    completed=[]
    for role in ("history","ema","canonical","no_n4"):
        children=[];handles=[]
        try:
            for gpu,fold in enumerate(("e","f")):
                path=logfile_dir/f"{fold}_{role}.log";handle=path.open("x");handles.append(handle)
                command=[sys.executable,str(ROOT/"scripts/train_no_n4_clean.py"),"--fold",fold,"--role",role,"--device",f"cuda:{gpu}"]
                if phase=="smoke":command.append("--smoke")
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                children.append((fold,child));print(f"started {phase} {fold}/{role} pid={child.pid}",flush=True)
            while any(child.poll() is None for _,child in children):
                if datetime.now(timezone.utc)>=deadline:raise RuntimeError("30h window reached")
                if any(child.poll() not in (None,0) for _,child in children):raise RuntimeError("child failed; inspect driver logs")
                time.sleep(3)
            if any(child.returncode!=0 for _,child in children):raise RuntimeError("child failed")
            for fold,_ in children:
                report=json.loads((p.WORK/phase/f"{fold}_{role}"/"TRAIN_DONE.json").read_text())
                if not report["all_gates_passed"]:raise RuntimeError("training gates failed")
                completed.append(f"{fold}_{role}")
            print(f"completed wave {phase}/{role}",flush=True)
        finally:
            for _,child in children:
                if child.poll() is None:child.terminate()
            for _,child in children:
                if child.poll() is None:
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
            for handle in handles:handle.close()
    if phase=="smoke":freeze("smoke")
    p.write_new(p.RECORD/(phase.upper()+"_RUNNER_DONE.json"),{"status":"complete","completed":completed,"new_inner_evaluation":False,
        "finished_utc":datetime.now(timezone.utc).isoformat()})
    print(f"{phase} complete; evaluation remains unlaunched",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("phase",choices=("smoke","formal"));main(ap.parse_args().phase)
