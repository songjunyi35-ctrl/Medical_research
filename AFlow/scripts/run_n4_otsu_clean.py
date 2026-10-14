"""Supervise Otsu students; formal waits for shared original-family training."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import n4_otsu_protocol as p
from scripts.freeze_n4_otsu_clean import check_implementation,main as freeze


def main(phase):
    check_implementation();deadline=datetime.fromisoformat("2026-09-06T14:02:39+00:00")
    if phase=="formal":
        smoke=json.loads((p.RECORD/"GPU_SMOKE_RECEIPT.json").read_text());assert smoke["all_pairing_gates_passed"]
        print("waiting for shared original-family formal training completion",flush=True)
        while not (p.SHARED_RECORD/"FORMAL_RUNNER_DONE.json").exists():
            if datetime.now(timezone.utc)>=deadline:raise RuntimeError("30h window reached while waiting")
            time.sleep(30)
        assert json.loads((p.SHARED_RECORD/"FORMAL_RUNNER_DONE.json").read_text())["status"]=="complete"
    logs=p.WORK/phase/"driver_logs";logs.mkdir(parents=True,exist_ok=False)
    p.write_new(p.RECORD/(phase.upper()+"_LAUNCH_RECEIPT.json"),{"created_utc":datetime.now(timezone.utc).isoformat(),"phase":phase,
        "implementation_sha256":p.sha(p.RECORD/"IMPLEMENTATION.json"),"automatic_evaluation":False,"paired_folds":["e","f"]})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",MPLCONFIGDIR="/tmp/aflow_matplotlib_cache")
    children=[];handles=[]
    try:
        for gpu,fold in enumerate(("e","f")):
            handle=(logs/f"{fold}_otsu.log").open("x");handles.append(handle)
            command=[sys.executable,str(ROOT/"scripts/train_n4_otsu_clean.py"),"--fold",fold,"--role","otsu","--device",f"cuda:{gpu}"]
            if phase=="smoke":command.append("--smoke")
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);children.append(child)
            print(f"started {phase} Otsu/{fold} pid={child.pid}",flush=True)
        while any(c.poll() is None for c in children):
            if datetime.now(timezone.utc)>=deadline:raise RuntimeError("30h window reached")
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError("student failed; inspect driver logs")
            time.sleep(3)
        if any(c.returncode!=0 for c in children):raise RuntimeError("student failed")
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for handle in handles:handle.close()
    if phase=="smoke":freeze("smoke")
    p.write_new(p.RECORD/(phase.upper()+"_RUNNER_DONE.json"),{"status":"complete","finished_utc":datetime.now(timezone.utc).isoformat(),"inner_access":False})
    print(f"Otsu {phase} complete; readout remains manual and frozen",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("smoke","formal"));main(parser.parse_args().phase)
