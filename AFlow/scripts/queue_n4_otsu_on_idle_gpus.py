"""Safe queue: no CUDA task until shared training completes and both GPUs are idle."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import n4_otsu_protocol as p
from scripts.freeze_n4_otsu_clean import check_implementation


def idle_gpu_rows():
    response=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    rows=[tuple(int(v.strip()) for v in line.split(",")) for line in response.stdout.splitlines() if line.strip()]
    return rows,len(rows)==2 and all(memory<=512 and usage==0 for _,memory,usage in rows)


def main():
    check_implementation()
    deadline=datetime.fromisoformat("2026-09-06T14:02:39+00:00")
    p.write_new(p.RECORD/"IDLE_QUEUE_RECEIPT.json",{"created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"queued_without_cuda_allocation","prerequisite":"shared FORMAL_RUNNER_DONE complete then3 consecutive idle GPU checks10s apart",
        "gpu_idle_gate":"exactly2 GPUs;eachmemory<=512MiB and utilization0","smoke_then_formal":True,
        "concurrent_training_smoke_forbidden":True,"deadline_utc":deadline.isoformat(),"runner_sha256":p.sha(Path(__file__))})
    print("queued: shared formal training must finish; no smoke/CUDA process started",flush=True)
    ready=p.SHARED_RECORD/"FORMAL_RUNNER_DONE.json"
    while not ready.exists():
        if datetime.now(timezone.utc)>=deadline:raise RuntimeError("window ended before shared training completed")
        time.sleep(30)
    assert json.loads(ready.read_text())["status"]=="complete"
    consecutive=0
    while consecutive<3:
        if datetime.now(timezone.utc)>=deadline:raise RuntimeError("window ended before idle GPU gate")
        rows,idle=idle_gpu_rows();consecutive=consecutive+1 if idle else 0
        print(json.dumps({"gpu_state":rows,"consecutive_idle_checks":consecutive}),flush=True)
        if consecutive<3:time.sleep(10)
    p.write_new(p.RECORD/"IDLE_GPU_GATE_PASSED.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"rows":rows,"consecutive":3,"shared_training_done_sha256":p.sha(ready)})
    # Both calls are the already-frozen runners. Evaluation remains unlaunched.
    for phase in ("smoke","formal"):
        subprocess.run([sys.executable,str(ROOT/"scripts/run_n4_otsu_clean.py"),phase],cwd=ROOT,check=True)
    print("Otsu smoke and formal completed on previously idle GPUs; no evaluation launched",flush=True)


if __name__=="__main__":main()
