"""Append-only, hash-verified GPU queue; root researcher supplies reviewed jobs."""
import fcntl
import importlib
import subprocess
import traceback
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *


def bootstrap():
    guard();TMP.mkdir(parents=True,exist_ok=True)
    expected=read(OLD/'CACHE_READY.json')['payload_sha256']
    actual=dict(images=sha(CACHE/'images.f32'),labels=sha(CACHE/'labels.u8'),index=sha(CACHE/'index.json'))
    assert actual==expected
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    rows,_,_,_=data()
    assert len(rows)==270
    write(R/'DATA_READY.json',dict(at=now(),status='passed',payload_sha256=actual,
          scans=270,subjects=119,new_preprocessing=False,test_tuned=True))


def main():
    lock=(R/'GPU_WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bootstrap();numeric_mode();torch.cuda.set_per_process_memory_fraction(.85)
    model=load_model('cuda');last_gpu=0.;idle=0.;failures=0
    write(R/'GPU_WORKER_PID.json',dict(at=now(),pid=os.getpid(),worker=1152682))
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        guard()
        queue=sorted((R/'queue').glob('*.json'))
        pending=[p for p in queue if not (R/'jobs'/read(p)['id']/'DONE.json').exists() and
                 not (R/'jobs'/read(p)['id']/'FAILED.json').exists()]
        if not pending:
            started=time.monotonic();status('gpu_queue_waiting',idle_seconds_total=idle)
            time.sleep(3);idle+=time.monotonic()-started;continue
        job=read(pending[0]);verify_job(job)
        try:
            module=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+job['handler'])
            module.execute(model,job)
        except BaseException as e:
            failure=dict(at=now(),job=job['id'],error=repr(e),traceback=traceback.format_exc(),test_tuned=True)
            write(R/'jobs'/job['id']/'FAILED.json',failure)
            append_log('ENGINEERING_FAILURES.jsonl',failure)
            failures+=1;torch.cuda.empty_cache();model.eval()
            if failures>=6:raise RuntimeError('engineering failure cap reached')
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
        append_log('GPU_UTILIZATION.jsonl',dict(at=now(),job=job['id'],sample=gpu,queue_remaining=len(pending)-1,idle_seconds_total=idle))
    status('research_window_gpu_queue_closed',idle_seconds_total=idle)


if __name__=='__main__':main()
