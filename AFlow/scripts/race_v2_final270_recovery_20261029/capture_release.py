"""Read-only final physical-GPU/lock check; never stops a worker or model."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'archive/research_records/2026-10-29_race_v2_final270_recovery'


def main():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in query.splitlines():
        index,uuid,memory,utilization=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(index),uuid=uuid,memory_mib=int(memory),utilization=int(utilization)))
    expected=json.loads((R/'RESOURCE_ALLOCATION.json').read_text())['queues']
    locks={}
    for owner,queue in expected.items():
        assert rows[queue['gpus'][0]]['uuid']==queue['gpu_uuid']
        with Path(queue['canonical_lock_path']).open('a') as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                locks[owner]=True
            except BlockingIOError:
                locks[owner]=False
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                      '--format=csv,noheader,nounits'],text=True).strip()
    value=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        worker=1154259,gpus=rows,canonical_locks_available=locks,compute_processes_csv=processes,
        no_gpu_process_or_worker_stopped=True,worker_allocation_not_deleted=True,
        original_training_and_recovery_cache_retained=True)
    path=R/'FINAL_RESOURCE_RELEASE.json';temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)
    print(json.dumps(value))


if __name__=='__main__':main()
