"""Small-memory GPU/CPU analysis queue, concurrent with frozen-model rendering."""
import fcntl
import importlib
import traceback
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *


def main():
    lock=(R/'ANALYSIS_WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    numeric_mode();torch.cuda.set_per_process_memory_fraction(.07)
    write(R/'ANALYSIS_WORKER_PID.json',dict(at=now(),pid=os.getpid()))
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        guard()
        pending=[]
        for path in sorted((R/'analysis_queue').glob('*.json')):
            job=read(path);folder=R/'analyses'/job['id']
            if (folder/'DONE.json').exists() or (folder/'FAILED.json').exists():continue
            if any(not (R/'jobs'/dep/'DONE.json').exists() for dep in job.get('depends_on',[])):continue
            pending.append(job)
        if not pending:time.sleep(3);continue
        job=pending[0];verify_job(job)
        try:
            importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+job['handler']).execute(job)
        except BaseException as e:
            failure=dict(at=now(),job=job['id'],error=repr(e),traceback=traceback.format_exc(),model_forward=False)
            write(R/'analyses'/job['id']/'FAILED.json',failure);append_log('ANALYSIS_FAILURES.jsonl',failure)
        torch.cuda.empty_cache()


if __name__=='__main__':main()
