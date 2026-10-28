"""Resource-lease-aware recovery; numerical handlers remain separately hashed."""
import fcntl
import importlib
import subprocess
import traceback
import torch
from scripts.segresnet_auto_inference12h_20261028 import common as shared
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
shared.guard=guard


def main():
    resource=read(R/'RESOURCE_CURRENT.json');token=resource['token']
    mode=os.environ.get('AFLOW_WORKER_MODE','producer')
    assert mode in ('producer','analysis','cpu')
    lock=(R/f'WORKER_V2_{mode}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def resource_guard():
        guard();assert read(R/'RESOURCE_CURRENT.json')['token']==token,'resource lease superseded'
    def report(state,**kwargs):
        write(R/('PIPELINE_STATUS.json' if mode=='producer' else ('ANALYSIS_STATUS.json' if mode=='analysis' else 'CPU_STATUS.json')),
              dict(at=now(),state=state,worker=resource['worker'],gpu_type=resource['gpu_type'],
                   resource_token=token,training=False,test_tuned=True,**kwargs))
    original_append=shared.append_log
    def log(name,value):
        original_append(name,dict(**value,execution_worker=resource['worker'],resource_token=token))
    shared.status=report;shared.append_log=log
    shared.CACHE=Path(resource['cache_dir'])
    resource_guard();numeric_mode()
    if mode!='cpu':
        fraction=resource['producer_memory_fraction'] if mode=='producer' else resource['analysis_memory_fraction']
        torch.cuda.set_per_process_memory_fraction(fraction)
    if mode=='producer':
        expected=read(OLD/'CACHE_READY.json')['payload_sha256']
        actual=dict(images=sha(shared.CACHE/'images.f32'),labels=sha(shared.CACHE/'labels.u8'),index=sha(shared.CACHE/'index.json'))
        assert actual==expected
        write(R/'DATA_READY_V2.json',dict(at=now(),worker=resource['worker'],status='passed',payload_sha256=actual,resource_token=token))
        model=load_model('cuda')
    else:model=None
    write(R/f'WORKER_V2_{mode}_PID.json',dict(at=now(),pid=os.getpid(),worker=resource['worker'],resource_token=token))
    queue=R/('queue_v2' if mode=='producer' else ('analysis_queue_v2' if mode=='analysis' else 'cpu_queue_v2'))
    jobbase=R/('jobs' if mode=='producer' else 'analyses')
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        resource_guard();pending=[]
        for path in sorted(queue.glob('*.json')):
            job=read(path)
            if job.get('resource_token')!=token:continue
            folder=jobbase/job['id']
            if (folder/'DONE.json').exists() or (folder/'FAILED.json').exists():continue
            if any(not (R/'jobs'/name/'DONE.json').is_file() for name in job.get('depends_on',[])):continue
            if any(not (R/'analyses'/name/'DONE.json').is_file() for name in job.get('analysis_dependencies',[])):continue
            pending.append(job)
        if not pending:report('queue_waiting',mode=mode);time.sleep(3);continue
        job=pending[0];verify_job(job)
        try:
            module=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+job['handler'])
            module.status=report;module.append_log=log;module.guard=resource_guard
            report('executing_job',job=job['id'],mode=mode)
            if mode=='producer':module.execute(model,job)
            else:module.execute(job)
        except BaseException as e:
            failure=dict(at=now(),job=job['id'],mode=mode,worker=resource['worker'],error=repr(e),traceback=traceback.format_exc(),resource_token=token)
            write(jobbase/job['id']/'FAILED.json',failure);original_append('RECOVERY_FAILURES.jsonl',failure)
            if model is not None:model.eval()
            with (R/'RECOVERY_COUNT.lock').open('a') as count_lock:
                fcntl.flock(count_lock,fcntl.LOCK_EX)
                counter=read(R/'RECOVERY_COUNT.json') if (R/'RECOVERY_COUNT.json').exists() else dict(used=1,reasons=['initial worker disappeared'])
                counter['used']+=1;counter.setdefault('reasons',[]).append(failure)
                write(R/'RECOVERY_COUNT.json',counter)
                if counter['used']>=6:raise RuntimeError('research engineering recovery cap reached')
        if mode!='cpu':torch.cuda.empty_cache()
    report('window_closed',mode=mode)


if __name__=='__main__':main()
