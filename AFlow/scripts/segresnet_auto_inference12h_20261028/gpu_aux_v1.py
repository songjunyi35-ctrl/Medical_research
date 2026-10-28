"""Independent second-GPU producer lane; fixed resource lease, bounded memory."""
import fcntl,importlib,traceback
import torch
from scripts.segresnet_auto_inference12h_20261028 import common as shared
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
shared.guard=guard


def main():
    resource=read(R/'RESOURCE_CURRENT.json');token=resource['token']
    lock=(R/'AUX_WORKER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def check():guard();assert read(R/'RESOURCE_CURRENT.json')['token']==token
    def report(state,**kw):write(R/'AUX_STATUS.json',dict(at=now(),state=state,worker=resource['worker'],physical_gpu=1,resource_token=token,training=False,test_tuned=True,**kw))
    original_append=shared.append_log
    def log(name,value):original_append(name,dict(**value,execution_worker=resource['worker'],resource_token=token,lane='aux'))
    shared.guard=check;shared.status=report;shared.append_log=log;shared.CACHE=Path(resource['cache_dir'])
    ready=read(R/'DATA_READY_V2.json');assert ready['resource_token']==token
    assert sha(shared.CACHE/'index.json')==ready['payload_sha256']['index']
    numeric_mode();torch.cuda.set_per_process_memory_fraction(.65);model=load_model('cuda')
    write(R/'AUX_WORKER_PID.json',dict(at=now(),pid=os.getpid(),worker=resource['worker'],resource_token=token))
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        check();jobs=[]
        for p in sorted((R/'queue_aux_v1').glob('*.json')):
            job=read(p);folder=R/'jobs'/job['id']
            if job.get('resource_token')!=token or (folder/'DONE.json').exists() or (folder/'FAILED.json').exists():continue
            if any(not (R/'jobs'/name/'DONE.json').exists() for name in job.get('depends_on',[])):continue
            jobs.append(job)
        if not jobs:report('queue_waiting');time.sleep(3);continue
        job=jobs[0];verify_job(job)
        try:
            module=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+job['handler'])
            module.guard=check;module.status=report;module.append_log=log
            # Render helper gate/forward functions are unchanged frozen numerical sources.
            report('executing_job',job=job['id']);module.execute(model,job)
        except BaseException as e:
            failure=dict(at=now(),job=job['id'],worker=resource['worker'],error=repr(e),traceback=traceback.format_exc(),lane='aux')
            write(R/'jobs'/job['id']/'FAILED.json',failure);original_append('RECOVERY_FAILURES.jsonl',failure)
            with (R/'RECOVERY_COUNT.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);counter=read(R/'RECOVERY_COUNT.json');counter['used']+=1;counter['reasons'].append(failure);write(R/'RECOVERY_COUNT.json',counter)
                if counter['used']>=6:raise
        model.eval();torch.cuda.empty_cache()
    report('window_closed')
if __name__=='__main__':main()
