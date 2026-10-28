"""Second scoring lane on GPU0 after the initial neural producer is finished."""
import fcntl,importlib,traceback
import torch
from scripts.segresnet_auto_inference12h_20261028 import common as shared
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
shared.guard=guard


def main():
    resource=read(R/'RESOURCE_CURRENT.json');token=resource['token']
    lock=(R/'ANALYSIS_AUX.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def check():guard();assert read(R/'RESOURCE_CURRENT.json')['token']==token
    def report(state,**kw):write(R/'ANALYSIS_AUX_STATUS.json',dict(at=now(),state=state,worker=resource['worker'],physical_gpu=0,resource_token=token,training=False,test_tuned=True,**kw))
    shared.guard=check;shared.status=report;shared.CACHE=Path(resource['cache_dir']);numeric_mode();torch.cuda.set_per_process_memory_fraction(.55)
    write(R/'ANALYSIS_AUX_PID.json',dict(at=now(),pid=os.getpid(),worker=resource['worker'],resource_token=token))
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        check();pending=[]
        for p in sorted((R/'analysis_queue_aux_v1').glob('*.json')):
            job=read(p);folder=R/'analyses'/job['id']
            if job.get('resource_token')!=token or (folder/'DONE.json').exists() or (folder/'FAILED.json').exists():continue
            if any(not (R/'jobs'/name/'DONE.json').exists() for name in job.get('depends_on',[])):continue
            if any(not (R/'analyses'/name/'DONE.json').exists() for name in job.get('analysis_dependencies',[])):continue
            pending.append(job)
        if not pending:report('queue_waiting');time.sleep(3);continue
        job=pending[0];verify_job(job)
        try:
            module=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+job['handler']);module.guard=check;module.status=report
            report('executing_job',job=job['id']);module.execute(job)
        except BaseException as e:
            failure=dict(at=now(),job=job['id'],worker=resource['worker'],error=repr(e),traceback=traceback.format_exc(),lane='analysis_aux')
            write(R/'analyses'/job['id']/'FAILED.json',failure);append_log('RECOVERY_FAILURES.jsonl',failure)
            with (R/'RECOVERY_COUNT.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX);counter=read(R/'RECOVERY_COUNT.json');counter['used']+=1;counter['reasons'].append(failure);write(R/'RECOVERY_COUNT.json',counter)
                if counter['used']>=6:raise
        torch.cuda.empty_cache()
    report('window_closed')
if __name__=='__main__':main()
