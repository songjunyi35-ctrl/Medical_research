import fcntl
import subprocess
import traceback
from scripts.segresnet_inference_test270_20261028.common import *


def main():
    lock=(R/'RUN.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    for stage,done in [('preflight','SYNTHETIC_GATE.json'),('evaluate','RAW_TEST270.json'),('publish','COMPLETION_AUDIT.json')]:
        if (R/done).exists():continue
        status('starting_'+stage)
        with (TMP/(stage+'.log')).open('ab') as log:
            p=subprocess.run([str(PY),'-m','scripts.segresnet_inference_test270_20261028.'+stage],
                  cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT,
                  timeout=max(1,read(R/'AUTHORIZATION.json')['deadline_unix']-time.time()))
        if p.returncode:raise RuntimeError(stage+' failed; see scratch log; no automatic repeated model forward')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        failure=dict(at=now(),error=repr(e),traceback=traceback.format_exc(),model_access=(R/'MODEL_PREACCESS.json').exists())
        write(R/f'FAILURE_{time.time_ns()}.json',failure)
        status('failed',failure=failure)
        raise
