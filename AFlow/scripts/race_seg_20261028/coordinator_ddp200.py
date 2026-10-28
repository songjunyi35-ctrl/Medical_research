"""Supervise only the user-authorized two-GPU algorithm continuation."""
import fcntl
import signal
import subprocess
import time
from scripts.race_seg_20261028.common import *


def main():
    lock=(R/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources('EXECUTION_FREEZE.json');verify_sources('DDP200_FREEZE.json')
    assert read(R/'SMOKE_DDP200.json')['status']=='passed'
    assert read(R/'control/STATUS.json')['state']=='stopped_by_user'
    process=None;recoveries=0
    try:
        while True:
            deadline();disk()
            log=(TMP/'train_ddp200.log').open('ab')
            e=env();e.update(NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN')
            process=subprocess.Popen([str(PY),'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                    '-m','scripts.race_seg_20261028.ddp200'],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close()
            write(R/'TRAIN_PIDS.json',dict(at=now(),worker=1152470,controller='coordinator_ddp200',processes={'race_ddp':process.pid}))
            while process.poll() is None:
                deadline();disk()
                write(R/'PIPELINE_STATUS.json',dict(at=now(),state='training_ddp200',training_started=True,
                      active_arms=['race'],world_size=2,budget=read(R/'BUDGET_DECISION.json'),recoveries=recoveries,
                      race_status=read(R/'race/STATUS.json'),race_heartbeat=read(R/'race/HEARTBEAT.json'),
                      control_state='stopped_by_user',test_access=False))
                time.sleep(20)
            if process.returncode==0:
                done=read(R/'race/DONE.json')
                assert done['last_complete_epoch']==200 and done['updates']==24400
                break
            logtext=(TMP/'train_ddp200.log').read_text(errors='replace')[-10000:]
            recoverable=any(x in logtext for x in ('Input/output error','DataLoader worker','Bus error','Connection reset'))
            if not recoverable or recoveries>=2:raise RuntimeError('DDP failed; no further automatic restart')
            recoveries+=1
            write(R/'race'/f'ENGINEERING_RECOVERY_{recoveries}.json',dict(at=now(),prior_exit=process.returncode,
                  retained_epoch=read(R/'race/STATUS.json')['last_complete_epoch'],mode='two-GPU DDP'))
        subprocess.run([str(PY),'-m','scripts.race_seg_20261028.finalize_ddp200'],cwd=ROOT,env=env(),check=True)
        for script in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py',
                       'verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py'):
            subprocess.run([str(PY),str(ROOT/'scripts'/script)],cwd=ROOT,env=env(),check=True)
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=20)
        write(R/'COORDINATOR_FAILURE.json',dict(at=now(),error=repr(error),controller='coordinator_ddp200'))
        subprocess.run([str(PY),'-m','scripts.race_seg_20261028.finalize_ddp200'],cwd=ROOT,env=env(),check=False)
        raise


if __name__=='__main__':main()
