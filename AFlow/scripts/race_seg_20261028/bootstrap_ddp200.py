"""Retain a complete algorithm checkpoint, gate DDP, then resume both GPUs."""
import signal
import subprocess
import time
import torch
from scripts.race_seg_20261028.common import *


def main():
    torch.set_num_threads(2)
    verify_sources('EXECUTION_FREEZE.json');verify_sources('DDP200_FREEZE.json');deadline();disk()
    assert not (R/'DDP_MIGRATION.json').exists(), 'do not restart a prior migration'
    assert read(R/'control/STATUS.json')['state']=='stopped_by_user'
    pid=read(R/'DDP200_PLAN.json')['prior_race_pid']
    assert b'scripts.race_seg_20261028.train\x00race' in Path(f'/proc/{pid}/cmdline').read_bytes()
    initial=read(R/'race/STATUS.json')['last_complete_epoch']
    write(R/'DDP_MIGRATION.json',dict(at=now(),state='waiting_for_complete_epoch',initial_complete_epoch=initial,old_race_pid=pid))
    stop_at=time.monotonic()+900
    while True:
        deadline();disk();assert time.monotonic()<stop_at
        status=read(R/'race/STATUS.json');epoch=status['last_complete_epoch']
        history=(R/'race/HISTORY.jsonl').read_text().splitlines()
        if epoch>initial and json.loads(history[-1])['last_complete_epoch']==epoch:break
        time.sleep(.1)
    os.kill(pid,signal.SIGSTOP)
    saved=torch.load(TMP/'runs/race/latest.pth',map_location='cpu',weights_only=False)
    best=torch.load(TMP/'runs/race/best.pth',map_location='cpu',weights_only=False)
    assert saved['epoch']==epoch and saved['updates']==epoch*122==status['updates']
    assert saved['best_epoch']==status['best_epoch']==best['epoch']
    assert saved['freeze_sha256']==sha(R/'EXECUTION_FREEZE.json')
    assert all(int(s['step'])==saved['updates'] for s in saved['optimizer_state_dict']['state'].values())
    assert all(bool(v.isfinite().all()) for v in saved['model_state_dict'].values())
    assert all(bool(v.isfinite().all()) for s in saved['optimizer_state_dict']['state'].values()
               for v in s.values() if isinstance(v,torch.Tensor))
    backup=TMP/'preddp200/race';backup.mkdir(parents=True,exist_ok=True)
    for name in ('latest.pth','best.pth'):
        shutil.copy2(TMP/'runs/race'/name,backup/name)
        assert sha(backup/name)==sha(TMP/'runs/race'/name)
    write(R/'race/PRE_DDP_STATUS.json',status)
    write(R/'DDP_MIGRATION.json',dict(at=now(),state='checkpoint_verified',old_race_pid=pid,
          retained_epoch=epoch,retained_updates=saved['updates'],latest_sha256=sha(backup/'latest.pth'),
          best_sha256=sha(backup/'best.pth'),optimizer_not_reset=True,
          heartbeat_at_pause=read(R/'race/HEARTBEAT.json'),
          incomplete_work='Any work after the complete checkpoint is discarded; exact attempted-update count unavailable',
          original_numerical_source_unchanged=True,planned_migration_count=1))
    del saved,best
    os.killpg(pid,signal.SIGKILL)
    time.sleep(2)
    e=env();e.update(NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN')
    with (TMP/'smoke_ddp200.log').open('ab') as log:
        subprocess.run([str(PY),'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
             '-m','scripts.race_seg_20261028.ddp200','--smoke'],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=360)
    assert read(R/'SMOKE_DDP200.json')['status']=='passed'
    shutil.copy2(R/'BUDGET_DECISION.json',R/'BUDGET_ORIGINAL_173.json')
    write(R/'BUDGET_DECISION.json',dict(at=now(),status='passed',epochs=200,total_updates_each=24400,
          updates_per_epoch=122,microbatch=2,effective_batch=8,world_size=2,local_batch=4,
          accumulation_per_rank=2,cpu_threads_per_rank=8,active_arms=['race'],control='stopped_by_user_at_epoch4',
          selection='User cancelled control and requested two-GPU algorithm training to cumulative200',
          deadline_unix=read(R/'AUTHORIZATION.json')['deadline_unix'],deadline_unchanged=True,
          original_173_budget='BUDGET_ORIGINAL_173.json',memory_ceiling_gib_per_gpu=38,
          smoke_sha256=sha(R/'SMOKE_DDP200.json'),historical_comparison_only=True))
    log=(TMP/'coordinator_ddp200.log').open('ab')
    process=subprocess.Popen([str(PY),'-m','scripts.race_seg_20261028.coordinator_ddp200'],cwd=ROOT,env=e,
          stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    write(R/'COORDINATOR_DDP200_PID.json',dict(at=now(),worker=1152470,pid=process.pid))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(R/'DDP_MIGRATION_FAILURE.json',dict(at=now(),error=repr(error)))
        raise
