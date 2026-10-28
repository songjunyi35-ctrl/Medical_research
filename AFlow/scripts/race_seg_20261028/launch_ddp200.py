"""Continue the verified migration after pre-patient cleanup was repaired."""
import subprocess
from scripts.race_seg_20261028.common import *


def main():
    verify_sources('EXECUTION_FREEZE.json');verify_sources('DDP200_FREEZE.json');deadline();disk()
    assert not (R/'COORDINATOR_DDP200_PID.json').exists(), 'do not launch twice'
    migration=read(R/'DDP_MIGRATION.json')
    assert migration['state']=='checkpoint_verified' and migration['retained_epoch']==5
    assert sha(TMP/'runs/race/latest.pth')==migration['latest_sha256']
    assert sha(TMP/'runs/race/best.pth')==migration['best_sha256']
    smoke=read(R/'SMOKE_DDP200.json')
    assert smoke['status']=='passed' and smoke['prefetched_augmented_values_exact']
    assert smoke['execution_freeze_sha256']==sha(R/'DDP200_FREEZE.json')
    assert read(R/'DDP_GATE_PROCESS_EXIT.json')['exit_code']==0
    for rank in (0,1):assert read(R/f'SMOKE_DDP200_EXIT_RANK{rank}.json')['compute_complete']
    assert read(R/'control/STATUS.json')['state']=='stopped_by_user'
    assert not (R/'BUDGET_ORIGINAL_173.json').exists()
    shutil.copy2(R/'BUDGET_DECISION.json',R/'BUDGET_ORIGINAL_173.json')
    write(R/'BUDGET_DECISION.json',dict(at=now(),status='passed',epochs=200,total_updates_each=24400,
          updates_per_epoch=122,microbatch=2,effective_batch=8,world_size=2,local_batch=4,
          accumulation_per_rank=2,cpu_threads_per_rank=8,cpu_augmentation_lookahead=1,
          active_arms=['race'],control='stopped_by_user_at_epoch4',
          selection='User cancelled control and requested two-GPU algorithm training to cumulative200',
          deadline_unix=read(R/'AUTHORIZATION.json')['deadline_unix'],deadline_unchanged=True,
          original_173_budget='BUDGET_ORIGINAL_173.json',memory_ceiling_gib_per_gpu=38,
          smoke_sha256=sha(R/'SMOKE_DDP200.json'),historical_comparison_only=True))
    e=env();e.update(NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN')
    with (TMP/'coordinator_ddp200.log').open('ab') as log:
        process=subprocess.Popen([str(PY),'-m','scripts.race_seg_20261028.coordinator_ddp200'],cwd=ROOT,env=e,
              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(R/'COORDINATOR_DDP200_PID.json',dict(at=now(),worker=1152470,pid=process.pid))
    migration.update(at=now(),state='ddp_supervisor_started',controller_pid=process.pid,
                     synthetic_cleanup_repaired=True,source_freeze_sha256=sha(R/'DDP200_FREEZE.json'))
    write(R/'DDP_MIGRATION.json',migration)


if __name__=='__main__':main()
