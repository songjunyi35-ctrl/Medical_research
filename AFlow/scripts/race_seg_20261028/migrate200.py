"""Bounded planned checkpoint migration; never changes the frozen training code."""
import signal
import subprocess
import time
import torch
from scripts.race_seg_20261028.common import *


def command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes()


def main():
    verify_sources('EXECUTION_FREEZE.json'); verify_sources('RESUME200_FREEZE.json')
    deadline(); disk()
    assert not (R/'BUDGET_MIGRATION.json').exists(), 'migration must not run twice'
    pids = read(R/'TRAIN_PIDS.json')['processes']
    for arm, pid in pids.items():
        assert b'scripts.race_seg_20261028.train' in command(pid)
        assert arm.encode() in command(pid)
    controller_pid = read(R/'RESUME200_PLAN.json')['prior_controller_pid']
    assert b'scripts.race_seg_20261028.coordinator\x00' in command(controller_pid)
    migration = dict(at=now(), state='waiting_for_complete_epochs', old_pids=pids,
                     prior_controller_pid=controller_pid, retained={}, user_budget_amendment=True,
                     failed_engineering_recoveries=0)
    write(R/'BUDGET_MIGRATION.json', migration)
    # Stop supervision first. Training children have independent sessions and
    # continue until their next complete model+optimizer checkpoint is retained.
    os.kill(controller_pid, signal.SIGTERM)
    initial = {arm:read(R/arm/'STATUS.json')['last_complete_epoch'] for arm in ARMS}
    paused = set()
    stop_at = time.monotonic() + 1000
    while len(paused) < len(ARMS):
        deadline(); disk()
        assert time.monotonic() < stop_at, 'no next complete checkpoint in bounded migration window'
        for arm in ARMS:
            if arm in paused:
                continue
            state = read(R/arm/'STATUS.json')
            epoch = state['last_complete_epoch']
            if epoch <= initial[arm]:
                continue
            history = (R/arm/'HISTORY.jsonl').read_text().splitlines()
            if not history or json.loads(history[-1])['last_complete_epoch'] != epoch:
                continue
            os.kill(pids[arm], signal.SIGSTOP)
            paused.add(arm)
            write(R/f'PRE200_{arm}_STATUS.json', state)
            migration['retained'][arm] = dict(epoch=epoch, updates=state['updates'],
                heartbeat_at_pause=read(R/arm/'HEARTBEAT.json'),
                incomplete_epoch_work='Any work after this full checkpoint is discarded; exact attempted-update count unavailable')
            write(R/'BUDGET_MIGRATION.json', migration)
        time.sleep(.1)
    # Read the complete optimizer state before terminating paused processes.
    for arm in ARMS:
        folder = TMP/'runs'/arm
        saved = torch.load(folder/'latest.pth', map_location='cpu', weights_only=False)
        best = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
        state = read(R/f'PRE200_{arm}_STATUS.json')
        assert saved['epoch'] == state['last_complete_epoch']
        assert saved['updates'] == saved['epoch']*122 == state['updates']
        assert saved['best_epoch'] == state['best_epoch'] == best['epoch']
        assert saved['freeze_sha256'] == sha(R/'EXECUTION_FREEZE.json')
        assert all(int(s['step']) == saved['updates'] for s in saved['optimizer_state_dict']['state'].values())
        assert all(bool(v.isfinite().all()) for v in saved['model_state_dict'].values())
        assert all(bool(v.isfinite().all()) for s in saved['optimizer_state_dict']['state'].values()
                   for v in s.values() if isinstance(v, torch.Tensor))
        migration['retained'][arm].update(latest_sha256=sha(folder/'latest.pth'),
                                        best_sha256=sha(folder/'best.pth'),
                                        optimizer_steps_verified=True, all_finite=True)
        # Scratch-only immutable recovery copy; persistent limit unchanged.
        backup = TMP/'pre200'/arm
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('latest.pth', 'best.pth'):
            shutil.copy2(folder/name, backup/name)
            assert sha(folder/name) == sha(backup/name)
        del saved, best
    shutil.copy2(R/'BUDGET_DECISION.json', R/'BUDGET_ORIGINAL_173.json')
    budget = read(R/'BUDGET_DECISION.json')
    budget.update(at=now(), epochs=200, total_updates_each=24400, cpu_threads=8,
                  selection='User amendment to align 200 epochs; CPU threads chosen by synthetic throughput only',
                  original_budget_record='BUDGET_ORIGINAL_173.json',
                  deadline_unchanged=True, completion_inside_24h_guaranteed=False)
    write(R/'BUDGET_DECISION.json', budget)
    for pid in pids.values():
        os.killpg(pid, signal.SIGKILL)
    for pid in pids.values():
        for _ in range(100):
            path = Path(f'/proc/{pid}/stat')
            if not path.exists() or path.read_text().split()[2] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('old training process did not exit')
    migration.update(at=now(), state='checkpoints_verified', original_freeze_unchanged=True,
                     effective_batch=8, microbatch=2, target_epochs_each=200,
                     cpu_threads_before=2, cpu_threads_after=8, deadline_unchanged=True)
    write(R/'BUDGET_MIGRATION.json', migration)
    log = (TMP/'coordinator200.log').open('ab')
    process = subprocess.Popen([str(PY),'-m','scripts.race_seg_20261028.coordinator200'],
        cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    write(R/'COORDINATOR200_PID.json', dict(at=now(),pid=process.pid,worker=1152470))


if __name__ == '__main__':
    main()
