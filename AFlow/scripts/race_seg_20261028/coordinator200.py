"""Administrative replacement for the user's 173 -> 200 budget amendment."""
import fcntl
import signal
import subprocess
import time
from scripts.race_seg_20261028.common import *
from scripts.race_seg_20261028.coordinator import status


def launch(arm):
    environment = env()
    environment['CUDA_VISIBLE_DEVICES'] = str(ARMS.index(arm))
    log = (TMP/f'train_{arm}.log').open('ab')
    try:
        return subprocess.Popen([str(PY), '-m', 'scripts.race_seg_20261028.resume200', arm],
                                cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()


def stop_owned(processes):
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes.values():
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def main():
    lock = (R/'COORDINATOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_sources('EXECUTION_FREEZE.json')
    verify_sources('RESUME200_FREEZE.json')
    assert read(R/'BUDGET_MIGRATION.json')['state'] == 'checkpoints_verified'
    budget = read(R/'BUDGET_DECISION.json')
    assert budget['epochs'] == 200 and budget['total_updates_each'] == 24400
    processes = {}
    recoveries = {arm: 0 for arm in ARMS}
    try:
        deadline(); disk()
        for arm in ARMS:
            processes[arm] = launch(arm)
        while True:
            write(R/'TRAIN_PIDS.json', dict(at=now(), worker=1152470,
                  controller='coordinator200', processes={a:p.pid for a,p in processes.items()}))
            deadline(); disk()
            snapshot = {}
            for arm, process in list(processes.items()):
                snapshot[arm] = {name:read(R/arm/name) for name in ('STATUS.json','HEARTBEAT.json')
                                 if (R/arm/name).exists()}
                code = process.poll()
                if code is None:
                    continue
                if code == 0:
                    done = read(R/arm/'DONE.json')
                    assert done['last_complete_epoch'] == 200 and done['updates'] == 24400
                    continue
                text = (TMP/f'train_{arm}.log').read_text(errors='replace')[-6000:]
                recoverable = any(x in text for x in ('Input/output error','DataLoader worker','Bus error','Connection reset'))
                if not recoverable or recoveries[arm] >= 2:
                    raise RuntimeError(f'{arm} failed with exit {code}; no unapproved restart')
                recoveries[arm] += 1
                write(R/arm/f'ENGINEERING_RECOVERY_{recoveries[arm]}.json', dict(at=now(),
                      prior_exit=code, retained_epoch=read(R/arm/'STATUS.json')['last_complete_epoch']))
                processes[arm] = launch(arm)
            status('training_200', training_started=True, budget=budget,
                   arms=snapshot, recoveries=recoveries, planned_budget_migrations=1)
            if all(p.poll() == 0 for p in processes.values()):
                break
            time.sleep(20)
    except BaseException as error:
        stop_owned(processes)
        write(R/'COORDINATOR_FAILURE.json', dict(at=now(), error=repr(error), controller='coordinator200'))
        status('failed', error=repr(error), training_started=bool(processes))
        subprocess.run([str(PY),'-m','scripts.race_seg_20261028.finalize'], cwd=ROOT, env=env(), check=False)
        raise
    subprocess.run([str(PY),'-m','scripts.race_seg_20261028.finalize'], cwd=ROOT, env=env(), check=True)
    for script in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py',
                   'verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py'):
        subprocess.run([str(PY), str(ROOT/'scripts'/script)], cwd=ROOT, env=env(), check=True)


if __name__ == '__main__':
    main()
