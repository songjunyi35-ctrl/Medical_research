"""Replace idle head inference waiters; no worker training operation exists here."""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .governance import ROOT, RECORD, SCRATCH, atomic_json
from .select_final_innovation import POLICY, verify_freeze, stage_guard


def alive(pid, module):
    path = Path(f'/proc/{pid}/cmdline')
    try:
        return module in path.read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False


def main():
    stage_guard()
    verify_freeze(POLICY)
    verify_freeze(RECORD / 'FINAL_PUBLICATION_INNOVATION_FREEZE.json')
    assert not (RECORD / 'FINAL_TEST_SELECTION.json').exists()
    old_module = 'scripts.race_v2_stage_20261029.final_report_watch_v2'
    old_pid = 2068677
    stopped = False
    if alive(old_pid, old_module):
        os.kill(old_pid, signal.SIGTERM)
        stopped = True
        for _ in range(20):
            if not alive(old_pid, old_module):
                break
            time.sleep(.1)
        assert not alive(old_pid, old_module)
    receipt = RECORD / 'INNOVATION_CHAIN_LAUNCH.json'
    prior = __import__('json').loads(receipt.read_text()) if receipt.exists() else {}
    supervisors = []
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for name in ('select_final_innovation', 'final_report_watch_innovation'):
        module = 'scripts.race_v2_stage_20261029.' + name
        previous = next((item for item in prior.get('supervisors', []) if item['module'] == module), None)
        if previous and alive(previous['pid'], module):
            supervisors.append(previous)
            continue
        log = SCRATCH / (name + '.log')
        environment = os.environ.copy()
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable, '-m', module], cwd=ROOT, env=environment,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        supervisors.append(dict(module=module, pid=child.pid, log_path=str(log)))
    atomic_json(receipt, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisors=supervisors, old_idle_publication_pid=old_pid, old_idle_publication_stopped=stopped,
        old_selector_remains_stopped=True, no_worker_training_changes=True,
        no_new270_selection_or_access=True, old_freezes_retained=True))
    print(receipt.read_text())


if __name__ == '__main__':
    main()
