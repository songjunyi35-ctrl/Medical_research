"""Activate final108 selection/publication after the bounded idea studies."""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .governance import ROOT, RECORD, SCRATCH, atomic_json
from .select_final_continuation import POLICY, verify_freeze, stage_guard


def main():
    stage_guard()
    verify_freeze(POLICY)
    verify_freeze(RECORD/'FINAL_PUBLICATION_CONTINUATION_FREEZE.json')
    assert not (RECORD/'FINAL_TEST_SELECTION.json').exists()
    old_pid = 2082077
    command = Path(f'/proc/{old_pid}/cmdline')
    old_module = b'scripts.race_v2_stage_20261029.final_report_watch_innovation_v2'
    if command.exists():
        assert old_module in command.read_bytes().split(b'\0')
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(20):
            if not command.exists():
                break
            time.sleep(.1)
        assert not command.exists()
    previous_selector = Path('/proc/2079633/cmdline')
    assert not previous_selector.exists(), 'previous selector must remain stopped'
    receipt = RECORD/'CONTINUATION_CHAIN_LAUNCH.json'
    assert not receipt.exists(), 'review existing launch before any retry'
    items = []
    for name in ('select_final_continuation','final_report_watch_continuation'):
        module = 'scripts.race_v2_stage_20261029.'+name
        log = SCRATCH/(name+'.log')
        with log.open('ab') as output:
            child = subprocess.Popen([sys.executable,'-m',module],cwd=ROOT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}, stdout=output,
                stderr=subprocess.STDOUT,start_new_session=True)
        items.append(dict(module=module,pid=child.pid,log_path=str(log)))
    atomic_json(receipt, dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisors=items,
        old_idle_publication_pid=old_pid,old_idle_publication_stopped=True,
        original_training_unchanged=True,all_worker_processes_untouched_by_this_launcher=True,
        no_new270_selection_at_launch=True))
    print(receipt.read_text())


if __name__ == '__main__':
    main()
