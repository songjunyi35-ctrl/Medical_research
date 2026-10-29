"""Replace only the idle head publication waiter with the reviewed version."""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .governance import ROOT, RECORD, SCRATCH, atomic_json
from .select_final_innovation import verify_freeze, stage_guard


def main():
    stage_guard()
    verify_freeze(RECORD / 'FINAL_PUBLICATION_INNOVATION_FREEZE_V2.json')
    assert not (RECORD / 'FINAL_TEST_SELECTION.json').exists()
    old_pid = 2079634
    old_module = 'scripts.race_v2_stage_20261029.final_report_watch_innovation'
    command = Path(f'/proc/{old_pid}/cmdline')
    if command.exists():
        arguments = command.read_bytes().split(b'\0')
        assert old_module.encode() in arguments
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(20):
            if not command.exists():
                break
            time.sleep(.1)
        assert not command.exists()
    module = 'scripts.race_v2_stage_20261029.final_report_watch_innovation_v2'
    log = SCRATCH / 'final_report_watch_innovation_v2.log'
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    with log.open('ab') as output:
        child = subprocess.Popen([sys.executable, '-m', module], cwd=ROOT, env=environment,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(RECORD / 'INNOVATION_PUBLICATION_V2_LAUNCH.json', dict(
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(), pid=child.pid, module=module,
        log_path=str(log), replaced_idle_pid=old_pid, original80_training_unchanged=True,
        selector_pid_unchanged=2079633, new270_selection_or_access=False))
    print(child.pid)


if __name__ == '__main__':
    main()
