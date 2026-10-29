"""External finite wall-clock bound for the already-running CPU cache builder."""
import signal
import subprocess
from scripts.race_v2_20261029.common import *

def main(pid=367):
    started=time.time();stop=min(started+4*3600,read(STAGE/'AUTHORIZATION.json')['workers']['1154259']['deadline_unix'])
    write(R/'PREPARATION_WATCH.json',dict(at=now(),pid=os.getpid(),builder_pid=pid,
          stop_unix=stop,numerical_builder_unchanged=True,reason='external bound on active CPU preparation'))
    while time.time()<stop:
        if (R/'DATA_READY.json').exists() or list(R.glob('PREPARE_FAILURE_*.json')):return
        try:os.kill(pid,0)
        except ProcessLookupError:return
        time.sleep(20)
    lines=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()
    pairs=[tuple(map(int,line.split())) for line in lines];targets={pid}
    while True:
        children={child for child,parent in pairs if parent in targets}
        if children<=targets:break
        targets|=children
    for target in sorted(targets,reverse=True):
        try:os.kill(target,signal.SIGTERM)
        except ProcessLookupError:pass
    write(R/f'PREPARE_FAILURE_{time.time_ns()}.json',dict(at=now(),reason='finite preparation wall-clock bound',
          stopped_pids=sorted(targets),patient_training_updates=0,test_pixels=False))

if __name__=='__main__':main()
