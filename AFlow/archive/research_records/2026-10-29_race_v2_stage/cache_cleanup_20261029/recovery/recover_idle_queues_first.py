"""Recover only pre-access dependency waiters after authorized disk cleanup.

Never restarts the training process or a queue that began a new model forward.
Historical failures/start records are archived byte-for-byte before relaunch.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .governance import ROOT, RECORD, atomic_json, guard, sha

A=ROOT/'archive/research_records/2026-10-29_race_v2_inference_a'
B=ROOT/'archive/research_records/2026-10-29_race_v2_inference_b'
QUEUES=[('a_snapshot',0,'scripts.race_v2_inference_a_20261029.snapshot108',['--watch'],None),
        ('a_final',0,'scripts.race_v2_inference_a_20261029.final270_watch',['--watch'],None),
        ('b_snapshot',1,'scripts.race_v2_inference_b_20261029.snapshot_watch',[],B/'final_snapshot/WATCH_START.json'),
        ('b_final',1,'scripts.race_v2_inference_b_20261029.final_watch',[],B/'final270/WATCH_START.json')]


def process_ids(module):
    pids=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:arguments=(process/'cmdline').read_bytes().split(b'\0')
        except (OSError,PermissionError):continue
        if module.encode() in arguments:pids.append(int(process.name))
    return pids


def main(restart):
    guard(1154121)
    assert shutil.disk_usage(ROOT).free>=22*2**30, 'require recovered headroom before idle restart'
    assert not (RECORD/'FINAL_TEST_SELECTION.json').exists()
    assert not list((RECORD/'SNAPSHOT108_QUEUE').glob('*.json'))
    for directory in (A/'snapshots',A/'final270',B/'final_snapshot',B/'final270'):
        assert not list(directory.glob('**/MODEL_PREACCESS.json')), 'new model access already started; review required'
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    scratch=Path('/tmp/aflow_race_v2_20261029/idle_queue_recovery');scratch.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,device,module,arguments,start in QUEUES:
        live=process_ids(module);assert len(live)<=1,(name,live)
        if live:
            results.append(dict(queue=name,state='already_alive_not_restarted',pid=live[0]));continue
        if not restart:
            results.append(dict(queue=name,state='dead_before_new_model_access'));continue
        history=RECORD/'cache_cleanup_20261029/queue_recovery_history'/stamp/name
        history.mkdir(parents=True,exist_ok=False)
        preserved={}
        if start is not None and start.exists():
            digest=sha(start);destination=history/'WATCH_START.previous.json'
            shutil.copy2(start,destination);assert sha(destination)==digest
            start.unlink();preserved[str(start)]=dict(path=str(destination),sha256=digest)
        folder=A if name.startswith('a') else B
        for failure in list(folder.glob('**/*WATCH_FAILURE*.json')):
            if 'queue_recovery_history' in failure.parts:continue
            destination=history/(failure.parent.name+'_'+failure.name)
            shutil.copy2(failure,destination);preserved[str(failure)]=dict(path=str(destination),sha256=sha(destination))
        environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(device),PYTHONPATH=str(ROOT),
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (scratch/(name+'_'+stamp+'.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module,*arguments],cwd=ROOT,env=environment,
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        results.append(dict(queue=name,state='restarted_idle_only',pid=child.pid,
                            prior_records_preserved=preserved,new_neural_forwards=0))
    value=dict(at=stamp,state='recovered_idle_dependency_waiters' if restart else 'inspected',
        queues=results,worker=1154121,free_bytes=shutil.disk_usage(ROOT).free,
        new_model_access_started=False,new270_scoring=False,training_process_restarted=False)
    if restart:atomic_json(RECORD/'IDLE_QUEUE_RECOVERY_'+stamp+'.json',value)
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restart',action='store_true');main(parser.parse_args().restart)
