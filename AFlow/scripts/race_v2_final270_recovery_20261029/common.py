"""Root-owned authorization and physical-card locks for bounded final recovery."""
import contextlib
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.race_v2_stage_20261029.governance import ROOT, atomic_json, sha

R = ROOT/'archive/research_records/2026-10-29_race_v2_final270_recovery'
TMP = Path('/tmp/aflow_race_v2_final270_recovery_20261029')
SELECTION = ROOT/'archive/research_records/2026-10-29_race_v2_stage/FINAL_TEST_SELECTION.json'
SELECTION_SHA = '5b74021964fe8b6d0f34fbdb35303d7d05be34d6796caec2adc6e6a68c6579bc'
EXPECTED_SELECTION_SHA = SELECTION_SHA
WORKER = 1154259


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


write = atomic_json


def guard():
    auth = read(R/'AUTHORIZATION.json')
    assert auth['state'] == 'authorized_bounded_reconstruction_and_fixed_evaluation'
    assert time.time() < auth['deadline_unix']
    assert sha(SELECTION) == SELECTION_SHA == auth['original_selection_sha256']
    assert shutil.disk_usage(ROOT).free >= 20*2**30
    roots = {ROOT/value for value in auth['persistent_roots']}
    total = sum(path.stat().st_size for folder in roots for path in folder.rglob('*') if path.is_file())
    assert total < auth['persistent_total_max_bytes'], total
    if TMP.exists():
        assert shutil.disk_usage(TMP).free >= 100*2**30
    return auth


def gpu_lock(owner):
    """Return a held canonical physical-card lease; keep it until work ends."""
    guard()
    allocation = read(R/'RESOURCE_ALLOCATION.json')
    key = owner.lower().removeprefix('inference_')
    expected = allocation['queues'][key]
    device = expected['gpus'][0]
    assert expected['worker'] == WORKER and os.environ.get('CUDA_VISIBLE_DEVICES') == str(device)
    csv = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    actual = {int(line.split(',')[0].strip()):line.split(',')[1].strip() for line in csv.splitlines()}
    assert actual[device] == expected['gpu_uuid']
    path = Path(expected['canonical_lock_path'])
    path.parent.mkdir(parents=True,exist_ok=True)
    handle = path.open('a+')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handle.seek(0);handle.truncate()
    json.dump(dict(worker=WORKER,gpu=device,uuid=actual[device],owner=key,pid=os.getpid(),at=now()),handle)
    handle.flush()
    return handle


@contextlib.contextmanager
def gpu_queue_lock(owner):
    lease = gpu_lock(owner)
    try:
        yield lease
    finally:
        lease.close()


def verify_sources(freeze_path):
    value = read(freeze_path)
    for relative,digest in value['source_sha256'].items():
        assert sha(ROOT/relative) == digest,relative
    return value


def selection():
    guard()
    value = read(SELECTION)
    assert value['registered_by'] == 'root' and len(value['candidates']) == 2
    assert {row['kind'] for row in value['candidates']} == {'single','fusion'}
    return value
