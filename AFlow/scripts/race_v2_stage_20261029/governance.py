"""Resource ownership, deadline and persistent-disk guards (no model access)."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / 'archive/research_records/2026-10-29_race_v2_stage'
SCRATCH = Path('/tmp/aflow_race_v2_stage_20261029')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def guard(worker, *, training=False):
    authorization = json.loads((RECORD / 'AUTHORIZATION.json').read_text())
    allocation = authorization['workers'][str(worker)]
    key = 'training_deadline_unix' if training else 'deadline_unix'
    if time.time() >= allocation[key]:
        raise RuntimeError(f'{worker} {key} reached; no automatic budget extension')
    free = shutil.disk_usage(ROOT).free
    if free < authorization['persistent_free_min_bytes']:
        raise RuntimeError(f'persistent disk below 20GiB: {free} bytes')
    roots = [ROOT / p for p in authorization['new_persistent_roots']]
    added = sum(p.stat().st_size for folder in roots for p in folder.rglob('*') if p.is_file())
    if added > authorization['persistent_new_max_bytes']:
        raise RuntimeError(f'new persistent allocation exceeded: {added} bytes')
    return {'persistent_free_bytes': free, 'stage_bytes': added, 'deadline_unix': allocation[key]}


@contextlib.contextmanager
def gpu_queue_lock(worker, devices, owner):
    """Hold every device in ascending order throughout a queue's GPU work."""
    allocation = json.loads((RECORD / 'RESOURCE_ALLOCATION.json').read_text())
    expected = allocation['queues'][owner]
    if int(worker) != expected['worker'] or sorted(devices) != expected['gpus']:
        raise RuntimeError('queue does not own these GPUs')
    locks = []
    directory = Path('/tmp/aflow_race_v2_20261029/locks')
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for device in sorted(devices):
            handle = (directory / f'worker{worker}_gpu{device}.lock').open('a+')
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            json.dump({'worker': worker, 'gpu': device, 'owner': owner, 'pid': os.getpid(),
                       'started_unix': time.time()}, handle)
            handle.flush()
        yield
    finally:
        for handle in reversed(locks):
            handle.close()
