"""Copy already-complete inference scratch to head scratch and verify every SHA.

This is recovery I/O only: no decoding, labels scored, or neural forwards.
Original worker paths and historical manifests remain unchanged.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

from .governance import ROOT, RECORD, atomic_json, sha

BACKUP = Path('/tmp/aflow_race_v2_inference_backup_20261029')
OLD = ROOT / 'archive/research_records/2026-10-29_race_e164_inference_test270'
HOST = 'tiger@[fdbd:dccd:cde2:1086:109d:c0df:800c:b0e2]'


def copy(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['scp', '-rq', '-P', '10725', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    '-o', 'StrictHostKeyChecking=yes', f'{HOST}:{remote}', str(local)], check=True, timeout=1800)


def main():
    # Auto-review explicitly rejected this destination for clinical payloads.
    # Keep the reviewable plan, but do not retry through another tool or agent.
    raise RuntimeError('Disabled: destination transfer rejected by automatic approval review; no copy performed')


def reviewed_copy_plan_not_authorized():
    if shutil.disk_usage('/tmp').free < 120 * 2**30:
        raise RuntimeError('head scratch recovery reserve below120GiB')
    BACKUP.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source = Path('/tmp/aflow_race_e164_test270_20261029')
    target = BACKUP / source.name
    target.mkdir(exist_ok=True)
    for name in ('native', 'canonical128', 'atoms'):
        copy(source / name, target / name)
        atomic_json(RECORD / 'INFERENCE_BACKUP_PROGRESS.json',
                    {'state': 'copying', 'completed_directory': name, 'elapsed_seconds': time.monotonic() - started})
    ready = json.loads((OLD / 'CACHE_READY.json').read_text())
    expected = [(target / 'native/index.json', ready['index_sha256'])]
    for name, relative in [('native_image', 'native/images.f32'), ('native_label', 'native/labels.u8'),
                           ('canon_image', 'canonical128/images.f32'), ('canon_label', 'canonical128/labels.u8')]:
        expected.append((target / relative, ready['payload_sha256'][name]))
    for rank in (0, 1):
        report = json.loads((OLD / f'FORWARD_COMPLETE_{rank}.json').read_text())
        for atom in report['atoms']:
            expected.append((target / Path(atom['path']).relative_to(source), atom['sha256']))
    def check(item):
        path, digest = item
        assert sha(path) == digest, path
        return path.stat().st_size
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        sizes = list(pool.map(check, expected))
    atomic_json(RECORD / 'INFERENCE_RECOVERY_BACKUP.json', dict(
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), state='passed',
        old270_files_verified=len(expected), old270_bytes=sum(sizes), old270_atoms=2160,
        source_worker=1154121, source_prefix=str(source), head_scratch_prefix=str(target),
        all_payload_and_atom_sha_exact=True, original_paths_unchanged=True,
        new_neural_forwards=0, new_scoring=0, elapsed_seconds=time.monotonic() - started,
        note='Recovery copy only; manifests still record original worker paths; relocate via prefix explicitly'))
    print(json.dumps({'state': 'passed', 'files': len(expected), 'bytes': sum(sizes)}))


if __name__ == '__main__':
    main()
