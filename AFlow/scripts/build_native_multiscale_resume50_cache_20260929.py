"""Same native pixel recipe on the new worker; no old receipts are overwritten."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_segresnet_pretrained_native_cache as builder

R = ROOT / 'archive/research_records/2026-09-29_segresnet_native_multiscale_resume50'
CACHE = Path('/tmp/aflow_multiscale_resume50_20260929')
OLD = ROOT / 'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/DATA_AUDIT.json'
builder.CACHE_ROOT = CACHE


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def main():
    freeze = json.loads((R / 'CACHE_FREEZE.json').read_text())
    for p, h in freeze['source_sha256'].items():
        if sha(ROOT / p) != h:
            raise RuntimeError('cache source changed: ' + p)
    if (R / 'DATA_PREACCESS.json').exists() or (CACHE.exists() and any(CACHE.iterdir())):
        raise RuntimeError('prior cache access exists; inspect before recovery')
    if shutil.disk_usage(ROOT).free < 20 * 2**30 or shutil.disk_usage('/tmp').free < 129 * 2**30:
        raise RuntimeError('insufficient persistent/scratch disk')
    subjects, manifest = {}, {}
    for split, (scans, patients) in [('train', (972, 444)), ('validation', (108, 50)), ('test', (270, 119))]:
        path = builder.MANIFEST[split]
        rows = builder.read_rows(path)
        if sha(path) != freeze['manifest_sha256'][split] or len(rows) != scans or len({r['case_id'] for r in rows}) != scans:
            raise RuntimeError(split + ' manifest mismatch')
        subjects[split] = {r['subject_id'] for r in rows}
        if len(subjects[split]) != patients:
            raise RuntimeError(split + ' patient count mismatch')
        manifest[split] = {'scans': scans, 'patients': patients, 'sha256': sha(path)}
    if any(subjects[a] & subjects[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise RuntimeError('patient overlap')
    CACHE.mkdir(parents=True)
    write(R / 'DATA_PREACCESS.json', {'utc': now(), 'worker': 1114913,
          'splits_pixels_opened': ['train', 'validation'], 'test_metadata_only': True,
          'test_pixels_access': False, 'manifest_audit': manifest,
          'cache_freeze_sha256': sha(R / 'CACHE_FREEZE.json')})
    old = json.loads(OLD.read_text())
    references = {d['role']: d for d in old['caches']}
    completed = []
    for split in ['train', 'validation']:
        write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'rebuilding_resume_cache',
              'split': split, 'worker': 1114913, 'running': True, 'resume_training_started': False,
              'original_complete_epochs': 50, 'test_pixels_access': False})
        result = builder.build(split, processes=24)
        if any(result[k] != references[split][k] for k in ['images_sha256', 'labels_sha256']):
            raise RuntimeError(split + ' payload differs from original stage')
        completed.append(result)
    write(R / 'DATA_AUDIT.json', {'utc': now(), 'status': 'passed', 'worker': 1114913,
          'manifest': manifest, 'patient_isolation': True, 'caches': completed,
          'historical_pixel_payload_exact': True, 'test_pixels_access': False,
          'old_data_audit_sha256': sha(OLD)})
    write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'cache_complete',
          'running': False, 'worker': 1114913, 'original_complete_epochs': 50,
          'resume_training_started': False, 'test_pixels_access': False})
    print(json.dumps({'state': 'cache_complete', 'historical_payload_exact': True}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(R / 'CACHE_FAILURE.json', {'utc': now(), 'reason': repr(exc),
              'resume_training_started': False, 'test_pixels_access': False})
        write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'cache_failed',
              'running': False, 'worker': 1114913, 'resume_training_started': False,
              'test_pixels_access': False})
        raise
