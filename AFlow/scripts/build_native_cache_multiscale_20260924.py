"""Rebuild the old native train/validation pixels in isolated worker scratch."""
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

R = ROOT / 'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision'
CACHE = Path('/tmp/aflow_multiscale_native_20260924')
OLD = ROOT / 'archive/research_records/2026-09-21_segresnet_pretrain_native_res/DATA_AUDIT.json'
builder.CACHE_ROOT = CACHE


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def main():
    freeze = json.loads((R / 'CACHE_FREEZE.json').read_text())
    for relative, expected in freeze['source_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('frozen cache source changed: ' + relative)
    if (R / 'DATA_PREACCESS.json').exists() or (CACHE.exists() and any(CACHE.iterdir())):
        raise RuntimeError('prior access or partial scratch cache exists; inspect before retry')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('persistent free below20GiB')
    if shutil.disk_usage('/tmp').free < 100 * 2**30 + 29 * 2**30:
        raise RuntimeError('worker scratch space insufficient')
    manifest = {}
    subjects = {}
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
    write(R / 'DATA_PREACCESS.json', {'utc': now(), 'worker': 1109780, 'splits_pixels_opened': ['train', 'validation'],
                                     'test_metadata_only': True, 'test_pixels_access': False,
                                     'manifest_audit': manifest, 'cache_freeze_sha256': sha(R / 'CACHE_FREEZE.json')})
    old = json.loads(OLD.read_text())
    old_by_split = {x['role']: x for x in old['caches']}
    completed = []
    for split in ['train', 'validation']:
        write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'building_native_cache', 'split': split,
              'running': True, 'worker': 1109780, 'training_started': False, 'test_pixels_access': False})
        data = builder.build(split, processes=12)
        prior = old_by_split[split]
        if data['images_sha256'] != prior['images_sha256'] or data['labels_sha256'] != prior['labels_sha256']:
            raise RuntimeError(split + ' native payload differs from historical frozen reference')
        completed.append(data)
    write(R / 'DATA_AUDIT.json', {'utc': now(), 'status': 'passed', 'worker': 1109780,
          'manifest': manifest, 'patient_isolation': True, 'caches': completed,
          'historical_pixel_payload_exact': True, 'test_pixels_access': False})
    write(R / 'PIPELINE_STATUS.json', {'utc': now(), 'state': 'cache_complete', 'running': False,
          'worker': 1109780, 'training_started': False, 'test_pixels_access': False})
    print(json.dumps({'status': 'cache_complete', 'splits': [c['role'] for c in completed]}), flush=True)


if __name__ == '__main__':
    main()
