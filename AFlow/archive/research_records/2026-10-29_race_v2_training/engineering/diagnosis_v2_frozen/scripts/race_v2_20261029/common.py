"""New-stage paths and guards. Never inherit a historical experiment deadline."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'archive/research_records/2026-10-29_race_v2_training'
STAGE = ROOT / 'archive/research_records/2026-10-29_race_v2_stage'
TMP = Path('/tmp/aflow_race_v2_20261029')
CACHE = TMP / 'native'
BACKUP = Path('/tmp/aflow_race_v2_training_backup_20261029')
SOURCE = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/race_20261028/race/best.pth'
SOURCE_SHA = '4bd6c35f7e12ecdf1540bf88128f9d3f9853340d801b32e77779fc4ba56eb3d7'
CANVAS = (160, 208, 160)
SEED = 20261029
PY = ROOT.parent / 'sammed3d-venv/bin/python'

def now(): return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)
def env():
    result = os.environ.copy()
    result.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR=str(TMP/'mpl'))
    return result
def disk(extra=0):
    if shutil.disk_usage(ROOT).free-extra < 20*2**30:
        raise RuntimeError('persistent free below20GiB')
    used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file()) if R.exists() else 0
    if used+extra > .5*2**30: raise RuntimeError('training persistent allocation exceeds0.50GiB')
def deadline(required=False):
    path = STAGE/'AUTHORIZATION.json'
    authorization = read(path) if path.exists() else {}
    stop = authorization.get('workers',{}).get('1154259',{}).get(
        'training_deadline_unix' if required else 'deadline_unix')
    if stop is None:
        if required: raise RuntimeError('new-stage resource deadline has not been supplied')
        return
    if time.time() >= stop: raise RuntimeError('new-stage resource deadline reached')
def verify_sources(name='EXECUTION_FREEZE.json'):
    for path, digest in read(R/name)['source_sha256'].items():
        if sha(ROOT/path) != digest: raise RuntimeError('frozen source changed: '+path)
def rows(split):
    if split not in ('train','validation'): raise ValueError('training code disallows test rows')
    from scripts.nnunet_resenc_20261013.common import rows as load
    return load(split)
