import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'archive/research_records/2026-10-28_race_seg'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/race_20261028'
TMP=Path('/tmp/aflow_race_20261028')
CACHE=TMP/'native'
PY=ROOT.parent/'sammed3d-venv/bin/python'
CANVAS=(160,208,160)
SEED=20261028
EPOCHS=200
ARMS=('control','race')

def now(): return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)
def env():
    result=os.environ.copy()
    result.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(TMP/'mpl'))
    return result
def disk(extra=0):
    if shutil.disk_usage(ROOT).free-extra<20*2**30: raise RuntimeError('persistent free below20GiB')
    used=sum(p.stat().st_size for base in (R,OUT) if base.exists() for p in base.rglob('*') if p.is_file())
    if used+extra>4*2**30: raise RuntimeError('new persistent data exceeds4GiB')
def deadline():
    if time.time()>=read(R/'AUTHORIZATION.json')['deadline_unix']: raise RuntimeError('24h budget reached')
def verify_sources(name):
    for path,digest in read(R/name)['source_sha256'].items():
        if sha(ROOT/path)!=digest: raise RuntimeError('frozen source changed: '+path)
def rows(split):
    from scripts.nnunet_resenc_20261013.common import rows as load
    return load(split)
def validate_manifests():
    from scripts.nnunet_resenc_20261013.common import verify_manifests
    return verify_manifests()
