import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'archive/research_records/2026-10-29_race_e164_inference_test270'
TMP=Path('/tmp/aflow_race_e164_test270_20261029')
CACHE=TMP/'native';CANON=TMP/'canonical128'
PY=ROOT.parent/'sammed3d-venv/bin/python'
CHECKPOINT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/race_20261028/race/best.pth'
CHECKPOINT_SHA='4bd6c35f7e12ecdf1540bf88128f9d3f9853340d801b32e77779fc4ba56eb3d7'
CANVAS=(160,208,168)
WORKER=1154121
SOURCE_RECIPE=ROOT/'archive/research_records/2026-10-28_segresnet_auto_inference12h/PORTABLE_RECIPE_CURRENT.json'
HISTORICAL_CACHE=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270/CACHE_READY.json'
CANON_INDEX=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/index.json'
from scripts.followup_test270_common import MANIFEST,SCHEMA,summary_ci,paired,validate_rows,numeric_mode


def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def exclusive(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def rows(split='test'):return [json.loads(s) for s in MANIFEST[split].read_text().splitlines() if s.strip()]
def guard():
    assert time.time()<read(R/'AUTHORIZATION.json')['deadline_unix'],'evaluation budget ended'
    assert shutil.disk_usage(ROOT).free>=20*2**30,'persistent free below20GiB'
    used=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
    assert used<=2*2**30,'new persistent above2GiB'
    if TMP.exists():assert shutil.disk_usage(TMP).free>=70*2**30,'scratch free below70GiB'
def verify(name='EXECUTION_FREEZE.json'):
    for path,digest in read(R/name)['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    guard()
def env():
    e=os.environ.copy();e.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
          MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(TMP/'mpl'))
    return e
def slices(g):return tuple(slice(int(o),int(o)+int(n)) for o,n in zip(g['canvas_offset'],g['extent']))


def manifests():
    expected={'train':(972,444),'validation':(108,50),'test':(270,119)};subjects={}
    for role in expected:
        data=rows(role);subjects[role]={d['subject_id'] for d in data}
        assert (len(data),len(subjects[role]))==expected[role]
        assert len({d['case_id'] for d in data})==len(data)
    assert not any(subjects[a]&subjects[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    assert read(CANON_INDEX)['case_ids']==[d['case_id'] for d in rows()]
