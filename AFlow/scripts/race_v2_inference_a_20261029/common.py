"""New bounded validation study; historical records remain immutable."""
import datetime, hashlib, json, os, shutil, time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'archive/research_records/2026-10-29_race_v2_inference_a'
STAGE=ROOT/'archive/research_records/2026-10-29_race_v2_stage'
TMP=Path('/tmp/aflow_race_v2_20261029/inference_a')
CACHE=Path('/tmp/aflow_race_v2_20261029/shared_validation/native')
OLD=ROOT/'archive/research_records/2026-10-29_race_e164_inference_test270'
CHECKPOINT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/race_20261028/race/best.pth'
CHECKPOINT_SHA='4bd6c35f7e12ecdf1540bf88128f9d3f9853340d801b32e77779fc4ba56eb3d7'
CANVAS=(160,208,160)
def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def rows():
    p=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def slices(g):return tuple(slice(int(a),int(a)+int(n)) for a,n in zip(g['canvas_offset'],g['extent']))
def guard():
    assert shutil.disk_usage(ROOT).free>=20*2**30,'persistent free <20GiB'
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<64*2**20,'A text budget'
    if TMP.exists():assert shutil.disk_usage(TMP).free>=100*2**30,'scratch free <100GiB'
    for path in [STAGE/'AUTHORIZATION.json',R/'AUTHORIZATION.json']:
        if path.exists():
            auth=read(path)
            limit=auth.get('deadline_unix',auth.get('workers',{}).get('1154121',{}).get('deadline_unix'))
            if limit is not None:assert time.time()<limit,'new authorization deadline'
def verify():
    guard()
    for p,d in read(R/'EXECUTION_FREEZE.json')['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
