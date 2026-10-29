"""Independent record and scratch locations; never modify older freezes."""
import shutil,time
from pathlib import Path
from scripts.race_v2_inference_a_20261029.common import ROOT,STAGE,CACHE,CHECKPOINT,CHECKPOINT_SHA,read,write,sha,now,rows,slices
BASE=ROOT/'archive/research_records/2026-10-29_race_v2_inference_a'
R=ROOT/'archive/research_records/2026-10-29_race_v2_inference_normalization_a'
TMP=Path('/tmp/aflow_race_v2_20261029/inference_normalization_a')
LOCK=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock')

def guard():
    assert shutil.disk_usage(ROOT).free>=20*2**30
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<64*2**20
    if TMP.exists():assert shutil.disk_usage(TMP).free>=100*2**30
    limit=read(STAGE/'AUTHORIZATION.json')['workers']['1154121']['deadline_unix']
    assert time.time()<limit

def verify():
    guard();freeze=read(R/'PREPARATION_FREEZE.json')
    for p,digest in freeze['source_sha256'].items():assert sha(ROOT/p)==digest,p
    assert sha(R/'PLAN.json')==freeze['plan_sha256']
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    return freeze
