"""Independent bounded RACE inference innovation; no training/test access."""
import os,shutil,time
from pathlib import Path
from scripts.race_v2_inference_a_20261029.common import ROOT,read,write,sha,now,rows,slices,CHECKPOINT_SHA
R=ROOT/'archive/research_records/2026-10-29_v2_inference_innovation_a'
STAGE=ROOT/'archive/research_records/2026-10-29_race_v2_stage'
BASE=ROOT/'archive/research_records/2026-10-29_race_v2_inference_a'
TMP=Path('/tmp/aflow_race_v2_20261029/inference_innovation_a')
CACHE=Path('/tmp/aflow_race_v2_20261029/shared_validation/native')
def guard():
    auth=read(STAGE/'AUTHORIZATION.json');assert time.time()<auth['workers']['1154121']['deadline_unix']
    assert shutil.disk_usage(ROOT).free>=20*2**30
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<=64*2**20
    if TMP.exists():assert shutil.disk_usage(TMP).free>=100*2**30
def verify():
    guard();frozen=read(R/'EXECUTION_FREEZE.json')
    for p,d in frozen['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(R/'PLAN.json')==frozen['plan_sha256']
    assert sha(BASE/'ATOMS_READY.json')==frozen['source_atoms_manifest_sha256']
    assert sha(STAGE/'STATISTICS_FREEZE.json')==frozen['statistics_freeze_sha256']

