import os,time,shutil
from pathlib import Path
from scripts.race_v2_inference_a_20261029.common import ROOT,read,write,sha,now,slices,CHECKPOINT,CHECKPOINT_SHA
BASE=ROOT/'archive/research_records/2026-10-29_race_v2_final270_recovery'
R=BASE/'a';TMP=Path('/tmp/aflow_race_v2_final270_recovery_20261029');CACHE=TMP/'native';CANON=TMP/'canonical128'
OLD=ROOT/'archive/research_records/2026-10-29_race_e164_inference_test270'
SELECTION=ROOT/'archive/research_records/2026-10-29_race_v2_stage/FINAL_TEST_SELECTION.json'
SELECTION_SHA='5b74021964fe8b6d0f34fbdb35303d7d05be34d6796caec2adc6e6a68c6579bc'
MANIFEST=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
CANON_INDEX=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270/index.json'
LOCK=TMP/'locks/worker1154259_gpu0.lock'

def guard():
    assert (BASE/'AUTHORIZATION.json').exists(),'new recovery authorization required'
    authorization=read(BASE/'AUTHORIZATION.json')
    deadline=authorization.get('deadline_unix',authorization.get('workers',{}).get('1154259',{}).get('deadline_unix'))
    assert deadline is not None and time.time()<deadline,'recovery deadline missing/expired'
    assert sha(SELECTION)==SELECTION_SHA,'original final selection cannot change'
    assert shutil.disk_usage(ROOT).free>=20*2**30
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<64*2**20
    if TMP.exists():assert shutil.disk_usage(TMP).free>=100*2**30

def cohort():
    rows=[__import__('json').loads(s) for s in MANIFEST.read_text().splitlines() if s.strip()]
    assert len(rows)==270 and len({r['case_id'] for r in rows})==270 and len({r['subject_id'] for r in rows})==119
    return rows

def verify():
    guard();f=read(R/'EXECUTION_FREEZE.json')
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(CHECKPOINT)==CHECKPOINT_SHA and sha(SELECTION)==f['selection_sha256']
    return f
