import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'archive/research_records/2026-10-13_nnunet_resenc_reference'
OUT = ROOT / 'workspace/BraTS/nnunet_t1ce_wt/resenc_20261013'
TMP = Path('/tmp/aflow_nnunet_20261013')
ENV = TMP / 'env'
PY = ENV / 'bin/python'
BASE_PY = ROOT.parent / 'sammed3d-venv/bin/python'
DATASET = 'Dataset813_AFlowT1ce'
PLANS = 'nnUNetResEncUNetLPlans'
RAW = TMP / 'raw' / DATASET
PRE = TMP / 'preprocessed' / DATASET
NATIVE = TMP / 'native'
MANIFEST_DIR = ROOT / 'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'
MANIFESTS = {'train':MANIFEST_DIR/'train_t1ce_wt.jsonl',
 'validation':MANIFEST_DIR/'validation_t1ce_wt.jsonl',
 'test':MANIFEST_DIR/'sealed_test/test_cases_t1ce.jsonl'}
EXPECTED_SHA = {'train':'6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65',
 'validation':'f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb',
 'test':'a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9'}

def now(): return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,d):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,path)
def rows(split): return [json.loads(s) for s in MANIFESTS[split].read_text().splitlines() if s.strip()]
def verify_manifests():
    subjects={};summary={}
    for s,n,m in [('train',972,444),('validation',108,50),('test',270,119)]:
        a=rows(s);subjects[s]={r['subject_id'] for r in a}
        assert len(a)==len({r['case_id'] for r in a})==n and len(subjects[s])==m
        assert sha(MANIFESTS[s])==EXPECTED_SHA[s]
        summary[s]={'scans':n,'subjects':m,'sha256':EXPECTED_SHA[s]}
    assert not (subjects['train']&subjects['validation'] or subjects['train']&subjects['test'] or subjects['validation']&subjects['test'])
    return summary
def disk():
    free=shutil.disk_usage(ROOT).free
    if free<20*2**30:raise RuntimeError('persistent free space below20GiB')
    return free
def env():
    d=os.environ.copy();d.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),
      nnUNet_raw=str(TMP/'raw'),nnUNet_preprocessed=str(TMP/'preprocessed'),
      nnUNet_results=str(OUT),nnUNet_compile='false',nnUNet_n_proc_DA='8',
      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
      NCCL_DEBUG='WARN',NCCL_IB_DISABLE='1',MPLCONFIGDIR=str(TMP/'mpl'))
    return d
def verify_source_freeze(name):
    d=read(R/name)
    for p,h in d['source_sha256'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('frozen source changed: '+p)
    return d
