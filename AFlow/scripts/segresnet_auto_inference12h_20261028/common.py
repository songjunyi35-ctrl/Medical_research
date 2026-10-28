"""Persistent queue protocol for the explicitly test-tuned 12h inference study."""
import json
import os
from pathlib import Path
import shutil
import time
from scripts.segresnet_inference_search_20261028.common import (
    ROOT, PY, CHECKPOINT, CHECKPOINT_SHA, now, sha, read, write, load_model, numeric_mode, slices)

R=ROOT/'archive/research_records/2026-10-28_segresnet_auto_inference12h'
TMP=Path('/tmp/aflow_auto_inference12h_20261028')
CACHE=Path('/tmp/aflow_inference_selected_test270_20261028/native')
CANVAS=(160,208,168)
OLD=ROOT/'archive/research_records/2026-10-28_segresnet_inference_selected_test270'


def guard():
    auth=read(R/'AUTHORIZATION.json')
    assert time.time()<auth['deadline_unix'], '12h window ended'
    assert shutil.disk_usage(ROOT).free>=20*2**30, 'persistent reserve below20GiB'
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())<2*2**30, 'new persistent above2GiB'
    if TMP.exists():assert shutil.disk_usage(TMP).free>=70*2**30, 'scratch reserve below70GiB'


def status(state,**kwargs):
    write(R/'PIPELINE_STATUS.json',dict(at=now(),state=state,worker=1152682,
          training=False,test_tuned=True,test_access=True,**kwargs))


def data():
    import numpy as np
    index=read(CACHE/'index.json')
    manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    assert len(rows)==270 and len({d['subject_id'] for d in rows})==119
    assert index['case_ids']==[d['case_id'] for d in rows] and tuple(index['canvas'])==CANVAS
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS))
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))
    return rows,index['geometry'],images,labels


def save_array(path,array):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:np.save(f,array)
    os.replace(temp,path)


def verify_job(job):
    guard()
    assert job['test_tuned'] is True and job['checkpoint_sha256']==CHECKPOINT_SHA
    for path,digest in job['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(CHECKPOINT)==CHECKPOINT_SHA


def append_log(name,value):
    with (R/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def env():
    e=os.environ.copy();e.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
      MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(TMP/'mpl'))
    return e
