import os,json,datetime,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.segresnet_plain972_common import ROOT,model_for,summary_ci,seed_all,sha,write,now
from scripts.segresnet_two_ideas_common import paired
from scripts.segresnet_native_res_test_common import (TEST_CANVAS,TEST_SCANS,TEST_SUBJECTS,SCHEMA,IMAGE_TOLERANCE,
    canvas_offset,canvas_slices,pad_to_canvas,case_metrics,NativeTestDataset as OldNativeTestDataset)
from scripts.train_t1ce_wt_capg3d import dice_per_case,summarize_cases,validate
S=ROOT/'archive/research_records/2026-09-21_segresnet_followup_test270'
RECORDS={
 'native':ROOT/'archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270',
 'patient':ROOT/'archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270'}
TRAIN_RECORDS={'native':ROOT/'archive/research_records/2026-09-21_segresnet_pretrain_native_res',
               'patient':ROOT/'archive/research_records/2026-09-21_segresnet_patient_equal_loss'}
EPOCHS={'native':166,'patient':172}
HASHES={'native':'b2a37a7ed4ea73922a454f5f60995d3a8807f18d122539ba0c85d9b5e62c8a18',
        'patient':'46ad9f1ccbb85a2c8f7790a2d225265ddeb50c667f8091e08bf8006e4039083d'}
MANIFEST={k:ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'/v for k,v in [('train','train_t1ce_wt.jsonl'),('validation','validation_t1ce_wt.jsonl'),('test','sealed_test/test_cases_t1ce.jsonl')]}
CANONICAL_TEST=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'
TEST_CACHE=Path('/tmp/aflow_followup_test270_20260921/native_test')
OLD_NATIVE=ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270'
PRETRAINED=ROOT/'archive/research_records/2026-09-21_segresnet_external_pretrain'
PURE=ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270'
REF_SHA='7006d44a2a2f765562dc0155f914c967a27b22ffba476cea5e966f01d69f7ad0'


def read_rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write_new(p,data):
    with Path(p).open('x') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def verify():
    d=json.loads((S/'FREEZE.json').read_text())
    for n,h in d['files'].items():assert sha(ROOT/n)==h,n
    return d
def numeric_mode():
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
def validate_rows(rows):
    expected=read_rows(MANIFEST['test'])
    assert len(rows)==len({r['case_id'] for r in rows})==270 and len({r['subject_id'] for r in rows})==119
    assert [(r['case_id'],r['subject_id']) for r in rows]==[(r['case_id'],r['subject_id']) for r in expected]
    assert all(np.isfinite(r['locator_dsc']) and 0<=r['locator_dsc']<=1 for r in rows)


class NativeTestDataset(OldNativeTestDataset):
    def __init__(self,path,ids):
        index=json.loads((Path(path)/'index.json').read_text())
        if index['count']!=270 or len(ids)!=270 or len(set(ids))!=270 or index['case_ids']!=list(ids):
            raise ValueError('test case identity/count mismatch')
        super().__init__(path,ids)
