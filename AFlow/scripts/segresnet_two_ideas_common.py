"""2026-09-21 two candidates, sequential on two A100s; no test-tuned training."""
import os, json, datetime, hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.segresnet_plain972_common import (
    ROOT, TRAIN_CACHE, VAL_CACHE, seed_all, safe_augment, summary_ci,
    BrainCropDataset, validate, dice_per_case, learning_rate_for_epoch,
    model_for as binary_model_for, disk, save, sha, now, write)
from scripts.train_t1ce_wt_capg3d import CACHE_SCHEMA, augment_batch
from sammed3d.t1ce_wt_cached import MODEL_SHAPE
from sammed3d.segresnet_semantic_wt import SemanticSegResNet, SemanticLoss, wt_logit

R = ROOT/'archive/research_records/2026-09-21_segresnet_two_experiments'
OUT = ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921'
RECORDS = {
    'semantic_regions': ROOT/'archive/research_records/2026-09-21_segresnet_semantic_regions',
    'finetune_pretrained': ROOT/'archive/research_records/2026-09-21_segresnet_external_pretrain'}
SOURCE_INVENTORY = ROOT/'archive/reviews/2026-09-06_openneuro_ds007045_source_audit/SOURCE_INVENTORY.json'
SCRATCH = Path('/tmp/aflow_two_ideas_20260921')
RAW_DIR = SCRATCH/'raw'; PRETRAIN_CACHE = SCRATCH/'cache_pretrain334'; CLASS_CACHE = SCRATCH/'classes972'
BASELINE_RUN = ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917'
BASELINE_SEED = PRETRAIN_SEED = 20260819
PRETRAIN_SPLIT = 'pretrain_openneuro_gbm334'
PRETRAIN_EPOCHS = 150; FINETUNE_EPOCHS = 200
PER_GPU_BATCH = 4; WORLD = 2; PARAMETERS = 4_700_897
RUNS = [('semantic_regions', 'semantic', BASELINE_SEED), ('pretrain', None, PRETRAIN_SEED), ('finetune_pretrained', 'pretrained', BASELINE_SEED)]
MANIFEST = {k: ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'/v for k,v in
            [('train','train_t1ce_wt.jsonl'), ('validation','validation_t1ce_wt.jsonl'), ('test','sealed_test/test_cases_t1ce.jsonl')]}


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def verify():
    d=json.loads((R/'FREEZE.json').read_text())
    for name,digest in d['files'].items():
        if sha(ROOT/name)!=digest: raise RuntimeError('frozen file changed: '+name)
    return d


def selected_cases():
    inv=json.loads(SOURCE_INVENTORY.read_text()); cases=inv['selected']['cases']
    assert len(cases)==len({c['subject_id'] for c in cases})==334
    return inv,cases


def cache_case_id(subject_id): return f'{subject_id}-0'
def steps_per_epoch(n,per_gpu=PER_GPU_BATCH): return -(-n//(WORLD*per_gpu))


def batches_for_epoch(n,seed,epoch,rank,per_gpu=PER_GPU_BATCH):
    if n%2 or rank not in (0,1): raise ValueError('equal two-rank split required')
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    for start in range(0,n,WORLD*per_gpu):
        whole=order[start:start+WORLD*per_gpu]; k=len(whole)//2
        yield whole[rank*k:(rank+1)*k]


def schedule(mode): return (150,75) if mode=='pretrain' else (200,100)
def lr_for(mode,epoch):
    total,const=schedule(mode)
    return learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=const,total_epochs=total)
def model_for(device,semantic=False):
    return SemanticSegResNet().to(device) if semantic else binary_model_for(device)


class ClassDataset(BrainCropDataset):
    def __init__(self,case_ids):
        super().__init__(TRAIN_CACHE,expected_split='train',case_ids=case_ids)
        index=json.loads((CLASS_CACHE/'index.json').read_text())
        if index['case_ids']!=self.case_ids or index['count']!=972 or index['test_access']:
            raise ValueError('class cache role/order mismatch')
        self._classes=None
    def __getitem__(self,slot):
        row=super().__getitem__(slot)
        if self._classes is None:
            self._classes=np.memmap(CLASS_CACHE/'classes.u8',dtype=np.uint8,mode='r',shape=(972,128,128,128))
        c=np.array(self._classes[slot]); c[c==4]=0
        if not np.array_equal(c>0,row['label'].numpy()[0]>0): raise ValueError('WT mismatch')
        row['label']=torch.from_numpy(c.astype(np.float32))[None]
        return row


def paired(candidate,baseline):
    right={r['case_id']:r for r in baseline}
    if len(right)!=len(candidate) or set(right)!={r['case_id'] for r in candidate}: raise ValueError('case sets differ')
    if any(r['subject_id']!=right[r['case_id']]['subject_id'] for r in candidate): raise ValueError('patients differ')
    subjects=sorted({r['subject_id'] for r in candidate})
    n=np.array([sum(r['subject_id']==s for r in candidate) for s in subjects])
    delta=np.array([r['locator_dsc']-right[r['case_id']]['locator_dsc'] for r in candidate])
    sums=np.array([sum(d for r,d in zip(candidate,delta) if r['subject_id']==s) for s in subjects])
    ix=np.random.default_rng(20260918).integers(0,len(subjects),(10000,len(subjects)))
    scan=sums[ix].sum(1)/n[ix].sum(1); sub=(sums/n)[ix].mean(1)
    return {'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),
            'scan_delta_ci95':np.quantile(scan,[.025,.975]).tolist(), 'subject_delta_ci95':np.quantile(sub,[.025,.975]).tolist(),
            'scan_delta_ci97_5':np.quantile(scan,[.0125,.9875]).tolist(),
            'subject_delta_ci97_5':np.quantile(sub,[.0125,.9875]).tolist(),
            'point_hits_delta':sum(r['automatic_point_hits_gt'] for r in candidate)-sum(r['automatic_point_hits_gt'] for r in baseline),
            'improved_scans':int((delta>0).sum()),'worse_scans':int((delta<0).sum()),'tied_scans':int((delta==0).sum()),
            'bootstrap':'10000 patient-cluster paired percentile, seed20260918; 97.5% for two planned primary comparisons'}
