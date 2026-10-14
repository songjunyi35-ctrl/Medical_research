"""Shared frozen contract: supervised WT pretraining on OpenNeuro ds007045 (334 pre-op GBM, CC0, single T1ce),
then fine-tuning on the fixed original972 with the unchanged pure-SegResNet recipe; matched random-init controls.

Everything downstream of initialization is the frozen 2026-09-17 baseline recipe on the SAME canonical 128^3 caches,
so validation108 numbers are directly comparable to the e193 baseline (0.740032).  No test access.
"""
import os,json,hashlib,datetime
from pathlib import Path
import numpy as np
import torch
from scripts.segresnet_plain972_common import (model_for,safe_augment,seed_all,summary_ci,TRAIN_CACHE,VAL_CACHE,
    learning_rate_for_epoch,BrainCropDataset,validate,dice_per_case)
from scripts.train_t1ce_wt_capg3d import CACHE_SCHEMA
from sammed3d.t1ce_wt_cached import MODEL_SHAPE

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-20_segresnet_openneuro_pretrain_finetune'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/openneuro_pretrain_finetune_20260920'
SOURCE_INVENTORY=ROOT/'archive/reviews/2026-09-06_openneuro_ds007045_source_audit/SOURCE_INVENTORY.json'
SCRATCH=Path('/tmp/aflow_openneuro_pretrain_20260920')
RAW_DIR=SCRATCH/'raw';PRETRAIN_CACHE=SCRATCH/'cache_pretrain334'
BASELINE_RUN=ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917'   # random-init seed A (e193 selected)
BASELINE_SEED=20260819
MANIFEST={'train':ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl','validation':ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl','test':ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'}
PRETRAIN_SPLIT='pretrain_openneuro_gbm334'
PRETRAIN_SEED=20260821
PRETRAIN_EPOCHS=150;PRETRAIN_CONSTANT_EPOCHS=75
FINETUNE_EPOCHS=200;FINETUNE_CONSTANT_EPOCHS=100
PER_GPU_BATCH=4;WORLD=2
SEEDS={'A':BASELINE_SEED,'B':20260820}
RUNS=[('pretrain',None,PRETRAIN_SEED),('finetune_pretrained_seedA','pretrained',SEEDS['A']),('finetune_pretrained_seedB','pretrained',SEEDS['B']),('finetune_random_seedB','random',SEEDS['B'])]
WALL_BUDGET_SECONDS=8*3600
FINETUNE_ESTIMATED_SECONDS=2.0*3600
PARAMETERS=4_700_897


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def disk():
    import shutil
    free=shutil.disk_usage(ROOT).free
    if free<2*(1<<30):raise RuntimeError('persistent disk below 2GiB reserve')
    return free
def save(path,data):
    disk();tmp=path.with_suffix('.tmp.pth');torch.save(data,tmp);os.replace(tmp,path)


def verify():
    d=json.loads((R/'FREEZE.json').read_text())
    for name,digest in d['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen file changed: '+name)
    return d


def selected_cases():
    """Frozen 334 (subject_id, image url/sha/bytes, label url/sha/bytes) from the 2026-09-06 versioned inventory."""
    inv=json.loads(SOURCE_INVENTORY.read_text());cases=inv['selected']['cases']
    assert len(cases)==334 and len({c['subject_id'] for c in cases})==334
    return inv,cases


def cache_case_id(subject_id):
    """BrainCropDataset derives subject ids with rsplit('-',1) and requires a numeric suffix."""
    return f'{subject_id}-0'


def steps_per_epoch(n,per_gpu=PER_GPU_BATCH):return -(-n//(WORLD*per_gpu))


def batches_for_epoch(n,seed,epoch,rank,per_gpu=PER_GPU_BATCH):
    """Every case exactly once per epoch, no replacement, tail kept, equal count on both ranks (n must be even)."""
    if n%2:raise ValueError('n must be even for an equal two-rank split')
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    for start in range(0,n,WORLD*per_gpu):
        whole=order[start:start+WORLD*per_gpu];k=len(whole)//2;yield whole[rank*k:(rank+1)*k]


def schedule(mode):
    return (PRETRAIN_EPOCHS,PRETRAIN_CONSTANT_EPOCHS) if mode=='pretrain' else (FINETUNE_EPOCHS,FINETUNE_CONSTANT_EPOCHS)


def lr_for(mode,epoch):
    total,const=schedule(mode);return learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=const,total_epochs=total)


def paired(left,right,seed=20260918):
    """right - left, patient-cluster paired bootstrap (same convention as the native-resolution study)."""
    assert [(x['case_id'],x['subject_id']) for x in left]==[(x['case_id'],x['subject_id']) for x in right]
    subjects=sorted({r['subject_id'] for r in left});n=np.array([sum(r['subject_id']==s for r in left) for s in subjects])
    dif=np.array([b['locator_dsc']-a['locator_dsc'] for a,b in zip(left,right)])
    sums=np.array([sum(d for r,d in zip(left,dif) if r['subject_id']==s) for s in subjects]);ix=np.random.default_rng(seed).integers(0,len(subjects),(10000,len(subjects)))
    return {'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),'scan_delta_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_delta_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'point_hits_delta':int(sum(r['automatic_point_hits_gt'] for r in right)-sum(r['automatic_point_hits_gt'] for r in left)),'improved_scans':int((dif>0).sum()),'worse_scans':int((dif<0).sum()),'tied_scans':int((dif==0).sum()),'bootstrap':f'10000 patient cluster paired percentile95 seed{seed}'}


def pooled_paired(pairs,seed=20260918):
    """Average the per-case delta over seeds first, then one patient-cluster bootstrap (pre-registered pooled gate)."""
    ids=[(x['case_id'],x['subject_id']) for x in pairs[0][0]]
    for l,r in pairs:assert [(x['case_id'],x['subject_id']) for x in l]==ids==[(x['case_id'],x['subject_id']) for x in r]
    dif=np.mean([[b['locator_dsc']-a['locator_dsc'] for a,b in zip(l,r)] for l,r in pairs],axis=0)
    subjects=sorted({s for _,s in ids});n=np.array([sum(s==t for _,s in ids) for t in subjects])
    sums=np.array([sum(d for (_,s),d in zip(ids,dif) if s==t) for t in subjects]);ix=np.random.default_rng(seed).integers(0,len(subjects),(10000,len(subjects)))
    return {'seeds':len(pairs),'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),'scan_delta_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_delta_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'improved_scans':int((dif>0).sum()),'worse_scans':int((dif<0).sum()),'bootstrap':f'per-case delta averaged over seeds, then 10000 patient cluster percentile95 seed{seed}'}
