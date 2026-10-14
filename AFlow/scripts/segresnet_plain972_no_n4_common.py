"""Fixed original split and audited pure SegResNet baseline utilities."""
import os,json,hashlib,random,time,shutil,math
from pathlib import Path
import numpy as np
import torch
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset,augment_batch,dice_per_case,validate
from scripts.train_t1ce_wt_plain_long_schedule import learning_rate_for_epoch
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-18_segresnet_plain972_no_n4'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_no_n4_20260918'
TRAIN_CACHE=Path('/tmp/aflow_plain972_no_n4_20260918/train')
VAL_CACHE=Path('/tmp/aflow_plain972_no_n4_20260918/validation')
SEED=20260819

def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def disk():
    free=shutil.disk_usage(ROOT).free
    if free<512*(1<<20):raise RuntimeError('persistent disk below512MiB reserve')
    return free

def save(path,data):
    disk();tmp=path.with_suffix('.tmp.pth');torch.save(data,tmp);os.replace(tmp,path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def batches_for_epoch(epoch,rank):
    order=torch.randperm(972,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
    for start in range(0,972,8):
        whole=order[start:start+8];n=len(whole)//2;yield whole[rank*n:(rank+1)*n]

def model_for(device):
    model=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device)
    assert sum(p.numel() for p in model.parameters())==4_700_897
    return model

def safe_augment(image,label):
    if bool((label.flatten(1).sum(1)<=0).any()):raise RuntimeError('empty original training target')
    old_image=image.clone();old_label=label.clone()
    image,label=augment_batch(image,label,max_shift=8)
    missing=label.flatten(1).sum(1)<=0
    if bool(missing.any()):image[missing]=old_image[missing];label[missing]=old_label[missing]
    return image,label,int(missing.sum())

def verify():
    d=json.loads((R/'FREEZE.json').read_text())
    for name,digest in d['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen file changed: '+name)
    return d

def summary_ci(cases):
    subjects=sorted({r['subject_id'] for r in cases});n=np.array([sum(r['subject_id']==s for r in cases) for s in subjects])
    sums=np.array([sum(r['locator_dsc'] for r in cases if r['subject_id']==s) for s in subjects])
    ix=np.random.default_rng(20260917).integers(0,len(subjects),(10000,len(subjects)))
    return {'scan_dice':float(sums.sum()/n.sum()),'subject_dice':float((sums/n).mean()),'scan_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'point_hit':float(np.mean([r['automatic_point_hits_gt'] for r in cases])),'scans':len(cases),'subjects':len(subjects),'bootstrap':'10000 patient cluster percentile95 seed20260917'}
