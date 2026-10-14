"""New combination: public pretrained initialization, native1mm downstream training.
All training geometry and main hyperparameters match the prior native arm.
All new validation/reference scoring uses the same deterministic numeric mode.
"""
import os,json,hashlib,random,shutil,datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import augment_batch,dice_per_case,subject_id
from scripts.train_t1ce_wt_plain_long_schedule import learning_rate_for_epoch

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-21_segresnet_pretrain_native_res'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/pretrained_native_res_20260921'
CACHE_ROOT=Path('/tmp/aflow_pretrained_native_res_20260921')
TRAIN_CACHE=CACHE_ROOT/'train'
VAL_CACHE=CACHE_ROOT/'validation'
CANONICAL={'train':ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train',
           'validation':ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108'}
MANIFEST={'train':ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl',
          'validation':ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl',
          'test':ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'}
BASELINE_CKPT=ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270/checkpoint_epoch193.pth'
BASELINE_CKPT_SHA='157fc5af363fc42b2c993df07639693ed8d9f193b8cadd11810a39f62c0f54a5'
BASELINE_VAL_REPORT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917/validation_e193.json'
CANVAS=(160,208,160)
SCHEMA='aflow.segresnet.native_res_canvas_cache'
SEED=20260819
EPOCHS=200
BATCH_CANDIDATES=(4,)      # per-GPU; the synthetic smoke measures real peak memory and picks the largest that fits
MEMORY_CEILING_BYTES=36*(1<<30)
TRAIN_SCANS=972
WALL_BUDGET_SECONDS=14*3600
PARAMETERS=4_700_897


def steps_per_epoch(per_gpu):return -(-TRAIN_SCANS//(2*per_gpu))
def batch_decision():
    d=json.loads((R/'BATCH_DECISION.json').read_text());per_gpu=int(d['per_gpu_batch'])
    if per_gpu not in BATCH_CANDIDATES:raise RuntimeError('batch decision outside frozen candidates')
    return per_gpu


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
    free=shutil.disk_usage(ROOT).free
    if free<2*(1<<30):raise RuntimeError('persistent disk below2GiB reserve')
    return free
def save(path,data):
    disk();tmp=path.with_suffix('.tmp.pth');torch.save(data,tmp);os.replace(tmp,path)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def fast_cudnn():
    # Throughput over bitwise determinism: cudnn autotuning and TF32 for the fp32 parts.
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True


def batches_for_epoch(epoch,rank,per_gpu):
    """Original972 once per epoch (no replacement, tail kept), global batch = 2*per_gpu split evenly over the two ranks.

    The tail batch is split so both ranks always receive the same count; 972 is even for every candidate.
    """
    order=torch.randperm(TRAIN_SCANS,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
    for start in range(0,TRAIN_SCANS,2*per_gpu):
        whole=order[start:start+2*per_gpu]
        if len(whole)%2:raise RuntimeError('odd tail cannot be split evenly across ranks')
        n=len(whole)//2;yield whole[rank*n:(rank+1)*n]


def model_for(device):
    model=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device)
    assert sum(p.numel() for p in model.parameters())==PARAMETERS
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


def canvas_offset(extent):
    """Center a native crop of `extent` inside CANVAS; raise if it does not fit."""
    if any(int(e)>c for e,c in zip(extent,CANVAS)) or any(int(e)<=0 for e in extent):raise ValueError(f'extent {list(extent)} does not fit canvas {list(CANVAS)}')
    return [int((c-int(e))//2) for e,c in zip(extent,CANVAS)]


def canvas_slices(geometry):
    off=geometry['canvas_offset'];ext=geometry['extent']
    return tuple(slice(int(o),int(o)+int(e)) for o,e in zip(off,ext))


def pad_to_canvas(volume,fill=0):
    volume=np.asarray(volume);off=canvas_offset(volume.shape)
    out=np.full(CANVAS,fill,dtype=volume.dtype);out[tuple(slice(o,o+s) for o,s in zip(off,volume.shape))]=volume
    return out,off


def case_metrics(logits,label,geometry):
    """Dice and label-free argmax point on the native crop grid only (canvas padding excluded).

    logits/label: tensors [1,D,H,W] on the canvas.  Returns crop-grid point coordinates.
    """
    s=canvas_slices(geometry);crop_logits=logits[(slice(None),)+s].float();crop_label=label[(slice(None),)+s].float()
    dsc=float(dice_per_case(crop_logits.unsqueeze(0),crop_label.unsqueeze(0))[0])
    flat=int(torch.argmax(crop_logits[0].flatten()));h,w=crop_logits.shape[-2:]
    z=flat//(h*w);rem=flat%(h*w);y=rem//w;x=rem%w
    return {'locator_dsc':dsc,'automatic_point':[int(z),int(y),int(x)],'automatic_point_hits_gt':bool(crop_label[0,z,y,x]>0.5),'crop_extent':[int(v) for v in geometry['extent']]}


class NativeCanvasDataset(Dataset):
    """Fixed-canvas float32 image / uint8 WT label memmaps, one slot per manifest row."""
    def __init__(self,cache_dir,*,expected_split,case_ids=None):
        self.cache_dir=Path(cache_dir);self.index=json.loads((self.cache_dir/'index.json').read_text())
        if self.index.get('schema')!=SCHEMA:raise ValueError('unexpected native cache schema')
        if self.index.get('test_access') is not False:raise ValueError('refusing a test-accessible cache')
        if self.index.get('split')!=expected_split:raise ValueError(f"cache split {self.index.get('split')!r} != {expected_split!r}")
        if tuple(self.index['canvas'])!=CANVAS:raise ValueError('cache canvas differs from frozen CANVAS')
        if self.index['preprocessing'].get('uses_gt_for_localization') is not False:raise ValueError('cache must not use GT for localization')
        self.all_ids=[str(x) for x in self.index['case_ids']];self.total=int(self.index['count'])
        if len(self.all_ids)!=self.total:raise ValueError('count/case_ids mismatch')
        wanted=self.all_ids if case_ids is None else [str(x) for x in case_ids]
        if sorted(wanted)!=sorted(self.all_ids):raise ValueError('requested case ids differ from cache')
        self.case_ids=wanted;self._slot={c:i for i,c in enumerate(self.all_ids)};self.geometry=self.index['geometry']
        self._images=None;self._labels=None
    def _open(self):
        if self._images is None:
            self._images=np.memmap(self.cache_dir/'images.f32',dtype=np.float32,mode='r',shape=(self.total,*CANVAS))
            self._labels=np.memmap(self.cache_dir/'labels.u8',dtype=np.uint8,mode='r',shape=(self.total,*CANVAS))
    def __len__(self):return len(self.case_ids)
    def __getitem__(self,i):
        self._open();cid=self.case_ids[i];slot=self._slot[cid]
        image=torch.from_numpy(np.array(self._images[slot],dtype=np.float32))[None]
        label=torch.from_numpy(np.array(self._labels[slot],dtype=np.float32))[None]
        return {'image':image,'label':label,'case_id':cid,'subject_id':subject_id(cid),'index':i}


def summary_ci(cases):
    subjects=sorted({r['subject_id'] for r in cases});n=np.array([sum(r['subject_id']==s for r in cases) for s in subjects])
    sums=np.array([sum(r['locator_dsc'] for r in cases if r['subject_id']==s) for s in subjects])
    ix=np.random.default_rng(20260917).integers(0,len(subjects),(10000,len(subjects)))
    return {'scan_dice':float(sums.sum()/n.sum()),'subject_dice':float((sums/n).mean()),'scan_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'point_hit':float(np.mean([r['automatic_point_hits_gt'] for r in cases])),'scans':len(cases),'subjects':len(subjects),'bootstrap':'10000 patient cluster percentile95 seed20260917'}


def paired(left,right):
    assert [(x['case_id'],x['subject_id']) for x in left]==[(x['case_id'],x['subject_id']) for x in right]
    subjects=sorted({r['subject_id'] for r in left});n=np.array([sum(r['subject_id']==s for r in left) for s in subjects])
    dif=np.array([b['locator_dsc']-a['locator_dsc'] for a,b in zip(left,right)])
    sums=np.array([sum(d for r,d in zip(left,dif) if r['subject_id']==s) for s in subjects]);ix=np.random.default_rng(20260918).integers(0,len(subjects),(10000,len(subjects)))
    return {'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),'scan_delta_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_delta_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'point_hits_delta':int(sum(r['automatic_point_hits_gt'] for r in right)-sum(r['automatic_point_hits_gt'] for r in left)),'improved_scans':int((dif>0).sum()),'worse_scans':int((dif<0).sum()),'tied_scans':int((dif==0).sum()),'bootstrap':'10000 patient cluster paired percentile95 seed20260918'}


def summarize(cases):
    import statistics
    from collections import defaultdict
    by=defaultdict(list)
    for c in cases:by[c['subject_id']].append(float(c['locator_dsc']))
    values=[float(c['locator_dsc']) for c in cases]
    return {'cases':len(cases),'subjects':len(by),'locator_dsc_mean':statistics.fmean(values),'locator_dsc_median':statistics.median(values),'locator_dsc_std_population':statistics.pstdev(values),'locator_subject_dsc_mean':statistics.fmean(statistics.fmean(v) for v in by.values()),'automatic_point_hit_rate':statistics.fmean(float(c['automatic_point_hits_gt']) for c in cases),'per_case':cases}


PRETRAIN_CKPT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/pretrain/final.pth'
PRETRAIN_SHA='3726eac66918b7afd32ae93500a554534d27528bc86a0fe9dd2a626ce5ac6b76'
REFERENCES={
 'pure193':(BASELINE_CKPT,BASELINE_CKPT_SHA,BASELINE_VAL_REPORT,'128'),
 'pretrained179':(ROOT/'archive/research_records/2026-09-21_segresnet_external_pretrain/checkpoint_selected.pth','7006d44a2a2f765562dc0155f914c967a27b22ffba476cea5e966f01d69f7ad0',ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/finetune_pretrained/validation_e179.json','128'),
 'native173':(ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270/checkpoint_epoch173.pth','57d6159802c9ac46e374c48d6f6ed42cd656422fa3b802c70cc5883351a33af4',ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_native_res_20260918/validation_e173.json','native')}

def validation_mode():
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True

def deterministic_validation(fn):
    import functools
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        old=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            validation_mode();return fn(*args,**kwargs)
        finally:
            torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old
    return wrapped
