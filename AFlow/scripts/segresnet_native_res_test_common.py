"""Frozen contract for the one-shot test270 evaluation of the native-resolution SegResNet epoch173.

Test crops are up to 160x200x161 (from the existing canonical test-cache geometry metadata), so the
inference canvas is 160x208x168: the training canvas extended by 8 on the last axis.  SegResNet is
fully convolutional with three /2 stages, so any canvas divisible by 8 is a valid input; padding is
zero, exactly as in training.  Metrics are computed on the native crop only (padding excluded).
"""
import os,json,hashlib,datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.segresnet_native_res_common import model_for,fast_cudnn,seed_all,summary_ci,paired,summarize,dice_per_case,subject_id,BASELINE_CKPT,BASELINE_CKPT_SHA,MANIFEST,SCHEMA

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-20_segresnet_native_res_e173_test270'
TRAIN_RECORD=ROOT/'archive/research_records/2026-09-18_segresnet_plain972_native_res'
CANDIDATE_CKPT=R/'checkpoint_epoch173.pth'
CANONICAL_TEST=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'
BASELINE_TEST_RAW=ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RAW_TEST270.json'
TEST_CACHE=Path('/tmp/aflow_native_res_test270_20260920/test')
TEST_CANVAS=(160,208,168)
TEST_SCANS=270;TEST_SUBJECTS=119
IMAGE_TOLERANCE=1e-3


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


def verify():
    d=json.loads((R/'FREEZE.json').read_text())
    for name,digest in d['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen file changed: '+name)
    return d


def canvas_offset(extent,canvas=TEST_CANVAS):
    if any(int(e)>c for e,c in zip(extent,canvas)) or any(int(e)<=0 for e in extent):raise ValueError(f'extent {list(extent)} does not fit canvas {list(canvas)}')
    return [int((c-int(e))//2) for e,c in zip(extent,canvas)]


def canvas_slices(geometry):
    return tuple(slice(int(o),int(o)+int(e)) for o,e in zip(geometry['canvas_offset'],geometry['extent']))


def pad_to_canvas(volume,fill=0,canvas=TEST_CANVAS):
    volume=np.asarray(volume);off=canvas_offset(volume.shape,canvas)
    out=np.full(canvas,fill,dtype=volume.dtype);out[tuple(slice(o,o+s) for o,s in zip(off,volume.shape))]=volume
    return out,off


def case_metrics(logits,label,geometry):
    """Dice and label-free argmax point on the native crop grid only; logits/label are [1,D,H,W] on the canvas."""
    s=canvas_slices(geometry);crop_logits=logits[(slice(None),)+s].float();crop_label=label[(slice(None),)+s].float()
    dsc=float(dice_per_case(crop_logits.unsqueeze(0),crop_label.unsqueeze(0))[0])
    flat=int(torch.argmax(crop_logits[0].flatten()));h,w=crop_logits.shape[-2:]
    z=flat//(h*w);rem=flat%(h*w);y=rem//w;x=rem%w
    return {'locator_dsc':dsc,'automatic_point':[int(z),int(y),int(x)],'automatic_point_hits_gt':bool(crop_label[0,z,y,x]>0.5),'crop_extent':[int(v) for v in geometry['extent']]}


class NativeTestDataset(Dataset):
    def __init__(self,cache_dir,expected_case_ids):
        self.cache_dir=Path(cache_dir);self.index=json.loads((self.cache_dir/'index.json').read_text())
        if self.index.get('schema')!=SCHEMA+'.test':raise ValueError('unexpected native test cache schema')
        if self.index.get('split')!='test' or self.index.get('test_access') is not True:raise ValueError('test evaluator requires a declared test cache')
        if tuple(self.index['canvas'])!=TEST_CANVAS:raise ValueError('cache canvas differs from frozen TEST_CANVAS')
        if self.index['preprocessing'].get('uses_gt_for_localization') is not False:raise ValueError('cache must not use GT for localization')
        self.case_ids=[str(x) for x in self.index['case_ids']]
        if self.case_ids!=list(expected_case_ids) or len(self.case_ids)!=TEST_SCANS!=self.index['count']:raise ValueError('test cache case order/count differs from frozen manifest')
        self.geometry=self.index['geometry'];self._images=None;self._labels=None
    def _open(self):
        if self._images is None:
            self._images=np.memmap(self.cache_dir/'images.f32',dtype=np.float32,mode='r',shape=(TEST_SCANS,*TEST_CANVAS))
            self._labels=np.memmap(self.cache_dir/'labels.u8',dtype=np.uint8,mode='r',shape=(TEST_SCANS,*TEST_CANVAS))
    def __len__(self):return TEST_SCANS
    def __getitem__(self,i):
        self._open();cid=self.case_ids[i]
        return {'image':torch.from_numpy(np.array(self._images[i],dtype=np.float32))[None],'label':torch.from_numpy(np.array(self._labels[i],dtype=np.float32))[None],'case_id':cid,'subject_id':subject_id(cid)}
