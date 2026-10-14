"""Bounded, explicitly test-tuned 2026-09-21 campaign. No historical writes."""
import os, sys, json, time, math, hashlib, random, shutil, datetime, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921'
SCRATCH=Path('/tmp/aflow_test270_10h_20260921')
NATIVE=Path('/tmp/aflow_pretrained_native_res_20260921')
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
P=json.loads((R/'PROTOCOL.json').read_text())
SEED=P['seed']
CANON={
 'train':ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train',
 'validation':ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108',
 'test':ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'}
MANIFEST={k:ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'/v for k,v in [('train','train_t1ce_wt.jsonl'),('validation','validation_t1ce_wt.jsonl'),('test','sealed_test/test_cases_t1ce.jsonl')]}
MANIFEST_SHA={'train':'6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65','validation':'f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb','test':'a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9'}
SOURCES={
 'pure193':('2026-09-17_segresnet_plain972_e193_test270/checkpoint_epoch193.pth','157fc5af363fc42b2c993df07639693ed8d9f193b8cadd11810a39f62c0f54a5','128'),
 'native173':('2026-09-20_segresnet_native_res_e173_test270/checkpoint_epoch173.pth','57d6159802c9ac46e374c48d6f6ed42cd656422fa3b802c70cc5883351a33af4','native'),
 'pretrained179':('2026-09-21_segresnet_external_pretrain/checkpoint_selected.pth','7006d44a2a2f765562dc0155f914c967a27b22ffba476cea5e966f01d69f7ad0','128'),
 'combination166':('2026-09-21_segresnet_pretrain_native_res/checkpoint_selected.pth','b2a37a7ed4ea73922a454f5f60995d3a8807f18d122539ba0c85d9b5e62c8a18','native'),
 'patient172':('2026-09-21_segresnet_patient_equal_loss/checkpoint_selected.pth','46ad9f1ccbb85a2c8f7790a2d225265ddeb50c667f8091e08bf8006e4039083d','128')}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def timestamp(s):return datetime.datetime.fromisoformat(s).timestamp()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def event(path,**obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps({'utc':now(),**obj},ensure_ascii=False,allow_nan=False)+'\n')
def rows(split):return [json.loads(s) for s in MANIFEST[split].read_text().splitlines() if s.strip()]
def used(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) if Path(path).exists() else 0
def disk(extra=0):
 persistent=used(R)+used(OUT)
 scratch=used(SCRATCH)
 if shutil.disk_usage(ROOT).free-extra<20*2**30 or persistent+extra>8*2**30:raise RuntimeError('persistent disk safety gate')
 if shutil.disk_usage('/tmp').free-extra<100*2**30 or scratch+extra>100*2**30:raise RuntimeError('scratch disk safety gate')
 return {'persistent_bytes':persistent,'scratch_bytes':scratch,'persistent_free':shutil.disk_usage(ROOT).free,'scratch_free':shutil.disk_usage('/tmp').free}
def save(p,obj):
 disk(256<<20);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def source(name):return ROOT/'archive/research_records'/SOURCES[name][0]
def verify():
 f=read(R/'FREEZE.json')
 for p,h in f['files'].items():
  if sha(ROOT/p)!=h:raise RuntimeError('frozen source changed: '+p)
 for s,h in MANIFEST_SHA.items():assert sha(MANIFEST[s])==h
 return f
def seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def model_for(device,path=None):
 from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
 m=AutomaticPromptLocator(center_aware=False,init_filters=16,dropout=.1).to(device)
 if path:
  state=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(state['model_state_dict'],strict=True)
 return m
def numeric():
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
def batches(epoch,rank=0,world=1):
 order=torch.randperm(972,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
 return [whole[rank*len(whole)//world:(rank+1)*len(whole)//world] for whole in [order[i:i+8] for i in range(0,972,8)]]
def starts(shape):
 if len(shape)!=3 or any(d<128 or d>256 for d in shape):raise ValueError('unsupported eight-patch canvas')
 return list(itertools.product(*[(0,d-128) for d in shape]))
def gaussian(device):
 v=torch.arange(128,device=device,dtype=torch.float32)-63.5
 g=torch.exp(-.5*(v/16)**2)
 return (g[:,None,None]*g[None,:,None]*g[None,None,:]).clamp_min(1e-7)[None,None]
def predict(model,image,mode):
 # Image-only API: no label, case ID, or ground-truth localization accepted.
 if mode!='patch':return model(image).segmentation_logits.float()
 shape=image.shape[-3:];out=torch.zeros((image.shape[0],1,*shape),device=image.device);den=torch.zeros_like(out);g=gaussian(image.device)
 for start in starts(shape):
  sl=tuple(slice(s,s+128) for s in start);z=model(image[(slice(None),slice(None),*sl)]).segmentation_logits.float()
  out[(slice(None),slice(None),*sl)]+=z*g;den[(slice(None),slice(None),*sl)]+=g
 return out/den
def crop_slices(geo):return tuple(slice(o,o+n) for o,n in zip(geo['canvas_offset'],geo['extent']))
def case_metric(logit,label,cid,threshold=.5):
 logit=logit.float();label=label.bool();pred=logit>math.log(threshold/(1-threshold))
 tp=int((pred&label).sum());fp=int((pred&~label).sum());fn=int((~pred&label).sum())
 denom=2*tp+fp+fn;point=np.unravel_index(int(logit.flatten().argmax()),logit.shape[-3:])
 return {'case_id':cid,'subject_id':cid.rsplit('-',1)[0],'locator_dsc':2*tp/denom if denom else 1.,'automatic_point':[int(x) for x in point],'automatic_point_hits_gt':bool(label.reshape(logit.shape[-3:])[point]),'tp':tp,'fp':fp,'fn':fn,'fnr':fn/(tp+fn) if tp+fn else 0.,'fdr':fp/(tp+fp) if tp+fp else 0.}
def summary(cases):
 from scripts.segresnet_pretrained_native_common import summary_ci
 result=summary_ci(cases);result['point_hits']=sum(r['automatic_point_hits_gt'] for r in cases);return result
def compare(reference,candidate):
 from scripts.segresnet_pretrained_native_common import paired
 return paired(reference,candidate)

class GridDataset(Dataset):
 def __init__(self,split,native=False,patch=False,epoch=0):
  if patch and split!='train':raise ValueError('label-guided patches only for training')
  self.split=split;self.native=native;self.patch=patch;self.epoch=epoch
  self.path=(SCRATCH/'test_native' if split=='test' else NATIVE/split) if native else CANON[split]
  self.index=read(self.path/'index.json');self.ids=self.index['case_ids'];self.shape=tuple(self.index['canvas'] if native else (128,128,128));self.n=len(self.ids)
  assert set(self.ids)=={r['case_id'] for r in rows(split)}
  self.images=None;self.labels=None
 def __len__(self):return self.n
 def __getitem__(self,i):
  if self.images is None:
   self.images=np.memmap(self.path/'images.f32',mode='r',dtype='float32',shape=(self.n,*self.shape))
   self.labels=np.memmap(self.path/'labels.u8',mode='r',dtype='uint8',shape=(self.n,*self.shape))
  x=np.array(self.images[i]);y=np.array(self.labels[i]);cid=self.ids[i]
  if self.patch:
   rng=np.random.default_rng(SEED+self.epoch*10000+i)
   coords=np.argwhere(y>0) if rng.random()<.5 else np.argwhere(x!=0)
   if not len(coords):raise ValueError('empty training sampling support')
   center=coords[rng.integers(len(coords))];start=np.clip(center-64,0,np.array(self.shape)-128)
   sl=tuple(slice(s,s+128) for s in start);x=x[sl].copy();y=y[sl].copy()
  return {'image':torch.from_numpy(x)[None],'label':torch.from_numpy(y.astype(np.float32))[None],'case_id':cid,'slot':i}

def native_targets(split):
 p=OUT/'label_relay'/split;idx=read(p/'index.json')
 return idx,np.load(p/'labels.npz',allow_pickle=False)

@torch.no_grad()
def evaluate(model,mode,split,rank=0,world=1,threshold=.5,logit_dir=None):
 numeric();model.eval();device=next(model.parameters()).device
 ds=GridDataset(split,native=(mode!='128'));idx,targets=native_targets(split)
 result=[];tick=time.monotonic()
 for start in range(rank*2,len(ds),world*2):
  items=[ds[i] for i in range(start,min(start+2,len(ds)))];x=torch.stack([it['image'] for it in items]).to(device)
  with torch.autocast('cuda',dtype=torch.bfloat16):z=predict(model,x,mode)
  for k,it in enumerate(items):
   cid=it['case_id'];geo=idx['geometry'][cid];target=torch.from_numpy(targets[cid].astype(np.float32)).to(device)
   if mode=='128':crop=F.interpolate(z[k:k+1],size=geo['extent'],mode='trilinear',align_corners=False)[0,0]
   else:crop=z[k,0][crop_slices(geo)]
   r=case_metric(crop,target,cid,threshold);result.append(r)
   if logit_dir:
    p=Path(logit_dir);p.mkdir(parents=True,exist_ok=True);np.save(p/(cid+'.npy'),crop.cpu().numpy().astype('float32'))
  if start%40==0:print(json.dumps({'utc':now(),'event':'evaluate','split':split,'mode':mode,'progress':start,'rank':rank}),flush=True)
 targets.close();torch.cuda.synchronize();return result,time.monotonic()-tick

def stop_rule(history):
 epoch=history[-1]['epoch'];s0=history[0]['score'];scores=[h['score'] for h in history];best=max(scores)
 if epoch==10 and len(history)>=3 and all(s<s0-.01 for s in scores[-3:]) and scores[-1]<=scores[-2]:return 'epoch10 deterioration without recovery'
 if epoch==25 and best-s0<.003 and best-max(h['score'] for h in history if h['epoch']<=15)<.001:return 'epoch25 gain and progress below budget gate'
 if epoch==50 and best-s0<.005:return 'epoch50 gain below0.005'
 return None

def criterion(logit,target,weights,kind):
 p=logit.float().sigmoid();target=target.float();axes=(1,2,3,4);tp=(p*target).sum(axes)
 if kind=='C':
  fp=(p*(1-target)).sum(axes);fn=((1-p)*target).sum(axes);loss=1-(tp+1e-5)/(tp+.3*fp+.7*fn+1e-5)
 else:loss=1-(2*tp+1e-5)/(p.sum(axes)+target.sum(axes)+1e-5)
 ce=F.binary_cross_entropy_with_logits(logit.float(),target,reduction='none').mean(axes)
 return ((loss+.1*ce)*weights).mean()
