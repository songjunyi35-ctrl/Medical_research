#!/usr/bin/env python3
"""Frozen local MSCL Eq.5 implementation on real SAM-Med3D, not an exact author-code reproduction."""
from __future__ import annotations
import argparse, concurrent.futures, datetime, hashlib, json, os, random, shutil, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import Dataset, DataLoader
from sammed3d import engine
from scripts import train_sammed3d_t1ce_wt_upstream_literal_va as upstream
from scripts.build_t1ce_wt_official15_cache import preprocess_row
from scripts.sammed3d_full972_protocol import check_roles, FIT, DEV
RECORD=ROOT/'archive/research_records/2026-09-16_sammed3d_graymean_single972'
WORK=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/graymean_single972_20260916'
CKPT=ROOT/'artifacts/checkpoints/sam_med3d_turbo.pth'
SEED=20260916

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def disk(gib):
 free=shutil.disk_usage(ROOT).free
 if free<gib*(1<<30): raise RuntimeError(f'disk gate: free={free/(1<<30):.2f}GiB < {gib}')
 return free

def check_freeze():
 for receipt in ['PREREG.json','IMPLEMENTATION_FREEZE.json']:
  data=json.loads((RECORD/receipt).read_text())
  for name,digest in data['files'].items():
   if sha(ROOT/name)!=digest: raise RuntimeError(f'frozen hash mismatch: {name}')
 return check_roles()

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def gray_loss(logits,image,target):
 p=torch.sigmoid(logits.float());x=image.float();g=target.float()
 dims=tuple(range(1,g.ndim));count=g.sum(dims)
 if bool((count<=0).any()): raise ValueError('empty GT, no silent case dropping')
 mu_g=(x*g).sum(dims)/count
 mu_p=(x*p).sum(dims)/p.sum(dims).clamp_min(1e-6)
 return ((mu_p-mu_g)**2).mean()

class FirstPromptLoss:
 def __init__(self,image,weight):
  self.image=image;self.weight=weight;self.calls=0;self.gray=None
  self.seg=DiceCELoss(sigmoid=True,squared_pred=True,reduction='mean',lambda_dice=.5,lambda_ce=.5)
 def __call__(self,logits,target):
  loss=self.seg(logits,target)
  if self.calls==0:
   self.gray=gray_loss(logits,self.image,target)
   loss=loss+self.weight*self.gray
  self.calls+=1
  return loss

def build_model(device):
 model=engine.sam_model_registry3D[engine.MODEL_TYPE](checkpoint=None)
 payload=torch.load(CKPT,map_location='cpu',weights_only=False)
 state=payload.get('model_state_dict',payload.get('state_dict',payload))
 state={k.removeprefix('module.'):v for k,v in state.items()}
 model.load_state_dict(state,strict=True)
 for p in model.parameters(): p.requires_grad_(True)
 count=sum(p.numel() for p in model.parameters())
 if count!=100509232: raise RuntimeError(f'unexpected SAM parameter count: {count}')
 return model.to(device)

def optimizer_for(model):
 return torch.optim.AdamW([
  {'params':model.image_encoder.parameters(),'lr':8e-4},
  {'params':model.prompt_encoder.parameters(),'lr':8e-5},
  {'params':model.mask_decoder.parameters(),'lr':8e-5}],weight_decay=.1)

def train_step(model,opt,scaler,image,target,weight):
 opt.zero_grad(set_to_none=True)
 loss_fn=FirstPromptLoss(image,weight)
 with torch.amp.autocast('cuda',dtype=torch.float16):
  logits,loss,_=upstream.upstream_train_interaction(model=model,image=image,target=target,loss_fn=loss_fn)
 if loss_fn.calls!=11 or not torch.isfinite(loss): raise RuntimeError('invalid interaction loss')
 scaler.scale(loss).backward();scaler.unscale_(opt)
 grads=[p.grad for p in model.parameters() if p.grad is not None]
 if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
  raise RuntimeError('nonfinite or missing gradient')
 norm=float(torch.linalg.vector_norm(torch.stack([g.detach().float().norm() for g in grads])))
 if not np.isfinite(norm) or norm==0: raise RuntimeError('invalid gradient norm')
 scaler.step(opt);scaler.update()
 return {'loss':float(loss.detach()),'gray':float(loss_fn.gray.detach()),'grad_norm':norm}

def synthetic_smoke():
 check_freeze();disk(10);seed_all(SEED);torch.set_num_threads(2)
 report={'started_utc':now(),'mode':'synthetic_only','checkpoint_deserialized':True,'data_pixels_accessed':False,'validation_outer_confirmation_test_access':False,'devices':[]}
 write(RECORD/'SMOKE_PREACCESS.json',report)
 for index in range(2):
  device=torch.device(f'cuda:{index}');torch.cuda.set_device(device)
  seed_all(SEED);model=build_model(device);model.train();opt=optimizer_for(model)
  scaler=torch.amp.GradScaler('cuda',init_scale=128.)
  image=torch.randn((1,1,128,128,128),device=device)
  target=torch.zeros_like(image);target[:,:,40:88,40:88,40:88]=1
  image=image+target*1.5
  tracked=next(model.image_encoder.parameters());before=tracked.detach().clone()
  torch.cuda.reset_peak_memory_stats(device);t=time.monotonic()
  results=[train_step(model,opt,scaler,image,target,1.0) for _ in range(2)]
  torch.cuda.synchronize(device)
  changed=float((tracked.detach()-before).abs().max())
  if changed<=0: raise RuntimeError('image encoder did not update')
  report['devices'].append({'index':index,'name':torch.cuda.get_device_name(index),'weight':1.0,'steps':results,'encoder_max_weight_change':changed,'seconds':time.monotonic()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
  del model,opt,scaler,image,target,tracked,before
  torch.cuda.empty_cache();print(json.dumps(report['devices'][-1]),flush=True)
 report.update(status='passed',finished_utc=now());write(RECORD/'SMOKE.json',report)

def cache_one(pair):
 role,row=pair;torch.set_num_threads(1)
 import SimpleITK as sitk
 sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
 disk(10)
 image,label,qc=preprocess_row(row)
 if not np.isfinite(image).all() or np.sum(label)<=0: raise RuntimeError('invalid preprocessed case')
 out=WORK/'cache'/f"{row['case_id']}.npz";tmp=out.with_suffix('.tmp.npz')
 np.savez_compressed(tmp,image=image.astype(np.float32),label=label.astype(np.uint8));os.replace(tmp,out)
 return {'case_id':row['case_id'],'subject_id':row['subject_id'],'role':role,'path':str(out.relative_to(ROOT)),'sha256':sha(out),'bytes':out.stat().st_size,'image_sha256':sha(row['image_path']),'label_sha256':sha(row['label_path']),'qc':qc}

def build_cache():
 fit,dev=check_freeze();disk(10)
 if json.loads((RECORD/'SMOKE.json').read_text())['status']!='passed':raise RuntimeError('smoke gate')
 target=RECORD/'CACHE_INDEX.json'
 if target.exists():raise RuntimeError('cache already finalized')
 (WORK/'cache').mkdir(parents=True,exist_ok=True)
 old_index=json.loads((RECORD/'SEED_CACHE_INDEX.json').read_text())
 old={r['case_id']:r for r in old_index['cases']}
 write(RECORD/'CACHE_PREACCESS.json',{'started_utc':now(),'fit_pixels_access':True,'fixed_validation_pixels_access':True,'development_model_evaluation':False,'test_pixels_access':False,'previous_confirmation_subset_now_original_train':True,'prereg_sha256':sha(RECORD/'PREREG.json'),'implementation_sha256':sha(RECORD/'IMPLEMENTATION_FREEZE.json'),'disk_free_before':disk(10)})
 items=[];missing=[]
 for role,rows in [('fit',fit),('development',dev)]:
  for row in rows:
   if row['case_id'] not in old:missing.append((role,row));continue
   cached=dict(old[row['case_id']])
   if cached['image_sha256']!=sha(Path(row['image_path'])) or cached['label_sha256']!=sha(Path(row['label_path'])) or cached['sha256']!=sha(ROOT/cached['path']):raise RuntimeError('reused source/cache changed')
   if cached['subject_id']!=row['subject_id']:raise RuntimeError('subject mismatch')
   cached.update(role=role,reused_cache=True);items.append(cached)
 print(f'REUSED {len(items)}; BUILD {len(missing)}',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for record in pool.map(cache_one,missing):
   record['reused_cache']=False;items.append(record)
   with (RECORD/'CACHE_PROGRESS.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if sum(x['bytes'] for x in items if not x['reused_cache'])>8*(1<<30):raise RuntimeError('cache size gate')
   if len(items)%12==0:print(f'cache {len(items)}/1080',flush=True)
 lookup={x['case_id']:x for x in items};items=[lookup[r['case_id']] for r in fit+dev]
 if len(items)!=1080:raise RuntimeError('cache count')
 write(target,{'created_utc':now(),'cases':items,'count':len(items),'bytes':sum(x['bytes'] for x in items),'new_bytes':sum(x['bytes'] for x in items if not x['reused_cache']),'preaccess_sha256':sha(RECORD/'CACHE_PREACCESS.json')})
 print('CACHE COMPLETE',flush=True)

class Cases(Dataset):
 def __init__(self,items,augment=False,epoch=0):self.items=items;self.augment=augment;self.epoch=epoch
 def __len__(self):return len(self.items)
 def __getitem__(self,i):
  row=self.items[i]
  with np.load(ROOT/row['path']) as data:
   x=torch.from_numpy(data['image'].copy()).float();y=torch.from_numpy(data['label'].copy()).float()
  if x.ndim==3:x=x.unsqueeze(0);y=y.unsqueeze(0)
  if self.augment:
   seed=int.from_bytes(hashlib.sha256(f'{SEED}:{self.epoch}:{row["case_id"]}'.encode()).digest()[:4],'little')
   rng=np.random.default_rng(seed)
   for axis in range(1,4):
    if rng.random()<.5:x=x.flip(axis);y=y.flip(axis)
  return {'image':x,'label':y,'case_id':row['case_id'],'subject_id':row['subject_id']}

@torch.no_grad()
def evaluate(model,items,device):
 state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(device))
 model.eval();rows=[]
 try:
  loader=DataLoader(Cases(items),batch_size=1,num_workers=2,pin_memory=True)
  for batch in loader:
   seed_all(upstream.stable_case_seed(batch['case_id'][0]))
   with torch.amp.autocast('cuda',dtype=torch.float16):
    curve=upstream.upstream_validation_curve(model=model,image=batch['image'].to(device),target=batch['label'].to(device))
   rows.append({'case_id':batch['case_id'][0],'subject_id':batch['subject_id'][0],'curve':curve,'final5':curve[-1],'best5':max(curve)})
 finally:
  random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3],device);model.train()
 subjects={r['subject_id'] for r in rows}
 return {'rows':rows,'scan_final5':float(np.mean([r['final5'] for r in rows])),'subject_final5':float(np.mean([np.mean([r['final5'] for r in rows if r['subject_id']==s]) for s in sorted(subjects)])),'scan_best5':float(np.mean([r['best5'] for r in rows])),'scans':len(rows),'subjects':len(subjects)}

def atomic_torch_save(path,data):
 disk(6);tmp=path.with_suffix('.tmp.pth');torch.save(data,tmp);os.replace(tmp,path)

def train(arm, resume=False):
 if arm != "graymean":raise ValueError("single approved arm: graymean only")
 check_freeze();disk(8);torch.set_num_threads(2)
 index=json.loads((RECORD/'CACHE_INDEX.json').read_text())
 if index['count']!=1080:raise RuntimeError('cache count')
 for r in index['cases']:
  if sha(ROOT/r['path'])!=r['sha256']:raise RuntimeError('cache hash changed')
 items={role:[r for r in index['cases'] if r['role']==role] for role in ['fit','development']}
 if [len(items[k]) for k in ['fit','development']] != [972,108]:raise RuntimeError('role count')
 out=WORK/arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'PREACCESS.json').exists() and not resume:raise RuntimeError('existing run; explicit resume required')
 if resume and not (out/'latest.pth').exists():raise RuntimeError('no resume checkpoint')
 weight={'control':0.,'graymean':1.}[arm]
 if not resume:
  write(out/'PREACCESS.json',{'created_utc':now(),'arm':arm,'gray_weight':weight,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'prereg_sha256':sha(RECORD/'PREREG.json'),'implementation_sha256':sha(RECORD/'IMPLEMENTATION_FREEZE.json'),'cache_index_sha256':sha(RECORD/'CACHE_INDEX.json'),'initial_checkpoint_sha256':sha(CKPT),'fit':972,'development':108,'fixed_validation_access':True,'outer_confirmation_test_evaluation':False})
 device=torch.device('cuda:0');seed_all(SEED);model=build_model(device);model.train()
 opt=optimizer_for(model);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[120,180],gamma=.1)
 scaler=torch.amp.GradScaler('cuda',init_scale=128.)
 best=-1.;best_epoch=0;best_report=None;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 first_epoch=1
 if resume:
  prior=json.loads((out/'STATUS.json').read_text())
  checkpoint=out/'latest.pth';digest=sha(checkpoint)
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if saved['arm']!=arm or saved['prereg_sha256']!=sha(RECORD/'PREREG.json') or saved['epoch']!=prior['last_complete_epoch']:raise RuntimeError('resume identity/epoch mismatch')
  if saved['epoch']>=200:raise RuntimeError('already completed')
  model.load_state_dict(saved['model_state_dict'],strict=True);opt.load_state_dict(saved['optimizer_state_dict']);scheduler.load_state_dict(saved['scheduler_state_dict']);scaler.load_state_dict(saved['scaler_state_dict'])
  random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state(saved['rng_cuda'],device)
  first_epoch=saved['epoch']+1;best=saved['best_score'];best_epoch=saved['best_epoch'];best_report=f'development_e{best_epoch:03d}.json' if best_epoch else None
  started-=prior['elapsed_seconds']
  write(out/f'RESUME_E{first_epoch:03d}.json',{'created_utc':now(),'checkpoint_sha256':digest,'from_epoch':saved['epoch'],'start_epoch':first_epoch,'prereg_sha256':sha(RECORD/'PREREG.json')})
  del saved
 for epoch in range(first_epoch,201):
  disk(6);tick=time.monotonic();dataset=Cases(items['fit'],True,epoch)
  generator=torch.Generator().manual_seed(SEED+epoch)
  loader=DataLoader(dataset,batch_size=1,shuffle=True,generator=generator,num_workers=2,pin_memory=True)
  totals=[]
  for step,batch in enumerate(loader,1):
   if step%32==1:disk(6)
   value=train_step(model,opt,scaler,batch['image'].to(device),batch['label'].to(device),weight);totals.append(value)
   if step%50==0: print(json.dumps({'utc':now(),'arm':arm,'epoch':epoch,'step':step,**value}),flush=True)
  scheduler.step();budget=time.monotonic()-started>=48*3600
  log={'utc':now(),'arm':arm,'epoch':epoch,'steps':len(totals),'train':{k:float(np.mean([r[k] for r in totals])) for k in totals[0]},'training_seconds':time.monotonic()-tick,'elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
  if epoch%5==0 or budget or epoch==200:
   marker=out/'DEVELOPMENT_PREACCESS.json'
   if not marker.exists():write(marker,{'created_utc':now(),'epoch':epoch,'role':'reused fixed validation108/50','prereg_sha256':sha(RECORD/'PREREG.json'),'test_outer_confirmation_access':False})
   result=evaluate(model,items['development'],device);result.update(epoch=epoch,arm=arm,created_utc=now())
   write(out/f'development_e{epoch:03d}.json',result)
   log['development']={k:v for k,v in result.items() if k!='rows'}
   if result['scan_final5']>best:
    best=result['scan_final5'];best_epoch=epoch;best_report=f'development_e{epoch:03d}.json'
    atomic_torch_save(out/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'arm':arm,'gray_weight':weight,'selection':'development final5 scan Dice'})
    write(out/'BEST.json',{'epoch':epoch,'scan_final5':best,'report':best_report,'report_sha256':sha(out/best_report),'checkpoint_sha256':sha(out/'best.pth')})
  atomic_torch_save(out/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),'arm':arm,'prereg_sha256':sha(RECORD/'PREREG.json')})
  with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
  write(out/'STATUS.json',{'status':'running','last_complete_epoch':epoch,'best_epoch':best_epoch,'updated_utc':now(),**{k:v for k,v in log.items() if k not in ['train','development']}})
  print(json.dumps(log),flush=True)
  if budget:break
 write(out/'STATUS.json',{'status':'completed','reason':'wall_budget' if budget else 'epoch_budget','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_score':best,'best_report':best_report,'updated_utc':now(),'elapsed_seconds':time.monotonic()-started,'checkpoint_sha256':sha(out/'best.pth') if best_epoch else None,'test_outer_confirmation_access':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','cache','train']);p.add_argument('--arm',choices=['graymean']);p.add_argument('--resume',action='store_true');a=p.parse_args()
 try:
  if a.mode=='smoke':synthetic_smoke()
  elif a.mode=='cache':build_cache()
  else:
   if not a.arm:p.error('--arm is required')
   train(a.arm,resume=a.resume)
 except Exception as e:
  failure=RECORD/f'FAILURE_{a.mode}_{a.arm or "shared"}_{datetime.datetime.now().strftime("%H%M%S")}.json'
  write(failure,{'created_utc':now(),'mode':a.mode,'arm':a.arm,'error':repr(e),'traceback':traceback.format_exc(),'access_evidence':'SMOKE_PREACCESS.json / CACHE_PREACCESS.json / per-arm PREACCESS.json and DEVELOPMENT_PREACCESS.json; presence denotes attempted access, not completion'})
  raise
