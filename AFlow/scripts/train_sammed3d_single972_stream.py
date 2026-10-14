#!/usr/bin/env python3
"""Single972 scientific protocol unchanged; lazy deterministic cache materialization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import *
import fcntl
BASE_CASES=Cases

def materialize(row):
 out=ROOT/row['path'];receipt=WORK/'cache'/f"{row['case_id']}.source.json"
 if out.exists():return
 with (WORK/'cache'/f"{row['case_id']}.lock").open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if out.exists():return
  torch.set_num_threads(1)
  import SimpleITK as sitk
  sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
  disk(6)
  source=row['source'];image,label,qc=preprocess_row(source)
  if not np.isfinite(image).all() or np.sum(label)<=0:raise RuntimeError('invalid lazy cache')
  tmp=out.with_name(out.stem+f'.{os.getpid()}.tmp.npz')
  np.savez_compressed(tmp,image=image.astype(np.float32),label=label.astype(np.uint8));os.replace(tmp,out)
  write(receipt,{'case_id':row['case_id'],'subject_id':row['subject_id'],'role':row['role'],'path':row['path'],'sha256':sha(out),'bytes':out.stat().st_size,'image_sha256':sha(Path(source['image_path'])),'label_sha256':sha(Path(source['label_path'])),'qc':qc,'reused_cache':False})

class Cases(BASE_CASES):
 def __getitem__(self,i):
  materialize(self.items[i])
  return super().__getitem__(i)

def ensure_complete_index():
 # Frozen evaluator waits for source integrity audit before formal validation.
 started=time.monotonic()
 while not (RECORD/'SOURCE_AUDIT_COMPLETE.json').exists():
  if (RECORD/'STREAM_FAILURE.json').exists():raise RuntimeError('source audit/cache failure')
  if time.monotonic()-started>3600:raise RuntimeError('source audit incomplete before validation')
  time.sleep(5)
 if not (RECORD/'CACHE_INDEX.json').exists():raise RuntimeError('full cache index missing')

def train(arm, resume=False):
 if arm != "graymean":raise ValueError("single approved arm: graymean only")
 check_freeze();disk(8);torch.set_num_threads(2)
 index=json.loads((RECORD/'INPUT_PLAN.json').read_text())
 if index['count']!=1080:raise RuntimeError('input plan count')
 for r in index['cases']:
  if r.get('sha256') and sha(ROOT/r['path'])!=r['sha256']:raise RuntimeError('cache content hash changed')
 items={role:[r for r in index['cases'] if r['role']==role] for role in ['fit','development']}
 if [len(items[k]) for k in ['fit','development']] != [972,108]:raise RuntimeError('role count')
 out=WORK/arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'PREACCESS.json').exists() and not resume:raise RuntimeError('existing run; explicit resume required')
 if resume and not (out/'latest.pth').exists():raise RuntimeError('no resume checkpoint')
 weight={'control':0.,'graymean':1.}[arm]
 if not resume:
  write(out/'PREACCESS.json',{'created_utc':now(),'arm':arm,'gray_weight':weight,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'prereg_sha256':sha(RECORD/'PREREG.json'),'implementation_sha256':sha(RECORD/'IMPLEMENTATION_FREEZE.json'),'input_plan_sha256':sha(RECORD/'INPUT_PLAN.json'),'cache_index_finalized':(RECORD/'CACHE_INDEX.json').exists(),'initial_checkpoint_sha256':sha(CKPT),'fit':972,'development':108,'fixed_validation_access':True,'outer_confirmation_test_evaluation':False})
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
   ensure_complete_index()
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
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
 try:train('graymean',resume=a.resume)
 except Exception as e:
  write(RECORD/'STREAM_TRAIN_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
