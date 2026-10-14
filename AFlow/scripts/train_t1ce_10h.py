"""Full972 coverage, DDP patch arm or single-GPU weighted/Tversky arm."""
import sys, time, datetime, math, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from scripts.t1ce_10h_common import *
from scripts.train_t1ce_wt_capg3d import augment_batch

class Forward(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):return self.model(x).segmentation_logits

def setup(arm):
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
 assert world==(2 if arm=='A' else 1);torch.cuda.set_device(local);torch.set_num_threads(2);seed(SEED);numeric()
 if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=10))
 return rank,world,torch.device('cuda',local)
def barrier(world):
 if world>1:dist.barrier()
def consensus(flag,device,world):
 t=torch.tensor(int(flag),device=device)
 if world>1:dist.all_reduce(t,op=dist.ReduceOp.MAX)
 return bool(t)
def trainstep(net,opt,x,y,w,arm):
 if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('nonfinite training tensors')
 before=y.flatten(1).sum(1)>0;oldx=x.clone();oldy=y.clone();x,y=augment_batch(x,y,max_shift=8)
 lost=before & (y.flatten(1).sum(1)==0)
 if bool(lost.any()):x[lost]=oldx[lost];y[lost]=oldy[lost]
 del oldx,oldy
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x);loss=criterion(z,y,w,arm)
 if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
 if not bool(norm>0):raise RuntimeError('zero gradient')
 opt.step();return float(loss.detach()),float(norm)

def smoke(arm):
 verify();rank,world,device=setup(arm);model=model_for(device);net=Forward(model)
 if world>1:net=DDP(net,device_ids=[device.index])
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-5);b=8//world
 x=torch.randn(b,1,128,128,128,device=device);y=torch.zeros_like(x);y[1:,:,30:90,30:90,30:90]=1
 values=[];torch.cuda.reset_peak_memory_stats()
 for n in [b,4//world]:values.append(trainstep(net,opt,x[:n],y[:n],torch.ones(n,device=device),arm))
 peak=torch.cuda.max_memory_reserved();assert peak<=36*2**30
 if world>1:
  for p in model.parameters():
   ref=p.detach().clone();dist.broadcast(ref,0);assert torch.equal(ref,p.detach())
 if rank==0:write(R/arm/'GPU_SMOKE.json',{'utc':now(),'status':'passed','world':world,'global_batch':8,'tail_global':4,'empty_patch_loss_finite':True,'peak_reserved_bytes':peak,'steps':values,'patient_pixels_access':False})
 barrier(world)
 if world>1:dist.destroy_process_group()

def validation(model,mode,epoch,rank,world):
 cases,seconds=evaluate(model,mode,'validation',rank,world)
 if world>1:
  allcases=[None]*world;dist.all_gather_object(allcases,cases);cases=[r for c in allcases for r in c]
 order={r['case_id']:i for i,r in enumerate(rows('validation'))};cases.sort(key=lambda r:order[r['case_id']])
 assert len(cases)==108 and len({c['case_id'] for c in cases})==108 and len({c['subject_id'] for c in cases})==50
 return {'utc':now(),'epoch':epoch,'split':'validation','metrics':summary(cases),'per_case':cases,'seconds':seconds,'test_tuned_development':True}

def gate_test(model,arm,mode,best_path,epoch,rank,world):
 barrier(world);answer=None
 if rank==0:
  from scripts.evaluate_t1ce_10h import score_checkpoint
  h=sha(best_path);path=OUT/arm/f'selected_{h[:16]}.pth'
  if not path.exists():shutil.copy2(best_path,path)
  answer=score_checkpoint(path,mode,f'{arm}_{h[:16]}',arm=arm,gate_epoch=epoch)
  event(R/arm/'TEST_GATES.jsonl',epoch=epoch,id=answer['id'],scan_dice=answer['metrics']['scan_dice'],checkpoint_sha256=h)
 if world>1:
  shared=[answer];dist.broadcast_object_list(shared,0);answer=shared[0]
 return answer

def train(arm):
 verify();assert read(R/arm/'GPU_SMOKE.json')['status']=='passed'
 execution=read(R/'EXECUTION_FREEZE.json')
 for p,h in execution['files'].items():assert sha(ROOT/p)==h,p
 rank,world,device=setup(arm);mode='patch' if arm=='A' else '128';out=OUT/arm;out.mkdir(parents=True,exist_ok=True)
 model=model_for(device,source(P['sources'][arm]));net=Forward(model)
 if world>1:net=DDP(net,device_ids=[device.index])
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-5)
 weights=read(R/'B/WEIGHTS.json')['case_weights'] if arm=='B' else {r['case_id']:1. for r in rows('train')}
 history=[];best=-1.;bestepoch=0;start=1;elapsed_before=0.;tests=[]
 if (out/'latest.pth').exists():
  state=torch.load(out/'latest.pth',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict']);opt.load_state_dict(state['optimizer_state_dict']);start=state['epoch']+1;best=state['best'];bestepoch=state['best_epoch'];history=state['history'];tests=state['tests'];elapsed_before=state['elapsed'];del state
 else:
  if rank==0:write(R/arm/'TRAIN_PREACCESS.json',{'utc':now(),'source_sha256':sha(source(P['sources'][arm])),'source':P['sources'][arm],'train':972,'validation':108,'segmentation_training_test_labels':False,'test_tuned':True,'execution_freeze_sha256':sha(R/'EXECUTION_FREEZE.json')})
  val=validation(model,mode,0,rank,world);best=val['metrics']['scan_dice'];history=[{'epoch':0,'score':best}]
  if rank==0:
   write(out/'validation_e000.json',val);save(out/'best.pth',{'model_state_dict':model.state_dict(),'epoch':0,'arm':arm});write(out/'BEST.json',{'epoch':0,'score':best,'report':'validation_e000.json','sha256':sha(out/'best.pth')})
 barrier(world);started=time.monotonic()-elapsed_before;reason=None;last=start-1;partial=False
 for epoch in range(start,P['candidate_epochs'][arm]+1):
  if consensus(time.time()>=timestamp(P['training_stop_utc']) or time.monotonic()-started>P['candidate_wall_hours'][arm]*3600,device,world):reason='wall budget';break
  disk();seed(SEED+epoch*10+rank);model.train();ds=GridDataset('train',native=(arm=='A'),patch=(arm=='A'),epoch=epoch)
  loader=DataLoader(ds,batch_sampler=batches(epoch,rank,world),num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(SEED+epoch*100+rank))
  lr=1e-6+.5*(2e-5-1e-6)*(1+math.cos(math.pi*(epoch-1)/max(1,P['candidate_epochs'][arm]-1)))
  for g in opt.param_groups:g['lr']=lr
  ids=[];losses=[];tick=time.monotonic()
  for step,b in enumerate(loader,1):
   if consensus(time.time()>=timestamp(P['training_stop_utc']) or (R/'STOP').exists(),device,world):partial=True;reason='global deadline or safety stop';break
   x=b['image'].to(device,non_blocking=True);y=b['label'].to(device,non_blocking=True);w=torch.tensor([weights[c] for c in b['case_id']],device=device)
   loss,norm=trainstep(net,opt,x,y,w,arm);ids+=b['case_id'];losses.append((len(b['case_id']),loss))
   if rank==0 and (step==1 or step%20==0 or step==122):
    status={'utc':now(),'state':'training','arm':arm,'epoch':epoch,'step':step,'last_complete_epoch':last,'cumulative_completed_updates':last*122,'loss':loss,'grad_norm':norm,'lr':lr,'elapsed':time.monotonic()-started,'best_epoch':bestepoch,'best_scan':best,'test_tuned':True}
    write(out/'STATUS.json',status);print(json.dumps(status),flush=True)
  if partial:break
  if world>1:
   gathered=[None]*world;dist.all_gather_object(gathered,ids);ids=[x for g in gathered for x in g]
  assert len(ids)==len(set(ids))==972 and set(ids)=={r['case_id'] for r in rows('train')} and step==122
  last=epoch;val=None
  if epoch==1 or epoch%5==0:
   val=validation(model,mode,epoch,rank,world);score=val['metrics']['scan_dice'];history.append({'epoch':epoch,'score':score})
   if rank==0:write(out/f'validation_e{epoch:03d}.json',val)
   if score>best:
    best=score;bestepoch=epoch
    if rank==0:
     save(out/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'arm':arm});write(out/'BEST.json',{'epoch':epoch,'score':best,'report':f'validation_e{epoch:03d}.json','sha256':sha(out/'best.pth')})
   reason=stop_rule(history)
  barrier(world)
  if epoch in P['test_gate_epochs'] or reason:
   report=gate_test(model,arm,mode,out/'best.pth',epoch,rank,world)
   if not tests or tests[-1]['id']!=report['id']:tests.append({'id':report['id'],'score':report['metrics']['scan_dice']})
   ref=read(R/'REFERENCE_PANEL.json')['strongest_scan']
   if len(tests)>=2 and all(t['score']<ref-.01 for t in tests[-2:]) and tests[-1]['score']-tests[-2]['score']<.002:reason=reason or 'two target gates below reference without progress'
  if rank==0:
   save(out/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'best':best,'best_epoch':bestepoch,'history':history,'tests':tests,'elapsed':time.monotonic()-started})
   event(out/'TRAIN.jsonl',epoch=epoch,unique_scans=972,updates=122,loss=sum(n*l for n,l in losses)/sum(n for n,l in losses),seconds=time.monotonic()-tick,best_score=best,validation_score=val['metrics']['scan_dice'] if val else None,stop_reason=reason)
  barrier(world)
  if reason:break
 if not partial:
  # Only a not-yet-evaluated validation best may consume a final remaining slot.
  naccess=len(list((R/'access').glob(arm+'_*.json')))
  if naccess<4:gate_test(model,arm,mode,out/'best.pth',last,rank,world)
 if rank==0:
  write(out/'STATUS.json',{'utc':now(),'state':'stopped' if reason else 'completed','arm':arm,'last_complete_epoch':last,'partial_next_epoch':partial,'completed_updates':last*122,'best_epoch':bestepoch,'best_scan':best,'reason':reason or 'epoch budget','elapsed':time.monotonic()-started,'test_tuned':True,'new_outer_confirmation_access':False})
  write(R/arm/'STOP_RESULT.json',read(out/'STATUS.json'))
 barrier(world)
 if world>1:dist.destroy_process_group()

if __name__=='__main__':
 arm=sys.argv[2]
 try:{'smoke':smoke,'train':train}[sys.argv[1]](arm)
 except BaseException:
  import traceback
  write(R/arm/f'FAILURE_{sys.argv[1]}_{os.environ.get("RANK","0")}_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
