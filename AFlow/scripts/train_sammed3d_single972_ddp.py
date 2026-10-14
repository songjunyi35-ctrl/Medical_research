"""Two GPUs, one grayscale model, fixed972 with no padded or dropped cases."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_sammed3d_single972_stream import Cases, ensure_complete_index
from scripts.sammed3d_graymean_single972 import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datetime import timedelta

class GrayForward(torch.nn.Module):
 def __init__(self,sam):super().__init__();self.sam=sam
 def forward(self,image,target):
  loss_fn=FirstPromptLoss(image,1.)
  _,loss,_=upstream.upstream_train_interaction(model=self.sam,image=image,target=target,loss_fn=loss_fn)
  if loss_fn.calls!=11:raise RuntimeError('11 interactions required')
  return loss,loss_fn.gray.detach()

def epoch_batches(epoch,per_gpu,rank):
 g=torch.Generator().manual_seed(SEED+epoch)
 # Match the single DataLoader generator's base_seed draw before RandomSampler.
 torch.empty((),dtype=torch.int64).random_(generator=g)
 order=torch.randperm(972,generator=g).tolist();size=per_gpu*2
 for start in range(0,972,size):
  whole=order[start:start+size]
  if len(whole)%2:raise RuntimeError('unequal last rank batch')
  half=len(whole)//2
  yield whole[rank*half:(rank+1)*half]

def step(ddp,opt,scaler,image,target):
 opt.zero_grad(set_to_none=True)
 with torch.amp.autocast('cuda',dtype=torch.float16):loss,gray=ddp(image,target)
 if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
 scaler.scale(loss).backward();scaler.unscale_(opt)
 grads=[p.grad for p in ddp.parameters() if p.grad is not None]
 norms=torch.stack([g.detach().float().norm() for g in grads]);norm=norms.norm()
 valid=torch.isfinite(norm)&(norm>0)
 dist.all_reduce(valid,op=dist.ReduceOp.MIN)
 if not bool(valid):raise RuntimeError('nonfinite/zero gradient')
 scaler.step(opt);scaler.update()
 summary=torch.stack([loss.detach().float(),gray.float(),norm]);dist.all_reduce(summary);summary/=dist.get_world_size()
 return dict(zip(['loss','gray','grad_norm'],summary.cpu().tolist()))

def setup():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
 dist.init_process_group('nccl',timeout=timedelta(minutes=60));seed_all(SEED)
 return rank,torch.device(f'cuda:{local}')

def smoke(per_gpu):
 rank,device=setup();model=build_model(device);ddp=DistributedDataParallel(GrayForward(model),device_ids=[device.index],find_unused_parameters=True)
 opt=optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
 image=torch.randn(per_gpu,1,128,128,128,device=device);target=torch.zeros_like(image);target[:,:,40:88,40:88,40:88]=1;image+=target*1.5
 before=next(model.image_encoder.parameters()).detach().clone();torch.cuda.reset_peak_memory_stats();t=time.monotonic();values=[]
 for _ in range(2):values.append(step(ddp,opt,scaler,image,target))
 # Exercise the real short final batch when global batch8 leaves4 samples.
 values.append(step(ddp,opt,scaler,image[:2],target[:2]))
 torch.cuda.synchronize();changed=float((next(model.image_encoder.parameters()).detach()-before).abs().max())
 maxdiff=torch.zeros((),device=device)
 for p in model.parameters():
  other=p.detach().clone();dist.broadcast(other,src=0);maxdiff=torch.maximum(maxdiff,(p.detach()-other).abs().max())
 if float(maxdiff)>0 or changed<=0:raise RuntimeError('DDP parameter synchronization failure')
 report={'rank':rank,'batch_per_gpu':per_gpu,'world_size':2,'steps':values,'seconds':time.monotonic()-t,'encoder_change':changed,'max_parameter_difference':float(maxdiff),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'data_access':False}
 gathered=[None,None];dist.all_gather_object(gathered,report)
 if rank==0:write(RECORD/'DDP_SMOKE.json',{'status':'passed','utc':now(),'ranks':gathered});print(json.dumps(gathered),flush=True)
 dist.destroy_process_group()

def train(per_gpu):
 rank,device=setup();check_freeze();disk(6)
 freeze=json.loads((RECORD/'DDP_FREEZE.json').read_text())
 for name,digest in freeze['files'].items():
  if sha(ROOT/name)!=digest:raise RuntimeError('DDP frozen file changed')
 if json.loads((RECORD/'DDP_SMOKE.json').read_text())['status']!='passed':raise RuntimeError('DDP smoke gate')
 transition=json.loads((RECORD/'DDP_TRANSITION.json').read_text());checkpoint=ROOT/transition['checkpoint_path']
 if sha(checkpoint)!=transition['checkpoint_sha256']:raise RuntimeError('transition checkpoint changed')
 plan=json.loads((RECORD/'INPUT_PLAN.json').read_text());items={role:[r for r in plan['cases'] if r['role']==role] for role in ['fit','development']}
 if [len(items[k]) for k in ['fit','development']]!=[972,108]:raise RuntimeError('original split counts')
 out=WORK/'graymean';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=build_model(device);model.load_state_dict(saved['model_state_dict'],strict=True);model.train();opt=optimizer_for(model);opt.load_state_dict(saved['optimizer_state_dict'])
 scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[120,180],gamma=.1);scheduler.load_state_dict(saved['scheduler_state_dict']);scaler=torch.amp.GradScaler('cuda',init_scale=128.);scaler.load_state_dict(saved['scaler_state_dict'])
 first=saved['epoch']+1;best=saved['best_score'];best_epoch=saved['best_epoch'];best_report=f'development_e{best_epoch:03d}.json' if best_epoch else None;del saved
 ddp=DistributedDataParallel(GrayForward(model),device_ids=[device.index],find_unused_parameters=True)
 started=time.monotonic();elapsed_prior=transition['elapsed_seconds'];torch.cuda.reset_peak_memory_stats()
 if rank==0:write(out/'DDP_PREACCESS.json',{'utc':now(),'first_epoch':first,'batch_per_gpu':per_gpu,'global_batch':2*per_gpu,'world_size':2,'checkpoint_sha256':sha(checkpoint),'ddp_freeze_sha256':sha(RECORD/'DDP_FREEZE.json'),'input_plan_sha256':sha(RECORD/'INPUT_PLAN.json'),'test_access':False})
 for epoch in range(first,201):
  disk(6);tick=time.monotonic();seed_all(SEED+epoch*10+rank)
  dataset=Cases(items['fit'],True,epoch);batches=list(epoch_batches(epoch,per_gpu,rank));loader=DataLoader(dataset,batch_sampler=batches,num_workers=2,pin_memory=True)
  totals=[];samples=0
  for number,batch in enumerate(loader,1):
   if number%16==1:disk(6)
   value=step(ddp,opt,scaler,batch['image'].to(device,non_blocking=True),batch['label'].to(device,non_blocking=True));n=len(batch['case_id'])*2;samples+=n;totals.append((n,value))
   if rank==0 and (number%10==0 or number==1):print(json.dumps({'utc':now(),'epoch':epoch,'step':number,'steps_per_epoch':len(batches),'samples_seen':samples,'global_batch':2*per_gpu,**value}),flush=True)
  if samples!=972:raise RuntimeError('missing/repeated samples')
  scheduler.step();elapsed=elapsed_prior+time.monotonic()-started;budget=elapsed>=48*3600
  log={'utc':now(),'arm':'graymean','epoch':epoch,'steps':len(totals),'samples':samples,'global_batch':2*per_gpu,'world_size':2,'train':{k:sum(n*v[k] for n,v in totals)/samples for k in totals[0][1]},'training_seconds':time.monotonic()-tick,'elapsed_seconds':elapsed,'ddp_elapsed_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
  if epoch%5==0 or budget or epoch==200:
   if rank==0:
    ensure_complete_index()
    if not (out/'DEVELOPMENT_PREACCESS.json').exists():write(out/'DEVELOPMENT_PREACCESS.json',{'utc':now(),'epoch':epoch,'role':'reused fixed validation108/50','prereg_sha256':sha(RECORD/'PREREG.json'),'ddp_freeze_sha256':sha(RECORD/'DDP_FREEZE.json'),'test_access':False})
    result=evaluate(model,items['development'],device);result.update(epoch=epoch,arm='graymean',created_utc=now());write(out/f'development_e{epoch:03d}.json',result);log['development']={k:v for k,v in result.items() if k!='rows'}
    if result['scan_final5']>best:
     best=result['scan_final5'];best_epoch=epoch;best_report=f'development_e{epoch:03d}.json';atomic_torch_save(out/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'arm':'graymean','gray_weight':1.,'selection':'fixed validation108 final5 scan Dice'})
     write(out/'BEST.json',{'epoch':epoch,'scan_final5':best,'report':best_report,'report_sha256':sha(out/best_report),'checkpoint_sha256':sha(out/'best.pth')})
   dist.barrier()
  rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)};rngs=[None,None];dist.all_gather_object(rngs,rng)
  if rank==0:
   atomic_torch_save(out/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'rng_by_rank':rngs,'world_size':2,'batch_per_gpu':per_gpu,'arm':'graymean','prereg_sha256':sha(RECORD/'PREREG.json'),'ddp_freeze_sha256':sha(RECORD/'DDP_FREEZE.json')})
   with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
   write(out/'STATUS.json',{'status':'running','last_complete_epoch':epoch,'best_epoch':best_epoch,'updated_utc':now(),**{k:v for k,v in log.items() if k not in ['train','development']}});print(json.dumps(log),flush=True)
  dist.barrier()
  if budget:break
 if rank==0:write(out/'STATUS.json',{'status':'completed','reason':'wall_budget' if budget else 'epoch_budget','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_score':best,'best_report':best_report,'updated_utc':now(),'elapsed_seconds':elapsed_prior+time.monotonic()-started,'ddp_elapsed_seconds':time.monotonic()-started,'single_gpu_elapsed_seconds':elapsed_prior,'checkpoint_sha256':sha(out/'best.pth') if best_epoch else None,'test_access':False,'world_size':2,'global_batch':per_gpu*2})
 dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train']);p.add_argument('--batch',type=int,required=True);a=p.parse_args()
 try:
  if a.mode=='smoke':smoke(a.batch)
  else:train(a.batch)
 except Exception as e:
  rank=os.environ.get('RANK','none');write(RECORD/f'DDP_FAILURE_{a.mode}_rank{rank}.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
