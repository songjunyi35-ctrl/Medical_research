"""One explicitly frozen Mumford-Shah run; exactly 972 scans and 2 updates/epoch."""
import os,sys,json,time,random,math,traceback,contextlib,threading,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset
from scripts import sammed3d_graymean_single972 as base
from scripts import train_sammed3d_t1ce_wt_upstream_literal_va as upstream
from scripts.sammed3d_mumford_shah_loss import FirstPromptMumfordShah
from scripts.sammed3d_amp_recovery import capture_rng,restore_rng
ROOT=base.ROOT;R=ROOT/'archive/research_records/2026-09-17_sammed3d_mumford_shah_single972'
OUT=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/mumford_shah_single972_20260917'
SEED=20260916;FIELDS=['loss','seg_sum11','ms_weighted','ms_region','ms_tv','train_final11_dice']
sha=base.sha;write=base.write;now=base.now


def windows(epoch,rank):
    g=torch.Generator().manual_seed(SEED+epoch)
    torch.empty((),dtype=torch.int64).random_(generator=g)
    order=torch.randperm(972,generator=g).tolist();batches=[]
    for start in range(0,972,8):
        ids=order[start:start+8];half=len(ids)//2;batches.append(ids[rank*half:(rank+1)*half])
    result=[batches[:63],batches[63:]]
    assert [sum(map(len,w))*2 for w in result]==[504,468]
    return result


class Forward(torch.nn.Module):
    def __init__(self,sam):super().__init__();self.sam=sam
    def forward(self,image,target):
        loss_fn=FirstPromptMumfordShah(image)
        logits,loss,_=upstream.upstream_train_interaction(model=self.sam,image=image,target=target,loss_fn=loss_fn)
        if loss_fn.calls!=11:raise RuntimeError('11 interactions required')
        with torch.no_grad():
            pred=logits>0;gt=target>0;dims=tuple(range(1,gt.ndim))
            dice=(2*(pred & gt).sum(dims)/(pred.sum(dims)+gt.sum(dims))).float().mean()
        metrics=torch.stack([loss.detach().float(),loss_fn.seg_sum,loss_fn.weighted.detach(),loss_fn.region.detach(),loss_fn.tv.detach(),dice])
        return loss,metrics


def run_window(ddp,opt,scaler,dataset,batches,device,loader_seed,workers=2,context=None):
    """A successful window updates once. Overflow replays the whole same window."""
    local_count=sum(map(len,batches));rng=capture_rng(device)
    buffers={n:b.detach().clone() for n,b in ddp.named_buffers()}
    for attempt in range(9):
        if attempt:
            restore_rng(rng,device)
            with torch.no_grad():
                for n,b in ddp.named_buffers():b.copy_(buffers[n])
        opt.zero_grad(set_to_none=True);totals=torch.zeros(len(FIELDS),device=device,dtype=torch.float64);seen=0
        generator=torch.Generator().manual_seed(loader_seed)
        loader=DataLoader(dataset,batch_sampler=batches,num_workers=workers,pin_memory=True,generator=generator)
        for i,batch in enumerate(loader):
            n=len(batch['image']);x=batch['image'].to(device,non_blocking=True);y=batch['label'].to(device,non_blocking=True)
            sync=ddp.no_sync() if i+1<len(batches) else contextlib.nullcontext()
            with sync:
                with torch.amp.autocast('cuda',dtype=torch.float16):loss,metrics=ddp(x,y)
                valid=(torch.isfinite(loss) & torch.isfinite(metrics).all()).to(torch.int32)
                dist.all_reduce(valid,op=dist.ReduceOp.MIN)
                if not bool(valid):raise RuntimeError('nonfinite loss/metric')
                scaler.scale(loss*(n/local_count)).backward()
            totals+=metrics.detach().double()*n;seen+=n
            if dist.get_rank()==0 and context and (i==0 or (i+1)%10==0 or i+1==len(batches)):
                print(json.dumps({'utc':now(),**context,'attempt':attempt,'microbatch':i+1,'microbatches_in_window':len(batches),'window_samples_seen_global':seen*2,'loss':float(loss.detach()),'amp_scale':scaler.get_scale()}),flush=True)
        if seen!=local_count:raise RuntimeError('window sample count mismatch')
        scaler.unscale_(opt)
        grads=[p.grad for p in ddp.parameters() if p.grad is not None]
        if not grads:raise RuntimeError('missing gradient')
        norm=torch.stack([g.detach().norm(dtype=torch.float64) for g in grads]).norm()
        finite=torch.isfinite(norm).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not bool(finite):
            if attempt==8 or scaler.get_scale()<=1:raise RuntimeError('gradient overflow persists after bounded window replay')
            before=scaler.get_scale();scaler.update(new_scale=before*scaler.get_backoff_factor())
            state=scaler.state_dict();state['_growth_tracker']=0;scaler.load_state_dict(state)
            if dist.get_rank()==0:print(json.dumps({'utc':now(),'event':'amp_window_replay','scale_before':before,'scale_after':scaler.get_scale(),'attempt':attempt+1,'optimizer_updated':False,**(context or {})}),flush=True)
            continue
        positive=(norm>0).to(torch.int32);dist.all_reduce(positive,op=dist.ReduceOp.MIN)
        if not bool(positive):raise RuntimeError('zero gradient')
        scaler.step(opt);scaler.update();dist.all_reduce(totals)
        count=local_count*dist.get_world_size()
        return {**dict(zip(FIELDS,(totals/count).cpu().tolist())),'grad_norm':float(norm),'amp_scale':scaler.get_scale(),'amp_window_retries':attempt,'samples':count,'optimizer_updates':1}
    raise AssertionError('unreachable')


def setup():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    if dist.get_world_size()!=2:raise RuntimeError('exactly two GPUs required')
    base.seed_all(SEED);return rank,torch.device(f'cuda:{local}')


def verify():
    freeze=json.loads((R/'FREEZE.json').read_text())
    for name,digest in freeze['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('frozen input changed: '+name)
    return freeze


def successful_exit():
    # This worker image previously hung during successful NCCL/Python teardown.
    dist.barrier();torch.cuda.synchronize()
    def force_exit():
        print(json.dumps({'event':'successful_cleanup_timeout','durable_outputs_committed':True}),flush=True);os._exit(0)
    timer=threading.Timer(20,force_exit);timer.daemon=True;timer.start()
    dist.destroy_process_group();sys.stdout.flush();os._exit(0)


class Synth(Dataset):
    def __init__(self):
        g=torch.Generator().manual_seed(947)
        self.x=torch.randn(4,1,128,128,128,generator=g);self.y=torch.zeros_like(self.x)
        self.y[:,:,40:88,40:88,40:88]=1;self.x+=self.y*1.5
    def __len__(self):return 4
    def __getitem__(self,i):return {'image':self.x[i],'label':self.y[i]}


def smoke():
    rank,device=setup();verify();model=base.build_model(device)
    ddp=DDP(Forward(model),device_ids=[device.index],find_unused_parameters=True)
    opt=base.optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    before=next(model.image_encoder.parameters()).detach().clone();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    dataset=Synth();results=[]
    for k,batches in enumerate([[[0,1,2,3],[0,1,2,3]],[[0,1,2,3],[0,1]]]):
        results.append(run_window(ddp,opt,scaler,dataset,batches,device,SEED+k,workers=0))
    torch.cuda.synchronize();difference=torch.zeros((),device=device)
    for param in model.parameters():
        other=param.detach().clone();dist.broadcast(other,0);difference=torch.maximum(difference,(other-param).abs().max())
    change=float((next(model.image_encoder.parameters())-before).abs().max())
    assert float(difference)==0 and change>0
    assert int(next(iter(opt.state.values()))['step'])==2
    row={'rank':rank,'status':'passed','windows':results,'optimizer_updates':2,'parameter_max_difference':float(difference),'encoder_change':change,'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'data_access':False}
    rows=[None,None];dist.all_gather_object(rows,row)
    if rank==0:write(R/'GPU_SMOKE.json',{'utc':now(),'status':'passed','ranks':rows});print(json.dumps(rows),flush=True)
    successful_exit()


def train():
    rank,device=setup();verify();base.disk(6)
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    assert json.loads((R/'ACCUMULATION_TEST.json').read_text())['status']=='passed'
    gate=json.loads((R/'EXECUTION_GATE.json').read_text())
    for name,digest in gate['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('execution gate changed: '+name)
    if max(r['peak_reserved_bytes'] for r in json.loads((R/'GPU_SMOKE.json').read_text())['ranks'])>34*(1<<30):raise RuntimeError('GPU memory gate')
    audit=json.loads((R/'DATA_AUDIT.json').read_text());index=json.loads((base.RECORD/'CACHE_INDEX.json').read_text())
    if sha(base.RECORD/'CACHE_INDEX.json')!=audit['cache_index_sha256']:raise RuntimeError('cache index changed')
    items={role:[r for r in index['cases'] if r['role']==role] for role in ['fit','development']}
    if [len(items[k]) for k in ['fit','development']]!=[972,108]:raise RuntimeError('fixed split mismatch')
    if rank==0:
        if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('refusing existing training directory')
        OUT.mkdir(parents=True)
        write(R/'TRAIN_PREACCESS.json',{'utc':now(),'initial_checkpoint_sha256':sha(base.CKPT),'freeze_sha256':sha(R/'FREEZE.json'),'cache_index_sha256':sha(base.RECORD/'CACHE_INDEX.json'),'train_scans':972,'validation_scans':108,'test_pixels_access':False,'prior_test_informed_method_choice':True,'source':'fresh Turbo; old learned weights not loaded'})
    dist.barrier()
    model=base.build_model(device);model.train();opt=base.optimizer_for(model)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[120,180],gamma=.1)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    ddp=DDP(Forward(model),device_ids=[device.index],find_unused_parameters=True)
    started=time.monotonic();best=-1.;best_epoch=0;updates=0;torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,201):
        base.disk(6);tick=time.monotonic();base.seed_all(SEED+epoch*10+rank)
        dataset=base.Cases(items['fit'],True,epoch);values=[]
        for k,batches in enumerate(windows(epoch,rank)):
            value=run_window(ddp,opt,scaler,dataset,batches,device,SEED+epoch*100+k*10+rank,context={'epoch':epoch,'update_window':k+1});values.append(value);updates+=1
        samples=sum(v['samples'] for v in values)
        if samples!=972 or len(values)!=2 or updates!=epoch*2:raise RuntimeError('exact972/two updates gate')
        scheduler.step();training_seconds=time.monotonic()-tick
        log={'utc':now(),'epoch':epoch,'samples':samples,'microbatches':122,'optimizer_updates':2,'cumulative_optimizer_updates':updates,'window_samples':[v['samples'] for v in values],'microbatch_per_gpu':4,'world_size':2,'train':{k:sum(v[k]*v['samples'] for v in values)/972 for k in FIELDS},'window_grad_norms':[v['grad_norm'] for v in values],'amp_window_retries':sum(v['amp_window_retries'] for v in values),'amp_scale':scaler.get_scale(),'training_seconds':training_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        if rank==0:
            if epoch==1:write(R/'VALIDATION_PREACCESS.json',{'utc':now(),'role':'fixed reused validation108/50','freeze_sha256':sha(R/'FREEZE.json'),'selection':'best5 scan Dice; every epoch; earlier tie','test_access':False})
            validation=base.evaluate(model,items['development'],device)
            subjects=sorted({r['subject_id'] for r in validation['rows']})
            validation['subject_best5']=float(np.mean([np.mean([r['best5'] for r in validation['rows'] if r['subject_id']==s]) for s in subjects]))
            validation.update(epoch=epoch,created_utc=now(),method='mumford_shah_single972')
            report=OUT/f'validation_e{epoch:03d}.json';write(report,validation)
            log['validation']={k:v for k,v in validation.items() if k!='rows'}
            if validation['scan_best5']>best:
                best=validation['scan_best5'];best_epoch=epoch
                base.atomic_torch_save(OUT/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'method':'mumford_shah_single972','selection':'fixed validation108 best5 scan Dice','freeze_sha256':sha(R/'FREEZE.json')})
                write(OUT/'BEST.json',{'epoch':epoch,'scan_best5':best,'report':report.name,'report_sha256':sha(report),'checkpoint_sha256':sha(OUT/'best.pth')})
        dist.barrier()
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)};rngs=[None,None];dist.all_gather_object(rngs,rng)
        if rank==0:
            log.update(elapsed_seconds=time.monotonic()-started)
            base.atomic_torch_save(OUT/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'cumulative_optimizer_updates':updates,'rng_by_rank':rngs,'world_size':2,'microbatch_per_gpu':4,'method':'mumford_shah_single972','freeze_sha256':sha(R/'FREEZE.json')})
            log['checkpoint_sha256']=sha(OUT/'latest.pth');log['validation_report_sha256']=sha(report);log['selection']=json.loads((OUT/'BEST.json').read_text())
            with (OUT/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            write(OUT/'STATUS.json',{'status':'running','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_score':best,'cumulative_optimizer_updates':updates,'updated_utc':now(),'elapsed_seconds':time.monotonic()-started,'test_pixels_access':False});print(json.dumps(log),flush=True)
        dist.barrier()
        budget=time.monotonic()-started>=24*3600
        if budget:break
    if rank==0:
        write(OUT/'STATUS.json',{'status':'completed' if epoch==200 else 'incomplete','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_score':best,'cumulative_optimizer_updates':updates,'updated_utc':now(),'elapsed_seconds':time.monotonic()-started,'reason':'epoch_budget' if epoch==200 else 'wall_budget','test_pixels_access':False})
    successful_exit()

if __name__=='__main__':
    try:
        if sys.argv[1]=='smoke':smoke()
        elif sys.argv[1]=='train':train()
        else:raise ValueError('smoke or train required')
    except BaseException as exc:
        write(R/f"FAILURE_{sys.argv[1]}_rank{os.environ.get('RANK','none')}.json",{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
