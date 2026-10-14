"""Public-pretrained native-resolution SegResNet: new original972 run, fixed two A100/global8.

Modes:
  smoke  synthetic canvas-size batches; measures real peak memory for each per-GPU batch candidate
         (largest first), writes GPU_SMOKE.json and BATCH_DECISION.json.  No patient pixels.
  train  200-epoch run reading BATCH_DECISION.json; validation every epoch on the native crop grid,
         split across both ranks; resumes from latest.pth automatically after an interruption.
"""
import sys,os,json,time,traceback,threading,datetime,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from scripts.segresnet_pretrained_native_common import *

class Forward(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x).segmentation_logits


def setup():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    assert dist.get_world_size()==2
    seed_all(SEED);fast_cudnn();return rank,torch.device(f'cuda:{local}')


def clean_exit():
    dist.barrier();torch.cuda.synchronize()
    def force():print(json.dumps({'event':'successful_cleanup_timeout','outputs_committed':True}),flush=True);os._exit(0)
    t=threading.Timer(20,force);t.daemon=True;t.start();dist.destroy_process_group();sys.stdout.flush();os._exit(0)


def optimizer_for(model):return torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,fused=True)
def loss_for():return DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)


def step(net,opt,loss_fn,image,label,distributed=True):
    def all_min(t):
        if distributed:dist.all_reduce(t,op=dist.ReduceOp.MIN)
        return t
    valid=all_min((torch.isfinite(image).all() & torch.isfinite(label).all() & (label.flatten(1).sum(1)>0).all()).to(torch.int32))
    if not bool(valid):raise RuntimeError('nonfinite image/label or empty source foreground')
    image,label,fallbacks=safe_augment(image,label)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        logits=net(image);loss=loss_fn(logits,label)
    if not bool(all_min(torch.isfinite(loss).to(torch.int32))):raise RuntimeError('nonfinite training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
    if not bool(norm>0):raise RuntimeError('zero gradient')
    opt.step()
    with torch.no_grad():dice=dice_per_case(logits.detach().float(),label).mean()
    values=torch.tensor([float(loss.detach()),float(dice),float(norm),float(fallbacks)],device=image.device,dtype=torch.float64)
    if distributed:dist.all_reduce(values);values[:3]/=dist.get_world_size()
    return dict(zip(['loss','train_dice','grad_norm_before_clip','augmentation_fallbacks'],values.cpu().tolist()))


def synthetic(per_gpu,device,rank):
    g=torch.Generator(device='cpu').manual_seed(SEED+rank+per_gpu)
    x=torch.randn(per_gpu,1,*CANVAS,generator=g).to(device);y=torch.zeros_like(x);y[:,:,40:110,60:150,40:110]=1;x+=y*1.5
    return x,y


def probe(per_gpu,device,rank,criterion,benchmark):
    """Per-rank memory/throughput probe without any collective, so an OOM on one rank cannot deadlock the other.

    benchmark=False gives the realistic activation footprint (cudnn autotuning otherwise inflates the peak with
    trial workspaces); benchmark=True gives the training-time speed at the chosen batch.
    """
    import gc
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();seed_all(SEED+rank);torch.backends.cudnn.benchmark=bool(benchmark)
    row={'per_gpu_batch':per_gpu,'cudnn_benchmark':bool(benchmark),'ok':False,'error':None,'seconds_per_step_after_warmup':None,'first_steps':[]}
    model=opt=net=x=y=None
    try:
        model=model_for(device);net=Forward(model);opt=optimizer_for(model);x,y=synthetic(per_gpu,device,rank)
        row['first_steps']=[step(net,opt,criterion,x.clone(),y.clone(),distributed=False) for _ in range(2)]   # includes cudnn autotune
        torch.cuda.synchronize();tick=time.monotonic()
        for _ in range(3):step(net,opt,criterion,x.clone(),y.clone(),distributed=False)
        torch.cuda.synchronize();row['seconds_per_step_after_warmup']=(time.monotonic()-tick)/3;row['ok']=True
    except torch.OutOfMemoryError as exc:row['error']='OOM: '+str(exc)[:300]
    row['peak_allocated_bytes']=torch.cuda.max_memory_allocated();row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    row['fits_ceiling']=bool(row['ok'] and row['peak_reserved_bytes']<=MEMORY_CEILING_BYTES)
    row['steps_per_epoch']=steps_per_epoch(per_gpu);row['estimated_training_seconds_per_epoch']=(row['seconds_per_step_after_warmup']*row['steps_per_epoch']) if row['ok'] else None
    del model,opt,net,x,y;gc.collect();torch.cuda.empty_cache();torch.backends.cudnn.benchmark=True
    return row


def smoke():
    rank,device=setup();verify();criterion=loss_for()
    # Pass A: realistic memory (no cudnn autotuning) for every candidate -> batch decision.
    memory=[probe(b,device,rank,criterion,benchmark=False) for b in BATCH_CANDIDATES]
    gathered=[None,None];dist.all_gather_object(gathered,memory)
    fits={b:all(t['fits_ceiling'] for ranks in gathered for t in ranks if t['per_gpu_batch']==b) for b in BATCH_CANDIDATES}
    # Pass B: training-time speed with autotuning at the chosen batch; fall back to the next candidate if autotuning itself OOMs.
    chosen=None;speed=[]
    for b in BATCH_CANDIDATES:
        if not fits[b]:continue
        t=probe(b,device,rank,criterion,benchmark=True);g=[None,None];dist.all_gather_object(g,t);speed.append(g)
        if all(x['ok'] for x in g):chosen=b;break
    if chosen is None:
        if rank==0:write(R/'GPU_SMOKE.json',{'utc':now(),'status':'failed','reason':'no batch candidate fits the memory ceiling','memory_probe_by_rank':gathered,'speed_probe':speed})
        raise RuntimeError('no batch candidate fits the memory ceiling')
    # DDP consistency at the chosen batch: identical parameters on both ranks after two real distributed steps.
    torch.cuda.reset_peak_memory_stats();seed_all(SEED+rank)
    model=model_for(device);ddp=DDP(Forward(model),device_ids=[device.index]);opt=optimizer_for(model);x,y=synthetic(chosen,device,rank)
    before=next(model.parameters()).detach().clone();values=[step(ddp,opt,criterion,x.clone(),y.clone()) for _ in range(2)]+[step(ddp,opt,criterion,x[:2].clone(),y[:2].clone())]
    maxdiff=torch.zeros((),device=device)
    for p in model.parameters():
        other=p.detach().clone();dist.broadcast(other,0);maxdiff=torch.maximum(maxdiff,(other-p).abs().max())
    change=float((next(model.parameters())-before).abs().max());assert float(maxdiff)==0 and change>0
    assert int(next(iter(opt.state.values()))['step'])==3
    model.eval()
    with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=ddp(x[:2])
    assert logits.shape==(2,1,*CANVAS)
    ddp_row={'rank':rank,'per_gpu_batch':chosen,'distributed_steps':values,'max_parameter_difference':float(maxdiff),'parameter_change':change,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    rows=[None,None];dist.all_gather_object(rows,ddp_row)
    if rank==0:
        write(R/'GPU_SMOKE.json',{'utc':now(),'status':'passed','memory_ceiling_bytes':MEMORY_CEILING_BYTES,'candidates':list(BATCH_CANDIDATES),'fits_on_both_ranks':{str(k):v for k,v in fits.items()},'memory_probe_by_rank_no_benchmark':gathered,'speed_probe_with_benchmark':speed,'ddp_check':rows,'real_model_parameters':PARAMETERS,'patient_pixels_access':False})
        write(R/'BATCH_DECISION.json',{'utc':now(),'per_gpu_batch':chosen,'global_batch':2*chosen,'steps_per_epoch':steps_per_epoch(chosen),'total_updates':EPOCHS*steps_per_epoch(chosen),'rule':'largest per-GPU batch among the frozen candidates whose peak reserved memory (cudnn.benchmark off) stays under the ceiling on both ranks in the synthetic canvas-size probe, and whose autotuned pass also completes','decided_before_training':True,'decided_from_patient_data':False})
        print(json.dumps({'chosen':chosen,'fits':{str(k):v for k,v in fits.items()},'memory_probe':gathered,'speed_probe':speed}),flush=True)
    clean_exit()


@torch.no_grad()
@deterministic_validation
def evaluation(model,dataset,indices,device,rank,epoch):
    """Native-crop-grid Dice/point for `indices`, gathered to rank0; canvas padding is excluded from the metrics."""
    model.eval();rows=[]
    for start in range(0,len(indices),2):
        items=[dataset[i] for i in indices[start:start+2]]
        image=torch.stack([it['image'] for it in items]).to(device,non_blocking=True);label=torch.stack([it['label'] for it in items]).to(device,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits
        for k,it in enumerate(items):
            m=case_metrics(logits[k].float(),label[k],dataset.geometry[it['case_id']]);m.update(case_id=it['case_id'],subject_id=it['subject_id'],index=int(it['index']));rows.append(m)
    gathered=[None,None];dist.all_gather_object(gathered,rows);model.train()
    if rank!=0:return None
    cases=sorted(gathered[0]+gathered[1],key=lambda r:r['index'])
    for r in cases:del r['index']
    result=summarize(cases);assert result['cases']==108 and result['subjects']==50 and all(np.isfinite(r['locator_dsc']) for r in cases)
    result.update(epoch=epoch,created_utc=now(),role='reused fixed validation108/50',grid='native 1mm image-only brain crop (canvas padding excluded)',threshold_raw_logit=0,test_access=False)
    return result


def train():
    rank,device=setup();verify();disk();per_gpu=batch_decision();steps=steps_per_epoch(per_gpu)
    gate=json.loads((R/'EXECUTION_GATE.json').read_text())
    for name,h in gate['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('execution gate changed: '+name)
    data=json.loads((R/'DATA_AUDIT.json').read_text())
    train_data=NativeCanvasDataset(TRAIN_CACHE,expected_split='train',case_ids=data['train_case_ids'])
    val_data=NativeCanvasDataset(VAL_CACHE,expected_split='validation',case_ids=data['validation_case_ids'])
    assert len(train_data)==972 and len(val_data)==108
    val_indices=list(range(108))[rank::2]
    resume=(OUT/'latest.pth').exists()
    if rank==0:
        if OUT.exists() and any(OUT.iterdir()) and not resume:raise RuntimeError('refuse nonempty output directory without latest.pth')
        OUT.mkdir(parents=True,exist_ok=True)
        if not resume:write(R/'TRAIN_PREACCESS.json',{'utc':now(),'mode':'public_pretrain150_native_finetune_single_arm','seed':SEED,'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'per_gpu_batch':per_gpu,'global_batch':2*per_gpu,'steps_per_epoch':steps,'freeze_sha256':sha(R/'FREEZE.json'),'data_audit_sha256':sha(R/'DATA_AUDIT.json'),'batch_decision_sha256':sha(R/'BATCH_DECISION.json'),'prior_test_informed_development':True,'test_pixels_access':False,'old_checkpoint_loaded':True,'source_pretrain_sha256':PRETRAIN_SHA})
    dist.barrier();model=model_for(device)
    if not resume:
        assert sha(PRETRAIN_CKPT)==PRETRAIN_SHA
        pretrained=torch.load(PRETRAIN_CKPT,map_location='cpu',weights_only=False);assert pretrained['epoch']==150
        model.load_state_dict(pretrained['model_state_dict'],strict=True);del pretrained
    ddp=DDP(Forward(model),device_ids=[device.index]);opt=optimizer_for(model);criterion=loss_for()
    start_epoch=1;best=-1.;best_epoch=0;elapsed_before=0.
    if resume:
        state=torch.load(OUT/'latest.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model_state_dict'],strict=True);opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch=int(state['epoch'])+1;best=float(state['best_score']);best_epoch=int(state['best_epoch']);elapsed_before=float(state.get('elapsed_seconds',0.))
        if rank==0:
            with (R/'RESUMES.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'resumed_after_epoch':state['epoch'],'checkpoint_sha256':sha(OUT/'latest.pth')})+'\n')
        del state
    elif rank==0:
        save(OUT/'initial.pth',{'model_state_dict':model.state_dict(),'seed':SEED,'epoch':0,'architecture':'plain MONAI SegResNet4700897','initialization':'public pretrain150','source_sha256':PRETRAIN_SHA})
        write(R/'INITIALIZATION.json',{'utc':now(),'checkpoint_sha256':sha(OUT/'initial.pth'),'seed':SEED,'parameter_count':PARAMETERS,'teacher':False,'distillation':False,'old_checkpoint_loaded':True,'source_pretrain_sha256':PRETRAIN_SHA})
    dist.barrier();started=time.monotonic()-elapsed_before;torch.cuda.reset_peak_memory_stats();epoch=start_epoch-1
    for epoch in range(start_epoch,EPOCHS+1):
        disk();tick=time.monotonic();seed_all(SEED+epoch*10+rank);model.train()
        lr=learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=EPOCHS)
        for group in opt.param_groups:group['lr']=lr
        loader=DataLoader(train_data,batch_sampler=list(batches_for_epoch(epoch,rank,per_gpu)),num_workers=6,pin_memory=True,prefetch_factor=2,generator=torch.Generator().manual_seed(SEED+epoch*100+rank))
        samples=0;n_steps=0;values=[];case_ids=[]
        for number,batch in enumerate(loader,1):
            image=batch['image'].to(device,non_blocking=True);label=batch['label'].to(device,non_blocking=True)
            value=step(ddp,opt,criterion,image,label);n=image.shape[0]*2;samples+=n;n_steps+=1;values.append((n,value));case_ids.extend(batch['case_id'])
            if rank==0 and (number==1 or number%20==0 or number==steps):print(json.dumps({'utc':now(),'epoch':epoch,'step':number,'steps_per_epoch':steps,'global_samples':samples,'learning_rate':lr,**value}),flush=True)
        gathered=[None,None];dist.all_gather_object(gathered,case_ids);ids=gathered[0]+gathered[1]
        if samples!=972 or n_steps!=steps or len(ids)!=972 or len(set(ids))!=972 or set(ids)!=set(data['train_case_ids']):raise RuntimeError('full972 unique coverage gate failed')
        train_seconds=time.monotonic()-tick;vtick=time.monotonic()
        if rank==0 and epoch==1:write(R/'VALIDATION_PREACCESS.json',{'utc':now(),'role':'fixed reused108/50','selection':'scan Dice max; every epoch; earliest tie','grid':'native crop 1mm; padding excluded','threshold_raw_logit':0,'no_tta':True,'no_postprocessing':True,'no_GT_location_or_clicks':True,'freeze_sha256':sha(R/'FREEZE.json'),'test_access':False})
        result=evaluation(model,val_data,val_indices,device,rank,epoch)
        log={'utc':now(),'epoch':epoch,'samples':samples,'unique_scans':len(set(ids)),'optimizer_updates':n_steps,'cumulative_optimizer_updates':epoch*steps,'per_gpu_batch':per_gpu,'learning_rate':lr,'train':{k:sum(n*v[k] for n,v in values)/972 for k in ['loss','train_dice','grad_norm_before_clip']},'augmentation_fallbacks':sum(v['augmentation_fallbacks'] for _,v in values),'training_seconds':train_seconds,'validation_seconds':time.monotonic()-vtick,'peak_allocated_bytes_per_gpu':torch.cuda.max_memory_allocated(),'peak_reserved_bytes_per_gpu':torch.cuda.max_memory_reserved(),'resumed_this_epoch':bool(resume and epoch==start_epoch)}
        if rank==0:
            report=OUT/f'validation_e{epoch:03d}.json';write(report,result)
            log['validation']={k:v for k,v in result.items() if k!='per_case'}
            if result['locator_dsc_mean']>best:
                best=result['locator_dsc_mean'];best_epoch=epoch
                save(OUT/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'method':'public-pretrained SegResNet original972 native resolution canvas','selection':'fixed validation scan Dice on native crop grid','freeze_sha256':sha(R/'FREEZE.json')})
                write(OUT/'BEST.json',{'epoch':epoch,'scan_dice':best,'subject_dice':result['locator_subject_dsc_mean'],'point_hit':result['automatic_point_hit_rate'],'report':report.name,'report_sha256':sha(report),'checkpoint_sha256':sha(OUT/'best.pth')})
            if epoch%10==0:save(OUT/f'epoch{epoch:03d}.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'method':'public-pretrained SegResNet original972 native resolution canvas','freeze_sha256':sha(R/'FREEZE.json')})
        dist.barrier()
        if rank==0:
            save(OUT/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'learning_rate':lr,'schedule':'constant100 then cosine to1e-5 at200','seed':SEED,'per_gpu_batch':per_gpu,'cumulative_optimizer_updates':epoch*steps,'elapsed_seconds':time.monotonic()-started,'freeze_sha256':sha(R/'FREEZE.json')})
            log.update(elapsed_seconds=time.monotonic()-started,checkpoint_sha256=sha(OUT/'latest.pth'),validation_report_sha256=sha(report),selection=json.loads((OUT/'BEST.json').read_text()))
            with (OUT/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            write(OUT/'STATUS.json',{'status':'running','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*steps,'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'test_pixels_access':False});print(json.dumps(log),flush=True)
        dist.barrier()
        budget=torch.tensor(int(time.monotonic()-started>=WALL_BUDGET_SECONDS),device=device);dist.all_reduce(budget,op=dist.ReduceOp.MAX)
        if bool(budget):break
    if epoch==EPOCHS:
        if rank==0:write(R/'RELOAD_PREACCESS.json',{'utc':now(),'role':'same fixed validation108','purpose':'strict reload consistency only; no selection/adaptation','test_access':False})
        dist.barrier();reloads=[];selected=json.loads((OUT/'BEST.json').read_text())
        for name,ep in [('latest.pth',EPOCHS),('best.pth',selected['epoch'])]:
            state=torch.load(OUT/name,map_location='cpu',weights_only=False);assert state['epoch']==ep;model.load_state_dict(state['model_state_dict'],strict=True);del state
            actual=evaluation(model,val_data,val_indices,device,rank,ep)
            if rank==0:
                expected=json.loads((OUT/f'validation_e{ep:03d}.json').read_text());a=actual['per_case'];b=expected['per_case']
                assert [x['case_id'] for x in a]==[x['case_id'] for x in b]
                error=max(abs(x['locator_dsc']-y['locator_dsc']) for x,y in zip(a,b));points=sum(x['automatic_point']!=y['automatic_point'] for x,y in zip(a,b))
                reloads.append({'checkpoint':name,'epoch':ep,'sha256':sha(OUT/name),'max_dice_error':error,'points_differing':points,'strict_load':True,'note':'validation in fixed deterministic mode; tolerance1e-6 Dice'})
                assert error<=1e-6
        if rank==0:write(R/'RELOAD_CHECK.json',{'utc':now(),'status':'passed','checks':reloads})
    if rank==0:write(OUT/'STATUS.json',{'status':'completed' if epoch==EPOCHS else 'incomplete','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*steps,'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'reason':'epoch_budget' if epoch==EPOCHS else 'wall_budget','test_pixels_access':False})
    clean_exit()


if __name__=='__main__':
    try:
        if sys.argv[1]=='smoke':smoke()
        elif sys.argv[1]=='train':train()
        else:raise ValueError('smoke/train mode required')
    except BaseException as exc:write(R/f"FAILURE_{sys.argv[1]}_rank{os.environ.get('RANK','none')}_{int(time.time())}.json",{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
