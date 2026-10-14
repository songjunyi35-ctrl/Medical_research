"""Fresh pure SegResNet: fixed972, full coverage, global batch8, single run."""
import sys,os,json,time,traceback,threading,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset
from monai.losses import DiceCELoss
from scripts.segresnet_plain972_no_n4_common import *

class Forward(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x).segmentation_logits


def setup():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    assert dist.get_world_size()==2
    seed_all(SEED);return rank,torch.device(f'cuda:{local}')


def clean_exit():
    dist.barrier();torch.cuda.synchronize()
    def force():print(json.dumps({'event':'successful_cleanup_timeout','outputs_committed':True}),flush=True);os._exit(0)
    t=threading.Timer(20,force);t.daemon=True;t.start();dist.destroy_process_group();sys.stdout.flush();os._exit(0)


def optimizer_for(model):return torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
def loss_for():return DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)


def step(ddp,opt,loss_fn,image,label):
    valid=(torch.isfinite(image).all() & torch.isfinite(label).all() & (label.flatten(1).sum(1)>0).all()).to(torch.int32)
    dist.all_reduce(valid,op=dist.ReduceOp.MIN)
    if not bool(valid):raise RuntimeError('nonfinite image/label or empty source foreground')
    image,label,fallbacks=safe_augment(image,label)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        logits=ddp(image);loss=loss_fn(logits,label)
    valid=torch.isfinite(loss).to(torch.int32);dist.all_reduce(valid,op=dist.ReduceOp.MIN)
    if not bool(valid):raise RuntimeError('nonfinite training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),5.,error_if_nonfinite=True)
    if not bool(norm>0):raise RuntimeError('zero gradient')
    opt.step()
    with torch.no_grad():dice=dice_per_case(logits.detach().float(),label).mean()
    values=torch.tensor([float(loss.detach()),float(dice),float(norm),float(fallbacks)],device=image.device,dtype=torch.float64)
    dist.all_reduce(values);values[:3]/=dist.get_world_size()
    return dict(zip(['loss','train_dice','grad_norm_before_clip','augmentation_fallbacks'],values.cpu().tolist()))


def smoke():
    rank,device=setup();verify();model=model_for(device);ddp=DDP(Forward(model),device_ids=[device.index]);opt=optimizer_for(model);criterion=loss_for()
    seed_all(SEED+rank);x=torch.randn(4,1,128,128,128,device=device);y=torch.zeros_like(x);y[:,:,36:92,36:92,36:92]=1;x+=y*1.5
    before=next(model.parameters()).detach().clone();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    values=[step(ddp,opt,criterion,x.clone(),y.clone()),step(ddp,opt,criterion,x[:2].clone(),y[:2].clone())]
    maxdiff=torch.zeros((),device=device)
    for p in model.parameters():
        other=p.detach().clone();dist.broadcast(other,0);maxdiff=torch.maximum(maxdiff,(other-p).abs().max())
    change=float((next(model.parameters())-before).abs().max());assert float(maxdiff)==0 and change>0
    assert int(next(iter(opt.state.values()))['step'])==2
    row={'rank':rank,'status':'passed','real_model_parameters':sum(p.numel() for p in model.parameters()),'full_and_short_batch':values,'optimizer_updates':2,'max_parameter_difference':float(maxdiff),'parameter_change':change,'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'patient_pixels_access':False}
    gathered=[None,None];dist.all_gather_object(gathered,row)
    if rank==0:write(R/'GPU_SMOKE.json',{'utc':now(),'status':'passed','ranks':gathered});print(json.dumps(gathered),flush=True)
    clean_exit()


def evaluation(model,loader,device,epoch):
    result=validate(model,loader,device)
    assert result['cases']==108 and result['subjects']==50
    assert all(np.isfinite(r['locator_dsc']) for r in result['per_case'])
    result.update(epoch=epoch,created_utc=now(),role='reused fixed validation108/50',test_access=False)
    return result


def train():
    rank,device=setup();verify();disk()
    gate=json.loads((R/'EXECUTION_GATE.json').read_text())
    for name,h in gate['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('execution gate changed: '+name)
    data=json.loads((R/'DATA_AUDIT.json').read_text())
    train_data=BrainCropDataset(TRAIN_CACHE,expected_split='train',case_ids=data['train_case_ids'])
    val_data=BrainCropDataset(VAL_CACHE,expected_split='validation',case_ids=data['validation_case_ids'])
    assert len(train_data)==972 and len(val_data)==108
    if rank==0:
        if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('refuse nonempty output directory')
        OUT.mkdir(parents=True)
        write(R/'TRAIN_PREACCESS.json',{'utc':now(),'mode':'fresh_random_init_single_arm','seed':SEED,'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'freeze_sha256':sha(R/'FREEZE.json'),'data_audit_sha256':sha(R/'DATA_AUDIT.json'),'prior_test_informed_development':True,'test_pixels_access':False,'old_checkpoint_loaded':False})
    dist.barrier();model=model_for(device);ddp=DDP(Forward(model),device_ids=[device.index]);opt=optimizer_for(model);criterion=loss_for()
    if rank==0:
        save(OUT/'initial.pth',{'model_state_dict':model.state_dict(),'seed':SEED,'epoch':0,'architecture':'plain MONAI SegResNet4700897'})
        reference=torch.load(ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917/initial.pth',map_location='cpu',weights_only=False)['model_state_dict']
        assert all(torch.equal(v.detach().cpu(),reference[k]) for k,v in model.state_dict().items())
        write(R/'INITIAL_PARITY.json',{'utc':now(),'same_random_initial_state_as_N4_baseline':True,'learned_checkpoint_used_for_initialization':False})
        del reference
        write(R/'INITIALIZATION.json',{'utc':now(),'checkpoint_sha256':sha(OUT/'initial.pth'),'seed':SEED,'parameter_count':4_700_897,'teacher':False,'distillation':False,'old_checkpoint_loaded':False})
    val_loader=DataLoader(val_data,batch_size=2,shuffle=False,num_workers=2,pin_memory=True) if rank==0 else None
    started=time.monotonic();best=-1.;best_epoch=0;torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,201):
        disk();tick=time.monotonic();seed_all(SEED+epoch*10+rank);model.train()
        lr=learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200)
        for group in opt.param_groups:group['lr']=lr
        loader=DataLoader(train_data,batch_sampler=list(batches_for_epoch(epoch,rank)),num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(SEED+epoch*100+rank))
        samples=0;steps=0;values=[];case_ids=[]
        for number,batch in enumerate(loader,1):
            image=batch['image'].to(device,non_blocking=True);label=batch['label'].to(device,non_blocking=True)
            value=step(ddp,opt,criterion,image,label);n=image.shape[0]*2;samples+=n;steps+=1;values.append((n,value));case_ids.extend(batch['case_id'])
            if rank==0 and (number==1 or number%30==0 or number==122):print(json.dumps({'utc':now(),'epoch':epoch,'step':number,'steps_per_epoch':122,'global_samples':samples,'learning_rate':lr,**value}),flush=True)
        gathered=[None,None];dist.all_gather_object(gathered,case_ids)
        ids=gathered[0]+gathered[1]
        if samples!=972 or steps!=122 or len(ids)!=972 or len(set(ids))!=972 or set(ids)!=set(data['train_case_ids']):raise RuntimeError('full972 unique coverage gate failed')
        log={'utc':now(),'epoch':epoch,'samples':samples,'unique_scans':len(set(ids)),'optimizer_updates':steps,'cumulative_optimizer_updates':epoch*122,'learning_rate':lr,'train':{k:sum(n*v[k] for n,v in values)/972 for k in ['loss','train_dice','grad_norm_before_clip']},'augmentation_fallbacks':sum(v['augmentation_fallbacks'] for _,v in values),'training_seconds':time.monotonic()-tick,'peak_allocated_bytes_per_gpu':torch.cuda.max_memory_allocated()}
        if rank==0:
            if epoch==1:write(R/'VALIDATION_PREACCESS.json',{'utc':now(),'role':'fixed reused108/50','selection':'scan Dice max; every epoch; earliest tie','threshold_raw_logit':0,'no_tta':True,'no_postprocessing':True,'no_GT_location_or_clicks':True,'freeze_sha256':sha(R/'FREEZE.json'),'test_access':False})
            result=evaluation(model,val_loader,device,epoch);report=OUT/f'validation_e{epoch:03d}.json';write(report,result)
            log['validation']={k:v for k,v in result.items() if k!='per_case'}
            if result['locator_dsc_mean']>best:
                best=result['locator_dsc_mean'];best_epoch=epoch
                save(OUT/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'method':'pure SegResNet original972 without extra N4','selection':'fixed validation scan Dice','freeze_sha256':sha(R/'FREEZE.json')})
                write(OUT/'BEST.json',{'epoch':epoch,'scan_dice':best,'subject_dice':result['locator_subject_dsc_mean'],'point_hit':result['automatic_point_hit_rate'],'report':report.name,'report_sha256':sha(report),'checkpoint_sha256':sha(OUT/'best.pth')})
            if epoch%10==0:save(OUT/f'epoch{epoch:03d}.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'method':'pure SegResNet original972 without extra N4','freeze_sha256':sha(R/'FREEZE.json')})
        dist.barrier()
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device)};rngs=[None,None];dist.all_gather_object(rngs,rng)
        if rank==0:
            save(OUT/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'learning_rate':lr,'schedule':'constant100 then cosine to1e-5 at200','seed':SEED,'rng_by_rank':rngs,'cumulative_optimizer_updates':epoch*122,'freeze_sha256':sha(R/'FREEZE.json')})
            log.update(elapsed_seconds=time.monotonic()-started,checkpoint_sha256=sha(OUT/'latest.pth'),validation_report_sha256=sha(report),selection=json.loads((OUT/'BEST.json').read_text()))
            with (OUT/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            write(OUT/'STATUS.json',{'status':'running','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*122,'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'test_pixels_access':False});print(json.dumps(log),flush=True)
        budget=torch.tensor(int(time.monotonic()-started>=8*3600),device=device);dist.all_reduce(budget,op=dist.ReduceOp.MAX)
        if bool(budget):break
    if rank==0:
        if epoch==200:
            write(R/'RELOAD_PREACCESS.json',{'utc':now(),'role':'same fixed validation108','purpose':'strict reload consistency only; no selection/adaptation','test_access':False})
            reloads=[];selected=json.loads((OUT/'BEST.json').read_text())
            for name,ep in [('latest.pth',200),('best.pth',selected['epoch'])]:
                state=torch.load(OUT/name,map_location='cpu',weights_only=False);assert state['epoch']==ep;model.load_state_dict(state['model_state_dict'],strict=True);del state
                actual=evaluation(model,val_loader,device,ep);expected=json.loads((OUT/f'validation_e{ep:03d}.json').read_text())
                a=actual['per_case'];b=expected['per_case'];assert [x['case_id'] for x in a]==[x['case_id'] for x in b]
                error=max(abs(x['locator_dsc']-y['locator_dsc']) for x,y in zip(a,b));assert error<=1e-7 and all(x['automatic_point']==y['automatic_point'] for x,y in zip(a,b))
                reloads.append({'checkpoint':name,'epoch':ep,'sha256':sha(OUT/name),'max_dice_error':error,'points_exact':True,'strict_load':True})
            write(R/'RELOAD_CHECK.json',{'utc':now(),'status':'passed','checks':reloads})
        write(OUT/'STATUS.json',{'status':'completed' if epoch==200 else 'incomplete','last_complete_epoch':epoch,'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*122,'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'reason':'epoch_budget' if epoch==200 else 'wall_budget','test_pixels_access':False})
    clean_exit()

if __name__=='__main__':
    try:
        if sys.argv[1]=='smoke':smoke()
        elif sys.argv[1]=='train':train()
        else:raise ValueError('smoke/train mode required')
    except BaseException as exc:write(R/f"FAILURE_{sys.argv[1]}_rank{os.environ.get('RANK','none')}.json",{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
