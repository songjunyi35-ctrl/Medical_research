"""Sequential two-A100 training: semantic_regions, pretrain150, finetune_pretrained200.
Pretrain uses its fixed final checkpoint, no validation selection. No test access here.
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
from scripts.segresnet_two_ideas_common import *

class Forward(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):
        return self.model.network(x) if isinstance(self.model,SemanticSegResNet) else self.model(x).segmentation_logits


def setup(seed):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60));assert dist.get_world_size()==WORLD
    seed_all(seed);return rank,torch.device(f'cuda:{local}')


def clean_exit():
    dist.barrier();torch.cuda.synchronize()
    def force():print(json.dumps({'event':'successful_cleanup_timeout','outputs_committed':True}),flush=True);os._exit(0)
    t=threading.Timer(20,force);t.daemon=True;t.start();dist.destroy_process_group();sys.stdout.flush();os._exit(0)


def optimizer_for(model):return torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
def loss_for():return DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)


def step(ddp,opt,loss_fn,image,label):
    valid=(torch.isfinite(image).all() & torch.isfinite(label).all() & (label.flatten(1).sum(1)>0).all()).to(torch.int32);dist.all_reduce(valid,op=dist.ReduceOp.MIN)
    if not bool(valid):raise RuntimeError('nonfinite image/label or empty source foreground')
    image,label,fallbacks=safe_augment(image,label);opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=ddp(image);loss=loss_fn(logits,label)
    valid=torch.isfinite(loss).to(torch.int32);dist.all_reduce(valid,op=dist.ReduceOp.MIN)
    if not bool(valid):raise RuntimeError('nonfinite training loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),5.,error_if_nonfinite=True)
    if not bool(norm>0):raise RuntimeError('zero gradient')
    opt.step()
    with torch.no_grad():dice=dice_per_case(wt_logit(logits.detach()) if logits.shape[1]==4 else logits.detach().float(),(label>0).float()).mean()
    values=torch.tensor([float(loss.detach()),float(dice),float(norm),float(fallbacks)],device=image.device,dtype=torch.float64);dist.all_reduce(values);values[:3]/=WORLD
    return dict(zip(['loss','train_dice','grad_norm_before_clip','augmentation_fallbacks'],values.cpu().tolist()))


def smoke(run_name):
    rank,device=setup(BASELINE_SEED);verify();semantic=run_name=='semantic_regions';model=model_for(device,semantic);ddp=DDP(Forward(model),device_ids=[device.index]);opt=optimizer_for(model);criterion=SemanticLoss() if semantic else loss_for()
    seed_all(BASELINE_SEED+rank);x=torch.randn(PER_GPU_BATCH,1,128,128,128,device=device);y=torch.zeros_like(x);y[:,:,36:92,36:92,36:92]=1;x+=y*1.5
    before=next(model.parameters()).detach().clone();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    values=[step(ddp,opt,criterion,x.clone(),y.clone()),step(ddp,opt,criterion,x[:3].clone(),y[:3].clone())]
    maxdiff=torch.zeros((),device=device)
    for p in model.parameters():
        other=p.detach().clone();dist.broadcast(other,0);maxdiff=torch.maximum(maxdiff,(other-p).abs().max())
    change=float((next(model.parameters())-before).abs().max());assert float(maxdiff)==0 and change>0
    # init-from-checkpoint path: strict load of the baseline initial weights must reproduce them exactly
    init=torch.load(BASELINE_RUN/'initial.pth',map_location='cpu',weights_only=False)['model_state_dict'];model.initialize_from_binary(init) if semantic else model.load_state_dict(init,strict=True)
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in init.items() if not (semantic and 'conv_final.2.conv' in k))
    row={'rank':rank,'status':'passed','real_model_parameters':sum(p.numel() for p in model.parameters()),'steps':values,'max_parameter_difference':float(maxdiff),'parameter_change':change,'strict_init_load':True,'seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'patient_pixels_access':False}
    g=[None,None];dist.all_gather_object(g,row)
    if rank==0:write(R/f'GPU_SMOKE_{run_name}.json',{'utc':now(),'status':'passed','ranks':g})
    clean_exit()


def evaluation(model,loader,device,epoch,role):
    result=validate(model,loader,device);assert result['cases']==108 and result['subjects']==50 and all(np.isfinite(r['locator_dsc']) for r in result['per_case'])
    result.update(epoch=epoch,created_utc=now(),role=role,test_access=False);return result


def train(run_name):
    mode,init_kind,seed=next((m if m=='pretrain' else 'finetune',k,s) for m,k,s in RUNS if m==run_name)
    rank,device=setup(seed);verify();disk();out=OUT/run_name;total,const=schedule(mode)
    gate=json.loads((R/'EXECUTION_GATE.json').read_text())
    for name,h in gate['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('execution gate changed: '+name)
    audit=json.loads((R/'DATA_AUDIT.json').read_text())
    if mode=='pretrain':
        receipt=json.loads((R/'PRETRAIN_CACHE_RECEIPT.json').read_text());assert receipt['status']=='passed'
        assert sha(PRETRAIN_CACHE/'index.json')==receipt['index_sha256']
        assert sha(PRETRAIN_CACHE/'images.f32')==receipt['images_sha256'] and sha(PRETRAIN_CACHE/'labels.u8')==receipt['labels_sha256']
        train_data=BrainCropDataset(PRETRAIN_CACHE,expected_split=PRETRAIN_SPLIT);n=334;expected_ids=set(audit['pretrain_case_ids'])
    else:train_data=ClassDataset(audit['train_case_ids']) if init_kind=='semantic' else BrainCropDataset(TRAIN_CACHE,expected_split='train',case_ids=audit['train_case_ids']);n=972;expected_ids=set(audit['train_case_ids'])
    val_data=BrainCropDataset(VAL_CACHE,expected_split='validation',case_ids=audit['validation_case_ids']);assert len(train_data)==n and len(val_data)==108
    steps=steps_per_epoch(n);resume=(out/'latest.pth').exists()
    if rank==0:
        if out.exists() and any(out.iterdir()) and not resume:raise RuntimeError('refuse nonempty output directory without latest.pth')
        out.mkdir(parents=True,exist_ok=True)
        if not resume:write(R/f'TRAIN_PREACCESS_{run_name}.json',{'utc':now(),'run':run_name,'mode':mode,'init':init_kind or 'random','seed':seed,'train_cases':n,'epochs':total,'constant_epochs':const,'steps_per_epoch':steps,'validation_scans':108,'freeze_sha256':sha(R/'FREEZE.json'),'data_audit_sha256':sha(R/'DATA_AUDIT.json'),'prior_test_informed_development':True,'test_pixels_access':False,'validation_role':'not evaluated during pretraining' if mode=='pretrain' else 'every epoch; best by scan Dice'})
    dist.barrier();model=model_for(device,init_kind=='semantic')
    if not resume:
        if init_kind=='pretrained':
            src=OUT/'pretrain'/'final.pth';state=torch.load(src,map_location='cpu',weights_only=False);assert state['epoch']==PRETRAIN_EPOCHS;model.load_state_dict(state['model_state_dict'],strict=True);init_receipt={'init':'pretrained','source':str(src.relative_to(ROOT)),'source_sha256':sha(src),'source_epoch':PRETRAIN_EPOCHS};del state
        elif init_kind=='semantic':
            src=BASELINE_RUN/'initial.pth';state=torch.load(src,map_location='cpu',weights_only=False);assert state['epoch']==0;model.initialize_from_binary(state['model_state_dict']);init_receipt={'init':'binary-equivalent semantic expansion','source':str(src.relative_to(ROOT)),'source_sha256':sha(src)};del state
        else:init_receipt={'init':'random','seed':seed}
        for p in model.parameters():dist.broadcast(p.data,0)   # identical init on both ranks regardless of per-rank RNG
        if rank==0:
            save(out/'initial.pth',{'model_state_dict':model.state_dict(),'seed':seed,'epoch':0,**init_receipt})
            write(R/f'INITIALIZATION_{run_name}.json',{'utc':now(),'checkpoint_sha256':sha(out/'initial.pth'),'parameter_count':sum(p.numel() for p in model.parameters()),**init_receipt,'teacher':False,'distillation':False})
    ddp=DDP(Forward(model),device_ids=[device.index]);opt=optimizer_for(model);criterion=SemanticLoss() if init_kind=='semantic' else loss_for()
    start_epoch=1;best=-1.;best_epoch=0;elapsed_before=0.
    if resume:
        state=torch.load(out/'latest.pth',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict'],strict=True);opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch=int(state['epoch'])+1;best=float(state['best_score']);best_epoch=int(state['best_epoch']);elapsed_before=float(state.get('elapsed_seconds',0.));del state
        if rank==0:
            with (R/f'RESUMES_{run_name}.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'resumed_after_epoch':start_epoch-1})+'\n')
    val_loader=DataLoader(val_data,batch_size=2,shuffle=False,num_workers=2,pin_memory=True) if rank==0 else None
    started=time.monotonic()-elapsed_before;torch.cuda.reset_peak_memory_stats();epoch=start_epoch-1
    for epoch in range(start_epoch,total+1):
        disk();tick=time.monotonic();seed_all(seed+epoch*10+rank);model.train();lr=lr_for(mode,epoch)
        for g in opt.param_groups:g['lr']=lr
        loader=DataLoader(train_data,batch_sampler=list(batches_for_epoch(n,seed,epoch,rank)),num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(seed+epoch*100+rank))
        samples=0;n_steps=0;values=[];ids=[]
        for number,batch in enumerate(loader,1):
            image=batch['image'].to(device,non_blocking=True);label=batch['label'].to(device,non_blocking=True)
            value=step(ddp,opt,criterion,image,label);k=image.shape[0]*WORLD;samples+=k;n_steps+=1;values.append((k,value));ids.extend(batch['case_id'])
            if rank==0 and (number==1 or number%30==0 or number==steps):print(json.dumps({'utc':now(),'run':run_name,'epoch':epoch,'step':number,'steps_per_epoch':steps,'learning_rate':lr,**value}),flush=True)
        g=[None,None];dist.all_gather_object(g,ids);all_ids=g[0]+g[1]
        if samples!=n or n_steps!=steps or len(set(all_ids))!=n or set(all_ids)!=expected_ids:raise RuntimeError('unique full coverage gate failed')
        log={'utc':now(),'run':run_name,'mode':mode,'epoch':epoch,'samples':samples,'unique_scans':n,'optimizer_updates':n_steps,'cumulative_optimizer_updates':epoch*steps,'learning_rate':lr,'train':{k:sum(w*v[k] for w,v in values)/n for k in ['loss','train_dice','grad_norm_before_clip']},'augmentation_fallbacks':sum(v['augmentation_fallbacks'] for _,v in values),'training_seconds':time.monotonic()-tick,'peak_allocated_bytes_per_gpu':torch.cuda.max_memory_allocated(),'resumed_this_epoch':bool(resume and epoch==start_epoch)}
        if rank==0:
            if mode=='finetune':
                role='reused fixed validation108/50; every epoch; best by scan Dice' if mode=='finetune' else 'reused fixed validation108/50; final-epoch monitoring only (zero-shot pre-op model on post-op validation); NOT used for selection'
                if epoch==start_epoch or epoch==total:write(R/f'VALIDATION_PREACCESS_{run_name}.json',{'utc':now(),'role':role,'threshold_raw_logit':0,'no_tta':True,'no_postprocessing':True,'test_access':False})
                result=evaluation(model,val_loader,device,epoch,role);report=out/f'validation_e{epoch:03d}.json';write(report,result);log['validation']={k:v for k,v in result.items() if k!='per_case'};log['validation_report_sha256']=sha(report)
                if mode=='finetune' and result['locator_dsc_mean']>best:
                    best=result['locator_dsc_mean'];best_epoch=epoch
                    save(out/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'run':run_name,'selection':'fixed validation scan Dice'})
                    write(out/'BEST.json',{'epoch':epoch,'scan_dice':best,'subject_dice':result['locator_subject_dsc_mean'],'point_hit':result['automatic_point_hit_rate'],'report':report.name,'report_sha256':sha(report),'checkpoint_sha256':sha(out/'best.pth')})
            if epoch%25==0:save(out/f'epoch{epoch:03d}.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'run':run_name})
            if mode=='pretrain' and epoch==total:save(out/'final.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'run':run_name,'role':'pretrained initialization; fixed final epoch, no selection'})
        dist.barrier()
        if rank==0:
            save(out/'latest.pth',{'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_score':best,'learning_rate':lr,'seed':seed,'run':run_name,'cumulative_optimizer_updates':epoch*steps,'elapsed_seconds':time.monotonic()-started})
            log.update(elapsed_seconds=time.monotonic()-started,checkpoint_sha256=sha(out/'latest.pth'));
            if (out/'BEST.json').exists():log['selection']=json.loads((out/'BEST.json').read_text())
            with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            write(out/'STATUS.json',{'status':'running','run':run_name,'last_complete_epoch':epoch,'total_epochs':total,'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*steps,'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'test_pixels_access':False});print(json.dumps(log),flush=True)
        dist.barrier()
    if rank==0:
        final={'status':'completed' if epoch==total else 'incomplete','run':run_name,'last_complete_epoch':epoch,'total_epochs':total,'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*steps,'elapsed_seconds':time.monotonic()-started,'updated_utc':now(),'test_pixels_access':False}
        if mode=='finetune' and epoch==total:
            sel=json.loads((out/'BEST.json').read_text());state=torch.load(out/'best.pth',map_location='cpu',weights_only=False);assert state['epoch']==sel['epoch'];model.load_state_dict(state['model_state_dict'],strict=True)
            actual=evaluation(model,val_loader,device,sel['epoch'],'strict reload consistency only');expected=json.loads((out/sel['report']).read_text())
            err=max(abs(a['locator_dsc']-b['locator_dsc']) for a,b in zip(actual['per_case'],expected['per_case']));final['reload_check']={'max_dice_error':err,'passed':err<=1e-6,'checkpoint_sha256':sha(out/'best.pth')}
            if err>1e-6:final['status']='reload_check_failed'
        write(out/'STATUS.json',final)
    clean_exit()


if __name__=='__main__':
    try:
        if sys.argv[1]=='smoke':smoke(sys.argv[2])
        elif sys.argv[1]=='train':train(sys.argv[2])
        else:raise ValueError('smoke|train <run>')
    except BaseException as exc:write(R/f"FAILURE_{'_'.join(sys.argv[1:3])}_rank{os.environ.get('RANK','none')}_{int(time.time())}.json",{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
