"""Single-node DDP smoke/diagnosis/finetuning with new optimizer accounting."""
import argparse
import datetime
import fcntl
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset
from monai.losses import DiceCELoss
from scripts.race_v2_20261029.common import *
from scripts.race_v2_20261029.data import RaceDataset,augment_batch
from scripts.race_v2_20261029.model import RaceBenefitSeg
from scripts.race_v2_20261029.benefit import objective
from scripts.race_v2_20261029.benefit import gain_calibration
from scripts import segresnet_pretrained_native_common as native

def initialize():
    rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=5))
    assert dist.get_world_size()==2
    torch.set_num_threads(8); native.seed_all(SEED+rank); native.fast_cudnn()
    assert sha(SOURCE)==SOURCE_SHA
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert source['epoch']==164 and 'optimizer_state_dict' not in source
    model=RaceBenefitSeg(width=32,dropout=.1,dim=16,enabled=True)
    model.load_state_dict(source['model_state_dict'],strict=True)
    assert all(torch.equal(v,source['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
    model=model.cuda(); del source
    return rank,model

def optimizer(model,lr=2e-5):
    return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5,fused=True)

def weight_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def check_sync(model):
    hashes=[None,None];dist.all_gather_object(hashes,weight_hash(model))
    assert hashes[0]==hashes[1]
    return hashes[0]

def checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(state,temporary);os.replace(temporary,path)

def finish(rank,mode):
    dist.barrier();write(R/f'{mode}_EXIT_RANK{rank}.json',dict(at=now(),compute_complete=True))
    def stop():sys.stdout.flush();sys.stderr.flush();os._exit(0)
    timer=threading.Timer(10,stop);timer.daemon=True;timer.start()
    dist.destroy_process_group();stop()

def prepared_batches(loader,epoch,rank):
    def local(batch,iteration):
        augmented=augment_batch(batch,SEED+epoch*100000+iteration*10)
        n=len(augmented['image']); assert n in (4,8); start=rank*n//2
        return {k:v[start:start+n//2] for k,v in augmented.items()}
    with ThreadPoolExecutor(max_workers=1) as pool:
        source=iter(loader);batch=next(source,None)
        if batch is None:return
        pending=pool.submit(local,batch,1); iteration=1
        while True:
            values=pending.result(); n=len(batch['image']); start=rank*n//2
            ids=batch.get('case_id',[])[start:start+n//2]
            following=next(source,None)
            future=pool.submit(local,following,iteration+1) if following is not None else None
            yield iteration,n,ids,values
            if following is None:break
            pending=future;batch=following;iteration+=1

def update(model,bare,opt,fn,local,iteration,coefficient=.05,margin=.002):
    n=len(local['image']);opt.zero_grad(set_to_none=True);records=[]
    for start in range(0,n,2):
        items={k:v[start:start+2].cuda(non_blocking=True) for k,v in local.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(items['image'],items['reflection'],items['valid'],corruption=iteration%2==0)
            loss,metrics=objective(bare,output,items['label'],items['classes'],items['valid'],fn,coefficient,margin)
        if not bool(loss.isfinite()):raise RuntimeError('nonfinite objective')
        (loss*len(items['image'])/n).backward()
        records.append((len(items['image']),dict(loss=float(loss.detach()),**metrics)))
    norm=torch.nn.utils.clip_grad_norm_(bare.parameters(),5.,error_if_nonfinite=True)
    assert bool(norm>0)
    grads={name:float(module.weight.grad.float().norm()) for name,module in
           [('gate',bare.gate_head),('correction',bare.correction),('query',bare.reference.query)]}
    grads['strength']=float(bare.strength.grad)
    opt.step()
    keys=[k for k in records[0][1] if not k.startswith('mismatch_')]
    values={key:sum(size*value[key] for size,value in records)/n for key in keys}
    values.update(gradient_norm=float(norm),**{'gradient_'+k:v for k,v in grads.items()})
    tensor=torch.tensor(list(values.values()),dtype=torch.float64,device='cuda');dist.all_reduce(tensor)
    return dict(zip(values,(tensor/2).cpu().tolist()))

def smoke(rank,bare):
    # Complete native canvas, real reference code, four global8 + one tail4.
    frozen=read(R/'SMOKE_SOURCE_FREEZE_V3.json')
    assert numerical_functions()==frozen['numerical_function_ast_sha256']
    for path,digest in frozen['source_sha256'].items():assert sha(ROOT/path)==digest
    generator=torch.Generator().manual_seed(21)
    image=torch.randn(8,1,*CANVAS,generator=generator)
    target=torch.zeros_like(image);target[:,:,48:80,80:100,64:80]=1
    classes=target.byte();classes[:,:,85:100,90:110,65:85]=2
    matrix=torch.eye(4).repeat(8,1,1);matrix[:,0,0]=-1;matrix[:,0,3]=CANVAS[0]-8
    batch=dict(image=image,label=target,classes=classes,valid=torch.ones_like(image),reflection=matrix)
    from sammed3d.race_seg import RaceSeg
    historical=RaceSeg(width=32,dropout=.1,dim=16,enabled=True).cuda().eval()
    historical.load_state_dict(bare.state_dict(),strict=True);bare.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        old=historical(image[:1].cuda(),matrix[:1].cuda(),torch.ones_like(image[:1]).cuda(),corruption=True)
        new=bare(image[:1].cuda(),matrix[:1].cuda(),torch.ones_like(image[:1]).cuda(),corruption=True)
    for name in ('segmentation_logits','base_logits','gate','bad_logits','bad_gate'):
        assert torch.equal(getattr(old,name),getattr(new,name)),name+' bf16 deployment mismatch'
    del historical,old,new;bare.train();torch.cuda.empty_cache()
    model=DDP(bare,device_ids=[rank],broadcast_buffers=False)
    opt=optimizer(bare);fn=DiceCELoss(sigmoid=True,lambda_dice=1.,lambda_ce=.1)
    batches=[batch]*4+[{k:v[:4] for k,v in batch.items()}]
    times=[];all_metrics=[];torch.cuda.reset_peak_memory_stats()
    for iteration,n,ids,local in prepared_batches(batches,1,rank):
        native.seed_all(SEED+rank+iteration*10);torch.cuda.synchronize();started=time.monotonic()
        metrics=update(model,bare,opt,fn,local,iteration)
        torch.cuda.synchronize();times.append(time.monotonic()-started);all_metrics.append(metrics)
        assert all(int(s['step'])==iteration for s in opt.state.values())
    digest=check_sync(bare);path=TMP/f'smoke_rank{rank}.pth'
    checkpoint(path,dict(model_state_dict=bare.state_dict(),optimizer_state_dict=opt.state_dict()))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    cpu=RaceBenefitSeg(width=32,dropout=.1,dim=16,enabled=True)
    cpu.load_state_dict(saved['model_state_dict'],strict=True)
    assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in bare.state_dict().items())
    assert all(int(s['step'])==5 for s in saved['optimizer_state_dict']['state'].values())
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    records=[None,None];dist.all_gather_object(records,dict(rank=rank,seconds_per_update=sum(times[1:4])/3,
        update_seconds=times,peak_reserved_gib=peak,new_updates=5,tail4=True,strict_cpu_reload=True,
        source_checkpoint_sha256=SOURCE_SHA,parameter_sha256=digest,metrics=all_metrics))
    path.unlink()
    if rank==0:write(R/'SMOKE_DDP.json',dict(at=now(),status='passed',patient_forwards=0,effective_batch=8,
       microbatch_per_gpu=2,optimizer_new=True,bf16_deployment_exact=True,ranks=records,
       numerical_function_ast_sha256=numerical_functions(),smoke_source_freeze_sha256=sha(R/'SMOKE_SOURCE_FREEZE_V3.json'),
       numerical_source_sha256={p:sha(ROOT/p) for p in ('scripts/race_v2_20261029/model.py','scripts/race_v2_20261029/benefit.py')}))
    finish(rank,'SMOKE_DDP')

def distribution(value,mask):
    values=value.float()[mask.bool()]
    if not len(values):return dict(n=0,mean=0.,p50=0.,p90=0.,p99=0.,maximum=0.)
    q=torch.quantile(values,torch.tensor([.5,.9,.99],device=values.device))
    return dict(n=len(values),mean=float(values.mean()),p50=float(q[0]),p90=float(q[1]),p99=float(q[2]),maximum=float(values.max()))

@native.deterministic_validation
def diagnose(rank,bare):
    assert read(R/'DATA_validation.json')['historical_native_sha_exact']
    bare.eval();fn=DiceCELoss(sigmoid=True,lambda_dice=1.,lambda_ce=.1)
    data=RaceDataset('validation')
    loader=DataLoader(Subset(data,list(range(rank,len(data),2))),batch_size=1,num_workers=2,
           multiprocessing_context='spawn',persistent_workers=True,pin_memory=True)
    cases=[];started=time.monotonic();source_hash=weight_hash(bare)
    for completed,batch in enumerate(loader,1):
        deadline();disk();cid=batch['case_id'][0]
        image=batch['image'].cuda();target=batch['label'].cuda();valid=batch['valid'].cuda();matrix=batch['reflection'].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=bare(image,matrix,valid,corruption=True)
            segmentation=fn(out.segmentation_logits,target)
            calibration,counts=gain_calibration(out.gate_logits,out.base_logits,out.proposal_logits,target,valid,out.has_reference)
            _,mismatch_counts=gain_calibration(out.bad_gate_logits,out.base_logits,out.bad_proposal_logits,target,valid,out.has_reference)
        parameters=[bare.gate_head.weight,bare.correction.weight,bare.strength]
        gradients=torch.autograd.grad(segmentation,parameters,retain_graph=True)
        calibration_grad=torch.autograd.grad(calibration,bare.gate_head.weight)[0]
        with torch.no_grad():
            lesion=target>.5
            dilation=torch.nn.functional.max_pool3d(target,7,stride=1,padding=3)>.5
            erosion=-torch.nn.functional.max_pool3d(-target,7,stride=1,padding=3)>.5
            boundary=dilation&~erosion
            regions=dict(lesion=lesion,boundary3mm=boundary,brain=valid>.5)
            gate_regions={name:torch.nn.functional.adaptive_max_pool3d(mask.float(),out.gate.shape[-3:])>.5 for name,mask in regions.items()}
            modes={}
            for mode,logits,gate in [('normal',out.segmentation_logits,out.gate),('base',out.base_logits,out.gate*0),('mismatch',out.bad_logits,out.bad_gate)]:
                metrics=native.case_metrics(logits[0].float(),target[0],data.geometry[cid])
                modes[mode]=dict(**metrics,
                     gate={name:distribution(gate,mask&out.has_reference) for name,mask in gate_regions.items()},
                     logit_correction={name:distribution((logits-out.base_logits).abs(),mask) for name,mask in regions.items()},
                     probability_correction={name:distribution((logits.float().sigmoid()-out.base_logits.float().sigmoid()).abs(),mask) for name,mask in regions.items()})
            cases.append(dict(case_id=cid,subject_id=batch['subject_id'][0],modes=modes,
                 gate_all=distribution(out.gate,torch.ones_like(out.gate,dtype=torch.bool)),
                 gradients=dict(gate_head_segmentation=float(gradients[0].norm()),correction_segmentation=float(gradients[1].norm()),
                       strength_segmentation=float(gradients[2]),gate_head_calibration=float(calibration_grad.norm())),
                 proposal_targets=counts,mismatch_proposal_targets=mismatch_counts))
        del out,segmentation,calibration,gradients,calibration_grad,image,target,valid,matrix
        write(R/f'DIAGNOSIS_JOURNAL_RANK{rank}.json',dict(at=now(),completed=completed,total=54,per_case=cases))
    assert weight_hash(bare)==source_hash
    gathered=[None,None];dist.all_gather_object(gathered,cases)
    if rank==0:
        cases=sorted([case for group in gathered for case in group],key=lambda c:c['case_id'])
        assert len(cases)==len({case['case_id'] for case in cases})==108
        metrics={}
        for mode in ('normal','base','mismatch'):
            rows_=[dict(case_id=c['case_id'],subject_id=c['subject_id'],**{k:v for k,v in c['modes'][mode].items() if k not in ('gate','logit_correction','probability_correction')}) for c in cases]
            metrics[mode]=native.summary_ci(rows_);metrics[mode]['point_hits']=sum(c['automatic_point_hits_gt'] for c in rows_)
        write(R/'DIAGNOSIS108.json',dict(at=now(),status='complete',source_checkpoint_sha256=SOURCE_SHA,
             parameters_unchanged=True,metrics=metrics,per_case=cases,seconds=time.monotonic()-started,
             network_case_forwards=108,reference_branches=216,training_updates=0,gradient_diagnostics_only=True,
             grid='native1mm 160x208x160 crop excludes padding',threshold=.5,test_pixels=False))
    finish(rank,'DIAGNOSIS')

@torch.inference_mode()
@native.deterministic_validation
def evaluate(bare,data,loader,epoch,rank):
    bare.eval();cases=[];gates=[]
    for batch in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=bare(batch['image'].cuda(),batch['reflection'].cuda(),batch['valid'].cuda())
        cid=batch['case_id'][0]
        item=native.case_metrics(output.segmentation_logits[0].float(),batch['label'][0].cuda(),data.geometry[cid])
        item.update(case_id=cid,subject_id=batch['subject_id'][0]);cases.append(item)
        gates.append(float(output.gate.float().mean()))
    gathered=[None,None];dist.all_gather_object(gathered,(cases,gates))
    report=None
    if rank==0:
        cases=sorted([c for part,_ in gathered for c in part],key=lambda c:c['case_id'])
        assert len(cases)==len({c['case_id'] for c in cases})==108
        metrics=native.summary_ci(cases);metrics['point_hits']=sum(c['automatic_point_hits_gt'] for c in cases)
        report=dict(at=now(),new_epoch=epoch,metrics=metrics,per_case=cases,
             gate_mean=sum(sum(g) for _,g in gathered)/108,grid='native1mm160x208x160 crop excludes padding',
             threshold=.5,test_pixels=False)
    result=[report];dist.broadcast_object_list(result,src=0);bare.train();return result[0]

def strict_reload(path,bare,updates,full=True):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    cpu=RaceBenefitSeg(width=32,dropout=.1,dim=16,enabled=True)
    cpu.load_state_dict(saved['model_state_dict'],strict=True)
    assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in bare.state_dict().items())
    if full: assert all(int(s['step'])==updates for s in saved['optimizer_state_dict']['state'].values())
    return dict(at=now(),status='passed',sha256=sha(path),new_updates=updates,strict_cpu_reload=True,
                optimizer_checked=full)

def train(rank,bare,resume=False):
    verify_sources();deadline(required=True);disk()
    from scripts.race_v2_stage_20261029.governance import guard
    guard(1154259,training=True)
    assert read(R/'DATA_READY.json')['status']=='passed'
    smoke=read(R/'SMOKE_DDP.json');assert smoke['status']=='passed' and smoke['bf16_deployment_exact']
    for path,digest in smoke['numerical_source_sha256'].items():assert sha(ROOT/path)==digest
    assert numerical_functions()==smoke['numerical_function_ast_sha256']
    budget=read(R/'BUDGET_DECISION.json');assert budget['status']=='passed'
    epochs=budget['new_epochs'];recipe=read(R/'RECIPE.json');freeze_sha=sha(R/'EXECUTION_FREEZE.json')
    target=TMP/'run';target.mkdir(parents=True,exist_ok=True)
    opt=optimizer(bare,recipe['learning_rate'])
    first=1;updates=0;elapsed_before=0.;baseline=read(R/'DIAGNOSIS108.json')
    best=baseline['metrics']['normal']['scan_dice'];best_epoch=0
    if resume:
        saved=torch.load(target/'latest.pth',map_location='cpu',weights_only=False)
        assert saved['freeze_sha256']==freeze_sha and saved['run_id']==recipe['run_id']
        assert saved['source_epoch']==164 and saved['source_sha256']==SOURCE_SHA
        assert 1<=saved['new_epoch']<=epochs and saved['new_updates']==saved['new_epoch']*122
        assert 0<=saved['best_new_epoch']<=saved['new_epoch'] and np.isfinite(saved['best_scan'])
        bare.load_state_dict(saved['model_state_dict'],strict=True);opt.load_state_dict(saved['optimizer_state_dict'])
        first=saved['new_epoch']+1;updates=saved['new_updates'];best=saved['best_scan'];best_epoch=saved['best_new_epoch'];elapsed_before=saved['elapsed_seconds']
        assert updates==(first-1)*122 and all(int(s['step'])==updates for s in opt.state.values())
        write(R/f'RESTORE_RANK{rank}_{first}.json',dict(at=now(),new_epoch_next=first,new_updates_retained=updates,
              source_checkpoint_sha256=sha(target/'latest.pth'),optimizer_restored=True))
        del saved
    elif rank==0:
        with (R/'TRAIN_PREACCESS.json').open('x') as handle:
            json.dump(dict(at=now(),run_id=recipe['run_id'],source_epoch=164,source_sha256=SOURCE_SHA,
              source_optimizer_missing=True,new_optimizer=True,new_epochs=epochs,new_update_budget=epochs*122,
              train=972,validation=108,test_pixels=False,freeze_sha256=freeze_sha),handle)
        checkpoint(target/'best.pth',dict(model_state_dict=bare.state_dict(),new_epoch=0,new_updates=0,
             source_epoch=164,source_sha256=SOURCE_SHA,run_id=recipe['run_id'],freeze_sha256=freeze_sha))
    dist.barrier();model=DDP(bare,device_ids=[rank],broadcast_buffers=False);check_sync(bare)
    if first>epochs:
        # A publication/I/O failure after final latest must not repeat updates.
        if rank==0:
            final_report=read(R/f'validation_e{epochs:03}.json')
            status=dict(at=now(),state='complete',new_epoch=epochs,new_updates=updates,
                 target_new_epochs=epochs,best_new_epoch=best_epoch,best_scan=best,metrics=final_report['metrics'],
                 elapsed_seconds=elapsed_before,effective_batch=8,world_size=2,microbatch_per_gpu=2,
                 optimizer_new_at_stage_start=True,test_pixels=False,archive_only_recovery=True)
            write(R/'STATUS.json',status);write(R/'DONE.json',status)
        finish(rank,'TRAIN');return
    fn=DiceCELoss(sigmoid=True,lambda_dice=1.,lambda_ce=.1)
    training=RaceDataset('train');validation=RaceDataset('validation');batches=[]
    loader=DataLoader(training,batch_sampler=batches,num_workers=3,pin_memory=True,prefetch_factor=2,
           multiprocessing_context='spawn',persistent_workers=True)
    validation_loader=DataLoader(Subset(validation,list(range(rank,len(validation),2))),batch_size=1,
           num_workers=2,pin_memory=True,multiprocessing_context='spawn',persistent_workers=True)
    begun=time.monotonic();bare.train()
    for epoch in range(first,epochs+1):
        guard(1154259,training=True);deadline(required=True);started=time.monotonic();native.seed_all(SEED+epoch*10+rank)
        progress=max(0.,(epoch-recipe['constant_epochs'])/max(1,epochs-recipe['constant_epochs']))
        lr=recipe['min_learning_rate']+.5*(recipe['learning_rate']-recipe['min_learning_rate'])*(1+np.cos(np.pi*progress))
        for group in opt.param_groups:group['lr']=float(lr)
        order=torch.randperm(972,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        batches[:]=[order[i:i+8] for i in range(0,972,8)];seen=[];records=[]
        for iteration,n,ids,local in prepared_batches(loader,epoch,rank):
            deadline(required=True);seen.extend(ids)
            metrics=update(model,bare,opt,fn,local,iteration,recipe['calibration_coefficient'],recipe['gain_margin'])
            updates+=1;records.append((n,metrics))
            if rank==0 and (iteration==1 or iteration%20==0 or iteration==122):
                write(R/'HEARTBEAT.json',dict(at=now(),phase='training',new_epoch=epoch,iteration=iteration,
                     new_updates=updates,target_new_epochs=epochs,effective_batch=8,**metrics))
        all_seen=[None,None];dist.all_gather_object(all_seen,seen);ids=[i for group in all_seen for i in group]
        assert len(ids)==len(set(ids))==972 and set(ids)==set(training.case_ids)
        assert updates==epoch*122 and all(int(s['step'])==updates for s in opt.state.values())
        training_seconds=time.monotonic()-started
        if rank==0:write(R/'HEARTBEAT.json',dict(at=now(),phase='validation',new_epoch=epoch,new_updates=updates))
        report=evaluate(bare,validation,validation_loader,epoch,rank)
        if report['metrics']['scan_dice']>best:
            best=report['metrics']['scan_dice'];best_epoch=epoch
            if rank==0:
                checkpoint(target/'best.pth',dict(model_state_dict=bare.state_dict(),new_epoch=epoch,new_updates=updates,
                    source_epoch=164,source_sha256=SOURCE_SHA,run_id=recipe['run_id'],freeze_sha256=freeze_sha))
                write(R/'BEST.json',dict(at=now(),new_epoch=epoch,new_updates=updates,metrics=report['metrics'],
                    checkpoint_path=str(target/'best.pth'),checkpoint_sha256=sha(target/'best.pth')))
        if rank==0:
            write(R/f'validation_e{epoch:03}.json',report)
            state=dict(model_state_dict=bare.state_dict(),optimizer_state_dict=opt.state_dict(),new_epoch=epoch,
                 new_updates=updates,best_scan=best,best_new_epoch=best_epoch,source_epoch=164,source_sha256=SOURCE_SHA,
                 run_id=recipe['run_id'],freeze_sha256=freeze_sha,elapsed_seconds=elapsed_before+time.monotonic()-begun,
                 recovery_convention='full completed epoch; each next epoch reseeds CPU/CUDA per rank and reconstructs deterministic global972 order',
                 seed=SEED,world_size=2,next_epoch_global_order_seed=SEED+epoch+1,
                 next_epoch_rank_rng_seeds=[SEED+(epoch+1)*10+r for r in (0,1)])
            checkpoint(target/'latest.pth',state)
            if epoch==first or epoch==epochs or epoch%10==0:
                write(R/f'RELOAD_e{epoch:03}.json',strict_reload(target/'latest.pth',bare,updates))
            status=dict(at=now(),state='complete' if epoch==epochs else 'training',new_epoch=epoch,new_updates=updates,
                 target_new_epochs=epochs,best_new_epoch=best_epoch,best_scan=best,metrics=report['metrics'],
                 training_seconds=training_seconds,epoch_seconds=time.monotonic()-started,
                 elapsed_seconds=state['elapsed_seconds'],learning_rate=float(lr),effective_batch=8,world_size=2,
                 microbatch_per_gpu=2,optimizer_new_at_stage_start=True,test_pixels=False,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                 train={key:sum(n*value[key] for n,value in records)/972 for key in records[0][1]})
            write(R/'STATUS.json',status)
            with (R/'HISTORY.jsonl').open('a') as handle:handle.write(json.dumps(status)+'\n')
        dist.barrier();digest=check_sync(bare)
        if rank==0 and epoch==first:
            write(R/'STARTUP_VERIFIED.json',dict(at=now(),status='passed',new_epoch=epoch,
                 new_updates=updates,real_train_cases=972,real_validation_cases=108,
                 effective_batch=8,updates_per_epoch=122,optimizer_new_at_stage_start=True,
                 source_checkpoint_sha256=SOURCE_SHA,strict_cpu_reload=read(R/f'RELOAD_e{epoch:03}.json'),
                 rank_parameter_sha256=digest,epoch_seconds=time.monotonic()-started,
                 metrics=report['metrics'],early_score_not_final_result=True,test_pixels=False))
    if rank==0:
        write(R/'DONE.json',status)
    finish(rank,'TRAIN')

def main(mode):
    disk();deadline();rank,bare=initialize()
    if mode=='smoke':smoke(rank,bare)
    elif mode=='diagnose':diagnose(rank,bare)
    elif mode=='train':train(rank,bare,os.environ.get('AFLOW_RESUME')=='1')
    else:raise ValueError('mode not yet implemented')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','diagnose','train'));args=parser.parse_args()
    try:main(args.mode)
    except BaseException as error:
        traceback.print_exc();write(R/f'{args.mode}_FAILURE_{os.environ.get("LOCAL_RANK","unknown")}_{time.time_ns()}.json',dict(at=now(),error=repr(error)))
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
