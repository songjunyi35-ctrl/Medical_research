import argparse
import fcntl
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from sammed3d.race_seg import RaceSeg,typed_contrast
from scripts.race_seg_20261028.common import *
from scripts.race_seg_20261028.data import RaceDataset,augment_batch
from scripts import segresnet_pretrained_native_common as native


def setup(arm):
    torch.set_num_threads(2);native.seed_all(SEED);native.fast_cudnn()
    model=RaceSeg(enabled=arm=='race').cuda()
    if arm=='control':
        for name,p in model.named_parameters():
            if not name.startswith('backbone.'): p.requires_grad_(False)
    return model


def make_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-5,fused=True)


def objective(model,out,target,classes,valid,fn,epoch):
    if not model.enabled:
        loss=fn(out.segmentation_logits,target)
        return loss,{'supervised':float(loss.detach()),'contrast':0.,'bad_gate':0.,'fallback':0.,'gate':0.}
    supervised=.5*fn(out.segmentation_logits,target)+.5*fn(out.base_logits,target)
    contrast=typed_contrast(out.embedding,classes,model.prototypes,valid)
    bad_gate=supervised*0;fallback=supervised*0
    if out.bad_gate is not None:
        support=valid[:,:,::8,::8,::8].float()
        bad_gate=(out.bad_gate.float()*support).sum()/support.sum().clamp_min(1)
        fallback=(torch.sigmoid(out.bad_logits.float())-torch.sigmoid(out.base_logits.detach().float())).square().mean()
    ramp=min(1.,max(0.,(epoch-10)/10))
    loss=supervised+.05*contrast+ramp*(.02*bad_gate+.02*fallback)
    return loss,{'supervised':float(supervised.detach()),'contrast':float(contrast.detach()),
                 'bad_gate':float(bad_gate.detach()),'fallback':float(fallback.detach()),'gate':float(out.gate.detach().mean())}


def step(model,opt,fn,batch,epoch,iteration,microbatch,augment=True):
    n=len(batch['image']);opt.zero_grad(set_to_none=True)
    if augment: batch=augment_batch(batch,SEED+epoch*100000+iteration*10)
    rows=[]
    for start in range(0,n,microbatch):
        items={k:v[start:start+microbatch].cuda(non_blocking=True) for k,v in batch.items() if isinstance(v,torch.Tensor)}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(items['image'],items['reflection'],items['valid'],corruption=model.enabled and iteration%2==0)
            loss,metrics=objective(model,out,items['label'],items['classes'],items['valid'],fn,epoch)
        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite objective')
        (loss*(len(items['image'])/n)).backward()
        rows.append((len(items['image']),{'loss':float(loss.detach()),**metrics}))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    if not bool(norm>0): raise RuntimeError('zero parameter gradient')
    strength_grad=float(model.strength.grad) if model.enabled and model.strength.grad is not None else 0.
    opt.step()
    result={key:sum(size*v[key] for size,v in rows)/n for key in rows[0][1]}
    result.update(gradient_norm=float(norm),strength_grad=strength_grad,strength=float(model.strength.detach()))
    return result


@torch.inference_mode()
@native.deterministic_validation
def evaluate(model,data,epoch):
    model.eval();cases=[];gates=[]
    loader=DataLoader(data,batch_size=1,num_workers=2,pin_memory=True)
    for batch in loader:
        image=batch['image'].cuda();matrix=batch['reflection'].cuda();valid=batch['valid'].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): out=model(image,matrix,valid)
        cid=batch['case_id'][0]
        item=native.case_metrics(out.segmentation_logits[0].float(),batch['label'][0].cuda(),data.geometry[cid])
        item.update(case_id=cid,subject_id=batch['subject_id'][0]);cases.append(item)
        gates.append(float(out.gate.float().mean()))
    cases.sort(key=lambda x:x['case_id'])
    assert len(cases)==len({x['case_id'] for x in cases})==108
    metrics=native.summary_ci(cases);metrics['point_hits']=sum(x['automatic_point_hits_gt'] for x in cases)
    model.train()
    return {'at':now(),'epoch':epoch,'metrics':metrics,'per_case':cases,'gate_mean':float(np.mean(gates)),
            'grid':'native1mm crop excludes padding','threshold':.5,'test_access':False}


def checkpoint(path,state):
    tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)


def train(arm,resume=False):
    verify_sources('EXECUTION_FREEZE.json');disk();deadline()
    assert read(R/'DATA_READY.json')['status']=='passed'
    budget=read(R/'BUDGET_DECISION.json');assert budget['status']=='passed'
    micro=int(budget['microbatch']);epochs=int(budget['epochs'])
    target=TMP/'runs'/arm;target.mkdir(parents=True,exist_ok=True)
    record=R/arm;record.mkdir(parents=True,exist_ok=True)
    lock=(record/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=setup(arm);opt=make_optimizer(model);fn=DiceCELoss(sigmoid=True,lambda_dice=1.,lambda_ce=.1)
    source_sha=sha(R/'EXECUTION_FREEZE.json')
    first=1;updates=0;best=-1.;best_epoch=0;elapsed_before=0.
    if resume:
        saved=torch.load(target/'latest.pth',map_location='cpu',weights_only=False)
        assert saved['freeze_sha256']==source_sha and saved['arm']==arm
        model.load_state_dict(saved['model_state_dict'],strict=True);opt.load_state_dict(saved['optimizer_state_dict'])
        first=saved['epoch']+1;updates=saved['updates'];best=saved['best'];best_epoch=saved['best_epoch'];elapsed_before=saved['elapsed_seconds']
        assert updates==(first-1)*122
        del saved
    else:
        backbone_hash=hashlib.sha256()
        for name,value in model.backbone.state_dict().items():
            backbone_hash.update(name.encode());backbone_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        initial_digest=backbone_hash.hexdigest()
        if (record/'TRAIN_PREACCESS.json').exists():raise FileExistsError('prior training access exists')
        write(record/'TRAIN_PREACCESS.json',{'at':now(),'arm':arm,'random_initialization':True,'seed':SEED,'epochs':epochs,
                       'backbone_initial_sha256':initial_digest,'train_scans':972,'validation_scans':108,'test_access':False})
        other='race' if arm=='control' else 'control'
        if (R/other/'TRAIN_PREACCESS.json').exists():
            assert read(R/other/'TRAIN_PREACCESS.json')['backbone_initial_sha256']==initial_digest
    tr=RaceDataset('train');va=RaceDataset('validation');assert len(tr)==972 and len(va)==108
    begun=time.monotonic();model.train()
    if first>epochs:
        # An I/O failure after the last complete checkpoint may leave DONE absent.
        # Finish publication from that checkpoint without repeating any forward.
        report=read(record/f'validation_e{epochs:03}.json')
        status={'at':now(),'state':'complete','arm':arm,'last_complete_epoch':epochs,'target_epochs':epochs,
                'updates':updates,'best_epoch':best_epoch,'best_scan':best,'validation':report['metrics'],
                'elapsed_seconds':elapsed_before,'microbatch':micro,'effective_batch':8,
                'archive_only_recovery':True,'test_access':False}
        write(record/'STATUS.json',status)
    for epoch in range(first,epochs+1):
        deadline();disk();started=time.monotonic();native.seed_all(SEED+epoch*10)
        lr=native.learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200)
        for group in opt.param_groups: group['lr']=lr
        order=torch.randperm(972,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        batches=[order[i:i+8] for i in range(0,972,8)]
        loader=DataLoader(tr,batch_sampler=batches,num_workers=3,pin_memory=True,prefetch_factor=2)
        seen=[];values=[]
        for iteration,batch in enumerate(loader,1):
            deadline();metrics=step(model,opt,fn,batch,epoch,iteration,micro)
            updates+=1;seen+=list(batch['case_id']);values.append((len(batch['image']),metrics))
            if iteration==1 or iteration%20==0 or iteration==122:
                write(record/'HEARTBEAT.json',{'at':now(),'phase':'training','epoch':epoch,'iteration':iteration,'updates':updates,**metrics})
        assert len(seen)==len(set(seen))==972 and set(seen)==set(tr.case_ids)
        assert updates==epoch*122 and all(int(s['step'])==updates for s in opt.state.values())
        training_seconds=time.monotonic()-started
        if not (record/'VALIDATION_PREACCESS.json').exists():
            write(record/'VALIDATION_PREACCESS.json',{'at':now(),'role':'reused108/50','selection':'maximumScan; earliest tie',
                  'every_epoch':True,'threshold':.5,'test_access':False,'freeze_sha256':source_sha})
        write(record/'HEARTBEAT.json',{'at':now(),'phase':'validation','epoch':epoch,'updates':updates})
        report=evaluate(model,va,epoch);write(record/f'validation_e{epoch:03}.json',report)
        if report['metrics']['scan_dice']>best:
            best=report['metrics']['scan_dice'];best_epoch=epoch
            checkpoint(target/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'arm':arm,'freeze_sha256':source_sha})
            (OUT/arm).mkdir(parents=True,exist_ok=True)
            portable=OUT/arm/'best.pth';temporary=portable.with_name('best.pth.tmp')
            shutil.copy2(target/'best.pth',temporary);os.replace(temporary,portable)
            write(record/'BEST.json',{'epoch':epoch,'metrics':report['metrics'],'checkpoint_sha256':sha(OUT/arm/'best.pth')})
        state={'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'epoch':epoch,'arm':arm,
               'updates':updates,'best':best,'best_epoch':best_epoch,'elapsed_seconds':elapsed_before+time.monotonic()-begun,'freeze_sha256':source_sha}
        checkpoint(target/'latest.pth',state)
        if epoch==first or epoch==epochs:
            loaded=torch.load(target/'latest.pth',map_location='cpu',weights_only=False)
            assert all(torch.equal(v.detach().cpu(),loaded['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
            assert all(int(s['step'])==updates for s in loaded['optimizer_state_dict']['state'].values())
            write(record/f'RELOAD_e{epoch}.json',{'at':now(),'status':'passed','sha256':sha(target/'latest.pth'),'updates':updates});del loaded
        status={'at':now(),'state':'complete' if epoch==epochs else 'training','arm':arm,'last_complete_epoch':epoch,'target_epochs':epochs,
                'updates':updates,'best_epoch':best_epoch,'best_scan':best,'validation':report['metrics'],
                'training_seconds':training_seconds,'epoch_seconds':time.monotonic()-started,'elapsed_seconds':elapsed_before+time.monotonic()-begun,
                'learning_rate':lr,'microbatch':micro,'effective_batch':8,'train':{k:sum(n*v[k] for n,v in values)/972 for k in values[0][1]},
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'test_access':False}
        write(record/'STATUS.json',status)
        with (record/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    # Preserve complete optimizer recovery state only in scratch; portable latest weights on persistent storage.
    checkpoint(OUT/arm/'latest.pth',{'model_state_dict':model.state_dict(),'epoch':epochs,'arm':arm,'freeze_sha256':source_sha})
    write(record/'DONE.json',status)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=ARMS);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try: train(args.arm,args.resume)
    except BaseException as error:
        write(R/args.arm/f'FAILURE_{time.time_ns()}.json',{'at':now(),'error':repr(error)});raise
