"""Pretrained full972, fixed inverse-patient-scan-count loss, one A100/global8."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from scripts.patient_equal_common import *


def setup():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.cuda.set_device(0);seed_all(SEED);torch.backends.cuda.matmul.allow_tf32=False
    model=model_for(torch.device('cuda:0'));opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    return model,opt,PatientEqualDiceCE()


def step(model,opt,criterion,image,label,weights):
    if not bool(torch.isfinite(image).all() and torch.isfinite(label).all() and (label.flatten(1).sum(1)>0).all()):raise RuntimeError('invalid input')
    image,label,fallback=safe_augment(image,label);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits;loss=criterion(logits,label,weights)
    if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    if not bool(norm>0):raise RuntimeError('zero gradient')
    opt.step()
    return {'weighted_loss':float(loss.detach()),'unweighted_train_dice':float(dice_per_case(logits.detach().float(),label).mean()),
            'grad_norm_before_clip':float(norm),'augmentation_fallbacks':fallback}


def smoke():
    verify();model,opt,criterion=setup();x=torch.randn(8,1,128,128,128,device='cuda');y=torch.zeros_like(x);y[:,:,36:92,36:92,36:92]=1
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        z=model(x).segmentation_logits;a=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)(z,y)
        b=criterion(z,y,torch.ones(8,device='cuda'));assert torch.allclose(a,b,atol=1e-6)
    before=next(model.parameters()).detach().clone();torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    w=torch.tensor([.218918919,.437837838,.72972973,1.0945946,1.0945946,2.1891892,1.0945946,2.1891892],device='cuda')
    values=[step(model,opt,criterion,x.clone(),y.clone(),w),step(model,opt,criterion,x[:4].clone(),y[:4].clone(),w[:4])]
    assert bool((next(model.parameters())-before).abs().max()>0)
    write(R/'GPU_SMOKE.json',{'utc':now(),'status':'passed','worker':1105022,'batch_sizes':[8,4],'steps':values,
          'uniform_weight_loss_error':abs(float(a)-float(b)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'seconds':time.monotonic()-tick,'patient_pixels_access':False})


def train():
    verify();assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    audit=json.loads((R/'DATA_AUDIT.json').read_text());weights=json.loads((R/'WEIGHTS.json').read_text())['case_weights']
    ds=WeightedDataset(audit['train_case_ids'],weights);val=BrainCropDataset(VAL_CACHE,expected_split='validation',case_ids=audit['validation_case_ids'])
    vl=DataLoader(val,batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
    model,opt,criterion=setup();resume=(OUT/'latest.pth').exists();start_epoch=1;best=-1.;best_epoch=0;elapsed_before=0.
    if OUT.exists() and any(OUT.iterdir()) and not resume:raise RuntimeError('refuse nonempty output')
    OUT.mkdir(parents=True,exist_ok=True)
    if resume:
        state=torch.load(OUT/'latest.pth',map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict'],strict=True);opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch=state['epoch']+1;best=state['best_score'];best_epoch=state['best_epoch'];elapsed_before=state['elapsed_seconds'];del state
    else:
        assert sha(SOURCE)==SOURCE_SHA;state=torch.load(SOURCE,map_location='cpu',weights_only=False);assert state['epoch']==150
        model.load_state_dict(state['model_state_dict'],strict=True);del state
        write(R/'TRAIN_PREACCESS.json',{'utc':now(),'train':972,'subjects':444,'source_sha256':SOURCE_SHA,'weight_sha256':sha(R/'WEIGHTS.json'),
          'epochs':200,'global_batch':8,'steps_per_epoch':122,'world_size':1,'worker':1105022,'test_access':False,
          'freeze_sha256':sha(R/'FREEZE.json'),'single_vs_old_dual_stochastic_difference_disclosed':True})
        save(OUT/'initial.pth',{'model_state_dict':model.state_dict(),'epoch':0,'source_sha256':SOURCE_SHA,'freeze_sha256':sha(R/'FREEZE.json')})
    started=time.monotonic()-elapsed_before;torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,201):
        disk();seed_all(SEED+epoch*10);model.train();tick=time.monotonic()
        for g in opt.param_groups:g['lr']=lr(epoch)
        loader=DataLoader(ds,batch_sampler=batches(epoch),num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(SEED+epoch*100))
        ids=[];values=[];sum_w=0.
        for i,batch in enumerate(loader,1):
            weights_tensor=batch['patient_weight'].to(device='cuda',dtype=torch.float32)
            value=step(model,opt,criterion,batch['image'].cuda(non_blocking=True),batch['label'].cuda(non_blocking=True),weights_tensor)
            ids.extend(batch['case_id']);values.append((len(batch['case_id']),value));sum_w+=float(batch['patient_weight'].sum())
            if i==1 or i%30==0 or i==122:print(json.dumps({'utc':now(),'epoch':epoch,'step':i,**value}),flush=True)
        assert len(ids)==len(set(ids))==972 and set(ids)==set(ds.case_ids) and len(values)==122 and abs(sum_w-972)<1e-7
        train_seconds=time.monotonic()-tick
        if epoch==1:write(R/'VALIDATION_PREACCESS.json',{'utc':now(),'role':'fixed reused108/50','selection':'unweighted Scan Dice, earliest tie','test_access':False})
        result=validate(model,vl,torch.device('cuda:0'));assert result['cases']==108 and result['subjects']==50
        result.update(epoch=epoch,created_utc=now(),role='reused validation108',test_access=False)
        report=OUT/f'validation_e{epoch:03d}.json';write(report,result)
        if result['locator_dsc_mean']>best:
            best=result['locator_dsc_mean'];best_epoch=epoch
            save(OUT/'best.pth',{'model_state_dict':model.state_dict(),'epoch':epoch,'source_sha256':SOURCE_SHA,'freeze_sha256':sha(R/'FREEZE.json')})
            write(OUT/'BEST.json',{'epoch':epoch,'scan_dice':best,'subject_dice':result['locator_subject_dsc_mean'],'point_hit':result['automatic_point_hit_rate'],
                  'report':report.name,'report_sha256':sha(report),'checkpoint_sha256':sha(OUT/'best.pth')})
        payload={'model_state_dict':model.state_dict(),'epoch':epoch,'best_score':best,'best_epoch':best_epoch,'elapsed_seconds':time.monotonic()-started,'freeze_sha256':sha(R/'FREEZE.json')}
        if epoch%25==0:save(OUT/f'epoch{epoch:03d}.pth',payload)
        save(OUT/'latest.pth',{**payload,'optimizer_state_dict':opt.state_dict()})
        row={'utc':now(),'epoch':epoch,'unique_scans':972,'unique_subjects':444,'weight_sum':sum_w,'updates':122,'cumulative_updates':epoch*122,
             'learning_rate':lr(epoch),'training_seconds':train_seconds,'elapsed_seconds':time.monotonic()-started,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'validation':{k:v for k,v in result.items() if k!='per_case'},
             'train':{k:sum(n*v[k] for n,v in values)/972 for k in ['weighted_loss','unweighted_train_dice']}}
        with (OUT/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(OUT/'STATUS.json',{'updated_utc':now(),'status':'running','last_complete_epoch':epoch,'total_epochs':200,
              'best_epoch':best_epoch,'best_scan_dice':best,'optimizer_updates':epoch*122,'elapsed_seconds':time.monotonic()-started,'worker':1105022,'test_access':False})
        print(json.dumps(row),flush=True)
    sel=json.loads((OUT/'BEST.json').read_text());state=torch.load(OUT/'best.pth',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True);del state
    actual=validate(model,vl,torch.device('cuda:0'));expected=json.loads((OUT/sel['report']).read_text())
    assert [r['case_id'] for r in actual['per_case']]==[r['case_id'] for r in expected['per_case']]
    err=max(abs(a['locator_dsc']-b['locator_dsc']) for a,b in zip(actual['per_case'],expected['per_case']));assert err<=1e-6
    write(R/'RELOAD_CHECK.json',{'utc':now(),'status':'passed','max_dice_error':err,'checkpoint_sha256':sha(OUT/'best.pth')})
    final=json.loads((OUT/'STATUS.json').read_text());final.update(status='completed',updated_utc=now());write(OUT/'STATUS.json',final)


if __name__=='__main__':
    try:{'smoke':smoke,'train':train}[sys.argv[1]]()
    except BaseException as e:write(R/f'FAILURE_{sys.argv[1]}_{int(time.time())}.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
