"""Hardware amendment: source pretraining only, one GPU/global batch8; unchanged downstream DDP."""
import sys,os,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from scripts.segresnet_two_ideas_common import *


def verify_single():
    verify();a=json.loads((R/'HARDWARE_AMENDMENT_003.json').read_text())
    for n,h in a['new_files'].items():assert sha(ROOT/n)==h,n
    for n,h in a['evidence_sha256'].items():assert sha(ROOT/n)==h,n
    return a


def batches(epoch,n=334):
    order=torch.randperm(n,generator=torch.Generator().manual_seed(BASELINE_SEED+epoch)).tolist()
    return [order[i:i+8] for i in range(0,n,8)]


def step(model,opt,criterion,image,label):
    if not bool(torch.isfinite(image).all() and torch.isfinite(label).all() and (label.flatten(1).sum(1)>0).all()):raise ValueError('invalid training input')
    image,label,fallback=safe_augment(image,label);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(image).segmentation_logits;loss=criterion(logits,label)
    if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    if not bool(norm>0):raise ValueError('zero gradient')
    opt.step()
    return {'loss':float(loss.detach()),'train_dice':float(dice_per_case(logits.detach().float(),label).mean()),
            'grad_norm_before_clip':float(norm),'augmentation_fallbacks':fallback}


def setup():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.cuda.set_device(0);seed_all(BASELINE_SEED)
    model=model_for(torch.device('cuda:0'))
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
    criterion=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)
    return model,opt,criterion


def smoke():
    verify_single();model,opt,criterion=setup();torch.cuda.reset_peak_memory_stats()
    x=torch.randn(8,1,128,128,128,device='cuda');y=torch.zeros_like(x);y[:,:,36:92,36:92,36:92]=1
    before=next(model.parameters()).detach().clone();start=time.monotonic()
    logs=[step(model,opt,criterion,x.clone(),y.clone()),step(model,opt,criterion,x[:6].clone(),y[:6].clone())]
    assert bool((next(model.parameters())-before).abs().max()>0)
    write(R/'GPU_SMOKE_SINGLE_A100.json',{'utc':now(),'status':'passed','worker':1105022,'batch_sizes':[8,6],
          'steps':logs,'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'patient_pixels_access':False,'parameters':sum(p.numel() for p in model.parameters())})


def train():
    verify_single();assert json.loads((R/'GPU_SMOKE_SINGLE_A100.json').read_text())['status']=='passed'
    assert json.loads((R/'OLD_EXTERNAL_QUEUE_STOPPED.json').read_text())['stopped']
    transfer=json.loads((R/'CACHE_TRANSFER_1105022.json').read_text());assert transfer['status']=='passed'
    for n,h in transfer['destination_sha256'].items():assert sha(PRETRAIN_CACHE/n)==h
    dataset=BrainCropDataset(PRETRAIN_CACHE,expected_split=PRETRAIN_SPLIT)
    audit=json.loads((R/'DATA_AUDIT.json').read_text());assert dataset.case_ids==audit['pretrain_case_ids'] and len(dataset)==334
    out=OUT/'pretrain';resume=(out/'latest.pth').exists()
    if out.exists() and any(out.iterdir()) and not resume:raise RuntimeError('nonempty pretrain output without resumable checkpoint')
    out.mkdir(parents=True,exist_ok=True);model,opt,criterion=setup();start_epoch=1;elapsed_before=0.
    if resume:
        state=torch.load(out/'latest.pth',map_location='cpu',weights_only=False);assert state['hardware_mode']=='single_gpu_global8'
        model.load_state_dict(state['model_state_dict'],strict=True);opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch=state['epoch']+1;elapsed_before=state['elapsed_seconds'];del state
    else:
        write(R/'TRAIN_PREACCESS_pretrain.json',{'utc':now(),'worker':1105022,'scans':334,'subjects':334,'epochs':150,
              'global_batch':8,'world_size':1,'steps_per_epoch':42,'hardware_amendment_sha256':sha(R/'HARDWARE_AMENDMENT_003.json'),
              'cache_receipt_sha256':sha(R/'CACHE_TRANSFER_1105022.json'),'test_access':False,'outer_confirmation_access':False,
              'stochastic_trajectory_differs_from_two_rank_recipe':True,'validation_access':False})
        state=torch.load(BASELINE_RUN/'initial.pth',map_location='cpu',weights_only=False);assert state['epoch']==0
        model.load_state_dict(state['model_state_dict'],strict=True);del state
        save(out/'initial.pth',{'model_state_dict':model.state_dict(),'epoch':0,'seed':BASELINE_SEED,'run':'pretrain','hardware_mode':'single_gpu_global8'})
    started=time.monotonic()-elapsed_before;torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,151):
        disk();seed_all(BASELINE_SEED+epoch*10);model.train();lr=lr_for('pretrain',epoch);tick=time.monotonic()
        for g in opt.param_groups:g['lr']=lr
        loader=DataLoader(dataset,batch_sampler=batches(epoch),num_workers=2,pin_memory=True,
                          generator=torch.Generator().manual_seed(BASELINE_SEED+epoch*100))
        ids=[];values=[]
        for i,b in enumerate(loader,1):
            result=step(model,opt,criterion,b['image'].cuda(non_blocking=True),b['label'].cuda(non_blocking=True))
            ids.extend(b['case_id']);values.append((len(b['case_id']),result))
            if i==1 or i%20==0 or i==42:print(json.dumps({'utc':now(),'epoch':epoch,'step':i,'run':'pretrain','worker':1105022,**result}),flush=True)
        assert len(ids)==len(set(ids))==334 and set(ids)==set(dataset.case_ids) and len(values)==42
        elapsed=time.monotonic()-started
        row={'utc':now(),'run':'pretrain','epoch':epoch,'unique_scans':334,'optimizer_updates':42,'cumulative_optimizer_updates':epoch*42,
             'training_seconds':time.monotonic()-tick,'elapsed_seconds':elapsed,'learning_rate':lr,'world_size':1,'global_batch':8,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'train':{k:sum(n*v[k] for n,v in values)/334 for k in ['loss','train_dice','grad_norm_before_clip']},
             'augmentation_fallbacks':sum(v['augmentation_fallbacks'] for _,v in values),'test_access':False}
        payload={'model_state_dict':model.state_dict(),'epoch':epoch,'run':'pretrain','hardware_mode':'single_gpu_global8',
                 'seed':BASELINE_SEED,'elapsed_seconds':elapsed,'cumulative_optimizer_updates':epoch*42}
        if epoch%25==0:save(out/f'epoch{epoch:03d}.pth',payload)
        if epoch==150:save(out/'final.pth',payload)
        save(out/'latest.pth',{**payload,'optimizer_state_dict':opt.state_dict()})
        with (out/'TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(out/'STATUS.json',{'updated_utc':now(),'status':'completed' if epoch==150 else 'running','run':'pretrain','worker':1105022,
              'last_complete_epoch':epoch,'total_epochs':150,'optimizer_updates':epoch*42,'elapsed_seconds':elapsed,
              'hardware_mode':'single_gpu_global8','test_pixels_access':False})
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    try:{'smoke':smoke,'train':train}[sys.argv[1]]()
    except BaseException as e:
        write(R/f'FAILURE_SINGLE_PRETRAIN_{sys.argv[1]}_{int(time.time())}.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
