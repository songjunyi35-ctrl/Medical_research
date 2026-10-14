"""Conditional batch8 hard-vs-soft fit smoke, never development scoring."""
from datetime import datetime,timezone
import gc
import json
import time
import torch
from monai.losses import DiceCELoss
from sammed3d.structured_background_aux import StructuredBackgroundLocator,conditional_rc_loss,export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import fractional_occupancy_protocol as p
from scripts import train_structured_background_aux as source_training
from scripts.train_fractional_occupancy import OccupancyData
from scripts.train_t1ce_wt_capg3d import augment_batch
from scripts.train_structured_background_aux import state_sha,tensor_sha


def require_primary_closed():
    record=p.ROOT/'archive/research_records/2026-09-10_weak_elastic'
    result=json.loads((record/'RESULT.json').read_text());verification=json.loads((record/'VERIFICATION_RECEIPT.json').read_text())
    assert verification['status']=='passed';p.checked(record/'RESULT.json',verification['result_sha256'])
    assert result['status']=='completed_reused_development_gate_failed' and result['selected_arm'] is None
    assert not (p.ROOT/'archive/research_records/2026-09-10_weak_elastic_full864/TEST_PREACCESS.json').exists()
    return result


def main():
    if (p.RECORD/'SMOKE_RECEIPT.json').exists():raise FileExistsError('soft target smoke already done')
    require_primary_closed();p.source.check_freeze();fit,_=p.check_roles()
    hard=source_training.FitData(fit);soft=OccupancyData(fit)
    batches={name:[dataset[i] for i in range(8)] for name,dataset in [('hard',hard),('soft',soft)]}
    tensors={name:(torch.stack([r['image'] for r in batch]),torch.stack([r['label'] for r in batch])) for name,batch in batches.items()}
    assert torch.equal(tensors['hard'][0],tensors['soft'][0])
    assert torch.equal(tensors['hard'][1][:,1],tensors['soft'][1][:,1])
    assert not torch.equal(tensors['hard'][1][:,0],tensors['soft'][1][:,0])
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    free,_=torch.cuda.mem_get_info(device)
    if free<28*1024**3:raise RuntimeError('wait for an idle GPU with28GiB free')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    results=[]
    for name in ('hard','soft'):
        seed=2026090711;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=StructuredBackgroundLocator().to(device);initial=state_sha(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        criterion=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
        x0,y0=tensors[name];history=[];torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
        for step in range(6):
            x,y=augment_batch(x0.to(device),y0.to(device),max_shift=8)
            image_sha=tensor_sha(x);rng=tensor_sha(torch.cuda.get_rng_state(device))
            assert y.min()>=0 and y.max()<=1
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                wt,rc=model(x);primary=criterion(wt,y[:,:1])
            auxiliary=conditional_rc_loss(rc,y[:,1:],y[:,:1]);loss=primary+0*auxiliary
            assert torch.isfinite(loss);loss.backward()
            assert float(model.network.conv_final[2].conv.weight.grad[1].abs().sum())==0
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            history.append({'image_sha256':image_sha,'base_rng':rng,'model_rng':tensor_sha(torch.cuda.get_rng_state(device)),'primary_loss':float(primary.detach())})
        model.eval();plain=AutomaticPromptLocator(center_aware=False).to(device).eval();plain.load_state_dict(export_plain_state(model),strict=True)
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):error=float((model(x[:1])[0].float()-plain(x[:1]).segmentation_logits.float()).abs().max())
        results.append({'arm':name,'initial_state_sha256':initial,'trained_state_sha256':state_sha(model.state_dict()),'history':history,'export_error':error,'peak_bytes':torch.cuda.max_memory_allocated(device),'seconds':time.monotonic()-start})
        del model,plain,opt,wt,rc,primary,auxiliary,loss,x,y;gc.collect();torch.cuda.empty_cache()
    control=json.loads((p.source.WORK/'seed2026090711_control/TRAIN_DONE.json').read_text())
    gates={'initial_matches_old_control':all(r['initial_state_sha256']==control['initial_state_sha256'] for r in results),'images_and_rng_exact':all(a['image_sha256']==b['image_sha256'] and a['base_rng']==b['base_rng'] and a['model_rng']==b['model_rng'] for a,b in zip(results[0]['history'],results[1]['history'],strict=True)),'soft_target_changes_training':results[0]['trained_state_sha256']!=results[1]['trained_state_sha256'],'zero_auxiliary_gradient':True,'export_exact':all(r['export_error']==0 for r in results),'memory_under24GiB':all(r['peak_bytes']<24*1024**3 for r in results),'six_steps_under90s':all(r['seconds']<90 for r in results)}
    files=['scripts/smoke_fractional_occupancy.py','scripts/train_fractional_occupancy.py','scripts/fractional_occupancy_protocol.py','tests/test_fractional_occupancy_training.py']
    p.write_new(p.RECORD/'SMOKE_RECEIPT.json',{'status':'passed' if all(gates.values()) else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'gates':gates,'results':results,'files':{n:p.sha(p.ROOT/n) for n in files},'cache_receipt_sha256':p.sha(p.RECORD/'CACHE_RECEIPT.json'),'fit_cases':[r['case_id'] for r in fit[:8]],'access':{'fit_pixels':8,'development':False,'test':False,'confirmation':False,'outer':False,'external':False}})
    print(json.dumps(gates,indent=2),flush=True);assert all(gates.values())


if __name__=='__main__':main()
