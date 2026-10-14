"""Conditional fit-only GPU compatibility smoke, never a selection readout."""
from datetime import datetime,timezone
import gc
import json
import time
import torch
from monai.losses import DiceCELoss
from sammed3d.weak_elastic import WeakElastic
from sammed3d.structured_background_aux import StructuredBackgroundLocator,conditional_rc_loss,export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import weak_elastic_full864_protocol as p
from scripts.train_weak_elastic_full864 import FullData,full_weights
from scripts.train_structured_background_aux import state_sha,tensor_sha
from scripts.train_t1ce_wt_capg3d import augment_batch


def eligible_source():
    result_path=p.source.RECORD/'RESULT.json';verification_path=p.source.RECORD/'VERIFICATION_RECEIPT.json'
    result=json.loads(result_path.read_text());verification=json.loads(verification_path.read_text())
    assert verification['status']=='passed';p.checked(result_path,verification['result_sha256'])
    assert result['status']=='completed_reused_development_supported' and result['selected_arm'] in p.source.ARMS
    assert result['arm_gates'][result['selected_arm']]['eligible']
    return result


def main():
    if (p.RECORD/'SMOKE_RECEIPT.json').exists():raise FileExistsError('full864 smoke already done')
    result=eligible_source();fit,_=p.check_roles();original_fit,_=p.source.check_roles()
    positions={r['case_id']:i for i,r in enumerate(fit)}
    data=FullData(fit);batches=[data[positions[r['case_id']]] for r in original_fit[:8]]
    x0=torch.stack([b['image'] for b in batches]);y0=torch.stack([b['label'] for b in batches]);assert y0[:,1].count_nonzero()==0
    native=json.loads(p.NATIVE.read_text())['native_wt_voxels'];weights=full_weights([r['case_id'] for r in fit],native,result['selected_arm'])
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    results=[]
    for arm in p.ARMS:
        seed=p.SEEDS[0];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=StructuredBackgroundLocator().to(device);initial=state_sha(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        loss_fn=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
        deform=WeakElastic(seed);history=[];torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
        for step in range(6):
            image,labels=augment_batch(x0.to(device),y0.to(device),max_shift=8)
            before=tensor_sha(torch.cuda.get_rng_state(device))
            if arm=='full_elastic':image,labels=deform(image,labels)
            assert before==tensor_sha(torch.cuda.get_rng_state(device))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                wt,rc=model(image);primary=loss_fn(wt,labels[:,:1])
            aux=conditional_rc_loss(rc,labels[:,1:],labels[:,:1]);loss=primary+0*aux
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            history.append({'base_rng':before,'model_rng':tensor_sha(torch.cuda.get_rng_state(device)),'loss':float(loss.detach())})
        model.eval();plain=AutomaticPromptLocator(center_aware=False).to(device).eval();plain.load_state_dict(export_plain_state(model),strict=True)
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            error=float((model(image[:1])[0].float()-plain(image[:1]).segmentation_logits.float()).abs().max())
        results.append({'arm':arm,'initial_state_sha256':initial,'history':history,'export_error':error,'peak_bytes':torch.cuda.max_memory_allocated(device),'seconds':time.monotonic()-start,'deformation':deform.receipt()})
        del model,plain,opt,wt,rc,primary,aux,loss,image,labels;gc.collect();torch.cuda.empty_cache()
    gates={'same_initial':results[0]['initial_state_sha256']==results[1]['initial_state_sha256'],'same_rng':all(a['base_rng']==b['base_rng'] and a['model_rng']==b['model_rng'] for a,b in zip(results[0]['history'],results[1]['history'],strict=True)),'export_exact':all(r['export_error']==0 for r in results),'memory_under24GiB':all(r['peak_bytes']<24*1024**3 for r in results),'elastic_exercised':results[1]['deformation']['selected']>0,'six_steps_under90s':all(r['seconds']<90 for r in results),'zero_auxiliary_targets':True}
    files=['scripts/smoke_weak_elastic_full864.py','scripts/train_weak_elastic_full864.py','scripts/weak_elastic_full864_protocol.py','tests/test_weak_elastic_full864.py','sammed3d/weak_elastic.py']
    p.write_new(p.RECORD/'SMOKE_RECEIPT.json',{'status':'passed' if all(gates.values()) else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'gates':gates,'results':results,'sampling_weight_sha256':tensor_sha(weights),'selected_arm':result['selected_arm'],'source_result_sha256':p.sha(p.source.RECORD/'RESULT.json'),'files':{n:p.sha(p.ROOT/n) for n in files},'fit_cases':[r['case_id'] for r in original_fit[:8]],'access':{'original_fit_pixels':8,'old_development_pixels':False,'test':False,'confirmation':False,'outer':False,'external':False}})
    print(json.dumps({'gates':gates,'sampling_weight_sha256':tensor_sha(weights)},indent=2),flush=True);assert all(gates.values())


if __name__=='__main__':main()
