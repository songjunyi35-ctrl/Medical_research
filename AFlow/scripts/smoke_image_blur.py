"""Fit-only paired 40GB GPU engineering smoke; no development inference."""
from datetime import datetime,timezone
import gc
import json
import time
import torch
from monai.losses import DiceCELoss
from sammed3d.image_blur import ImageBlur
from sammed3d.structured_background_aux import StructuredBackgroundLocator,conditional_rc_loss,export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import image_blur_protocol as p
from scripts.train_structured_background_aux import FitData,state_sha,tensor_sha
from scripts.train_t1ce_wt_capg3d import augment_batch


def main():
    output=p.RECORD/'SMOKE_RECEIPT.json'
    if output.exists():raise FileExistsError(output)
    p.source.check_freeze();fit,_=p.check_roles()
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    data=FitData(fit)
    batches=[data[i] for i in range(8)]
    image0=torch.stack([b['image'] for b in batches]);labels0=torch.stack([b['label'] for b in batches])
    results=[]
    for blur in (False,True):
        seed=2026090711;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=StructuredBackgroundLocator().to(device)
        initial=state_sha(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        criterion=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
        deform=ImageBlur(seed)
        history=[];torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
        for step in range(6):
            x,y=augment_batch(image0.to(device),labels0.to(device),max_shift=8)
            before=tensor_sha(torch.cuda.get_rng_state(device))
            original_labels=y.clone()
            if blur:x,y=deform(x,y)
            assert torch.equal(y,original_labels)
            del original_labels
            after=tensor_sha(torch.cuda.get_rng_state(device));assert before==after
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                wt,rc=model(x);primary=criterion(wt,y[:,:1])
            aux=conditional_rc_loss(rc,y[:,1:],y[:,:1]);loss=primary+0*aux
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            history.append({'base_cuda_rng_sha256':before,'post_model_cuda_rng_sha256':tensor_sha(torch.cuda.get_rng_state(device)),'loss':float(loss.detach())})
        model.eval();plain=AutomaticPromptLocator(center_aware=False).to(device).eval();plain.load_state_dict(export_plain_state(model),strict=True)
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            error=float((model(x[:1])[0].float()-plain(x[:1]).segmentation_logits.float()).abs().max())
        torch.cuda.synchronize()
        results.append({'blur':blur,'initial_sha256':initial,'history':history,'export_max_abs_error':error,'peak_bytes':torch.cuda.max_memory_allocated(device),'seconds':time.monotonic()-start,'deformation':deform.receipt()})
        del model,plain,optimizer,wt,rc,primary,aux,loss,x,y
        gc.collect();torch.cuda.empty_cache()
    old=json.loads(p.control_path(2026090711,'native_blur').read_text())
    gates={'initial_matches_old':all(r['initial_sha256']==old['initial_state_sha256'] for r in results),'base_and_model_rng_equal':all(a['base_cuda_rng_sha256']==b['base_cuda_rng_sha256'] and a['post_model_cuda_rng_sha256']==b['post_model_cuda_rng_sha256'] for a,b in zip(results[0]['history'],results[1]['history'],strict=True)),'export_exact':all(r['export_max_abs_error']==0 for r in results),'memory_under24GiB':all(r['peak_bytes']<24*1024**3 for r in results),'blur_exercised':results[1]['deformation']['selected']>0,'bounded':results[1]['deformation']['max_observed_sigma']<=1.000001,'under_90_seconds_per_six_steps':all(r['seconds']<90 for r in results)}
    report={'status':'passed' if all(gates.values()) else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'gates':gates,'results':results,'devices':[{'name':torch.cuda.get_device_name(i),'total_memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())],'fit_cases':[r['case_id'] for r in fit[:8]],'access':{'fit_pixels':8,'development_pixels':False,'test':False,'confirmation':False,'outer':False,'external':False},'files':{name:p.sha(p.ROOT/name) for name in ['scripts/smoke_image_blur.py','sammed3d/image_blur.py','scripts/train_image_blur.py','scripts/image_blur_protocol.py']}}
    p.write_new(output,report);print(json.dumps(report,indent=2),flush=True)
    assert all(gates.values())


if __name__=='__main__':main()
