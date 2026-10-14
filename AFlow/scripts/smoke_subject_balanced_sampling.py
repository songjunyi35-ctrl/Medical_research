"""After preceding study finishes, verify the new CPU sampler/GPU training interface."""
from datetime import datetime,timezone
import json
import time
import torch
from monai.losses import DiceCELoss
from sammed3d.structured_background_aux import StructuredBackgroundLocator,conditional_rc_loss,export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import subject_balanced_protocol as p
from scripts.train_subject_balanced_sampling import balanced_weights
from scripts.train_structured_background_aux import state_sha,tensor_sha
from scripts.train_t1ce_wt_capg3d import augment_batch
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights


def main():
    out=p.RECORD/'SMOKE_RECEIPT.json'
    if out.exists():raise FileExistsError(out)
    assert json.loads((p.SOURCE_RECORD/'CHAIN_DONE.json').read_text())['status']=='completed'
    fit,_=p.check_roles();p.checked(p.NATIVE,p.NATIVE_SHA)
    native=json.loads(p.NATIVE.read_text())['native_wt_voxels'];ids=[r['case_id'] for r in fit]
    original=native_sampling_weights(ids,native);balanced=balanced_weights(ids,native)
    mapping={r['case_id']:r['subject_id'] for r in fit}
    groups={s:[i for i,c in enumerate(ids) if mapping[c]==s] for s in set(mapping.values())}
    mass_error=max(abs(float(balanced[ix].sum())-1) for ix in groups.values())
    ratio_error=max(float(((balanced[ix]/balanced[ix][0])-(original[ix]/original[ix][0])).abs().max()) for ix in groups.values())
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    image0=torch.randn(8,1,128,128,128,generator=torch.Generator().manual_seed(2026090802))
    labels0=torch.zeros(8,2,128,128,128);labels0[:,0,35:65,35:65,35:65]=1;labels0[:,1,70:95,70:95,70:95]=1
    results=[]
    for policy,weights in [('native',original),('subject_balanced',balanced)]:
        torch.manual_seed(2026090711);torch.cuda.manual_seed_all(2026090711)
        model=StructuredBackgroundLocator().to(device);initial=state_sha(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        criterion=DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1,lambda_ce=.1)
        indices=torch.multinomial(weights,32,replacement=True,generator=torch.Generator().manual_seed(2026090711))
        history=[];torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
        for i in range(4):
            # Synthetic image offset depends on sampled ID; no patient image loaded.
            offset=indices[i*8:(i+1)*8].float().reshape(8,1,1,1,1)/6840
            image,labels=augment_batch((image0+offset).to(device),labels0.to(device).clone(),max_shift=8)
            rng=tensor_sha(torch.cuda.get_rng_state(device))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                wt,rc=model(image);primary=criterion(wt,labels[:,:1])
            auxiliary=conditional_rc_loss(rc,labels[:,1:],labels[:,:1]);loss=primary+0*auxiliary
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            assert torch.isfinite(loss)
            assert float(model.network.conv_final[2].conv.weight.grad[1].abs().sum())==0
            history.append({'augmentation_rng':rng,'loss':float(loss.detach())})
        model.eval();plain=AutomaticPromptLocator(center_aware=False).to(device).eval();plain.load_state_dict(export_plain_state(model),strict=True)
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            error=float((model(image[:1])[0].float()-plain(image[:1]).segmentation_logits.float()).abs().max())
        results.append({'policy':policy,'initial_sha256':initial,'indices':indices.tolist(),'history':history,'export_error':error,'peak_bytes':torch.cuda.max_memory_allocated(device),'seconds':time.monotonic()-start})
        del model,plain,opt,loss,primary,auxiliary,wt,rc,image,labels
        torch.cuda.empty_cache()
    control=json.loads((p.SOURCE_WORK/'seed2026090711_control/TRAIN_DONE.json').read_text())
    gates={'patient_mass_error_below1e12':mass_error<1e-12,'within_patient_ratio_error_below1e12':ratio_error<1e-12,'native_weights_match_control':tensor_sha(original)==control['sampling_weight_sha256'],'initial_matches_control':all(r['initial_sha256']==control['initial_state_sha256'] for r in results),'augmentation_rng_equal':all(a['augmentation_rng']==b['augmentation_rng'] for a,b in zip(results[0]['history'],results[1]['history'])),'indices_change_as_intended':results[0]['indices']!=results[1]['indices'],'export_exact':all(r['export_error']==0 for r in results),'memory_gate':all(r['peak_bytes']<24*1024**3 for r in results)}
    p.write_new(out,{'status':'passed' if all(gates.values()) else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'gates':gates,'mass_error':mass_error,'ratio_error':ratio_error,'results':results,'patient_arrays_accessed':False,'model_sha256':p.sha(p.ROOT/'sammed3d/structured_background_aux.py'),'runner_sha256':p.sha(__file__)})
    print(json.dumps(gates,indent=2))
    assert all(gates.values())


if __name__=='__main__':main()
