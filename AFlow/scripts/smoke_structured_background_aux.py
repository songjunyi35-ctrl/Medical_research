"""Paired synthetic GPU training, gradient and exact export checks; no case arrays."""
import json
import time
import torch
from monai.losses import DiceCELoss
from sammed3d.structured_background_aux import StructuredBackgroundLocator, conditional_rc_loss, export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import structured_background_protocol as p
from scripts.train_structured_background_aux import state_sha, tensor_sha
from scripts.train_t1ce_wt_capg3d import augment_batch


def main():
    output = p.RECORD / 'SMOKE_RECEIPT.json'
    if output.exists():
        raise FileExistsError(output)
    p.check_roles()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    generator = torch.Generator().manual_seed(2026090702)
    source_image = torch.randn(8,1,128,128,128,generator=generator)
    source_labels = torch.zeros(8,2,128,128,128)
    source_labels[:,0,35:65,35:65,35:65] = 1
    source_labels[:,1,70:95,70:95,70:95] = 1
    results = []
    for coefficient in (0., .1):
        torch.manual_seed(2026090711); torch.cuda.manual_seed_all(2026090711)
        model = StructuredBackgroundLocator().to(device)
        initial = state_sha(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5)
        loss_fn = DiceCELoss(sigmoid=True,squared_pred=False,lambda_dice=1.,lambda_ce=.1)
        torch.cuda.reset_peak_memory_stats(device)
        records = []
        torch.cuda.synchronize(); start=time.monotonic()
        for step in range(4):
            image, labels = augment_batch(source_image.to(device).clone(),source_labels.to(device).clone(),max_shift=8)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                wt_logits, rc_logits = model(image)
                primary = loss_fn(wt_logits,labels[:,:1])
            auxiliary=conditional_rc_loss(rc_logits,labels[:,1:],labels[:,:1])
            loss=primary+coefficient*auxiliary
            loss.backward()
            grad=model.network.conv_final[2].conv.weight.grad[1].abs().sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            records.append({'image':tensor_sha(image),'label':tensor_sha(labels),'primary':float(primary.detach()),'auxiliary':float(auxiliary.detach()),'aux_gradient_abs_sum':grad})
        torch.cuda.synchronize(); elapsed=time.monotonic()-start
        model.eval(); plain=AutomaticPromptLocator(center_aware=False).to(device).eval()
        plain.load_state_dict(export_plain_state(model),strict=True)
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            error=float((model(image[:1])[0].float()-plain(image[:1]).segmentation_logits.float()).abs().max())
        results.append({'coefficient':coefficient,'initial':initial,'records':records,'seconds_four_steps':elapsed,'export_error':error,'peak_bytes':torch.cuda.max_memory_allocated(device)})
        del model,plain,optimizer,loss,primary,auxiliary,wt_logits,rc_logits,image,labels
        torch.cuda.empty_cache()
    a,b=results
    gates={'same_initial_state':a['initial']==b['initial'], 'same_augmentation':all((x['image'],x['label'])==(y['image'],y['label']) for x,y in zip(a['records'],b['records'])), 'same_first_primary_loss':a['records'][0]['primary']==b['records'][0]['primary'], 'control_aux_gradient_zero':all(x['aux_gradient_abs_sum']==0 for x in a['records']), 'candidate_aux_gradient_nonzero':all(x['aux_gradient_abs_sum']>0 for x in b['records']), 'exact_export':all(x['export_error']==0 for x in results), 'memory':all(x['peak_bytes']<24*1024**3 for x in results)}
    receipt={'status':'passed' if all(gates.values()) else 'failed','gates':gates,'results':results,'synthetic_only':True,'patient_arrays_accessed':False,'new_evaluation_metrics':False,'model_sha256':p.sha(p.ROOT/'sammed3d/structured_background_aux.py')}
    p.write_new(output,receipt)
    print(json.dumps(receipt,indent=2))
    if not all(gates.values()):
        raise RuntimeError(gates)


if __name__ == '__main__':
    main()
