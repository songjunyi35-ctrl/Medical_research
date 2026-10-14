"""Fit8 CPU loader contract: only primary labels differ, no model forwards."""
from datetime import datetime,timezone
import json
import torch
from scripts import fractional_occupancy_protocol as p
from scripts import train_structured_background_aux as original
from scripts.train_fractional_occupancy import OccupancyData


def main():
    out=p.RECORD/'DATASET_CPU_RECEIPT.json'
    if out.exists():raise FileExistsError(out)
    verification=json.loads((p.RECORD/'CACHE_VERIFICATION.json').read_text());assert verification['status']=='passed'
    p.checked(p.RECORD/'CACHE_RECEIPT.json',verification['cache_receipt_sha256'])
    fit,_=p.check_roles();torch.set_num_threads(1)
    hard=original.FitData(fit);soft=OccupancyData(fit);results=[]
    for slot in range(8):
        a,b=hard[slot],soft[slot]
        assert a['case_id']==b['case_id']==fit[slot]['case_id']
        assert torch.equal(a['image'],b['image']) and torch.equal(a['label'][1],b['label'][1])
        assert b['label'].dtype==torch.float32 and b['label'].shape==(2,128,128,128)
        assert torch.isfinite(b['label']).all() and b['label'].min()>=0 and b['label'].max()<=1
        expected=json.loads((p.RECORD/'CACHE_RECEIPT.json').read_text())['details'][slot]
        assert float(b['label'][0].double().sum())==expected['stored_float16_sum']
        results.append({'case_id':a['case_id'],'image_sha256':original.tensor_sha(a['image']),'image_equal':True,'rc_equal':True,'primary_target_changed':not torch.equal(a['label'][0],b['label'][0]),'new_primary_sum':float(b['label'][0].double().sum()),'dtype':'float32'})
    assert any(r['primary_target_changed'] for r in results)
    files=['scripts/verify_fractional_occupancy_dataset.py','scripts/train_fractional_occupancy.py','scripts/fractional_occupancy_protocol.py','scripts/train_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py']
    p.write_new(out,{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'files':{name:p.sha(p.ROOT/name) for name in files},'cache_verification_sha256':p.sha(p.RECORD/'CACHE_VERIFICATION.json'),'access':{'fit_images_and_labels':8,'development':False,'test':False,'confirmation':False,'outer':False,'external':False},'model_forwards':0,'training_admitted':False})
    print('fit8 loader: images/RC exact, only WT targets changed, no model forwards',flush=True)


if __name__=='__main__':main()
