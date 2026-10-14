"""Limited batch8 fit-only inference smoke alongside ongoing training."""
from datetime import datetime,timezone
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.peak_protected_cleanup import cleanup_logits
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts import peak_cleanup_protocol as p


@torch.inference_mode()
def main():
    out=p.RECORD/'SMOKE_RECEIPT.json'
    if out.exists():raise FileExistsError(out)
    controls=p.source_controls();fit,_=p.check_roles()
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    free,total=torch.cuda.mem_get_info(device)
    if free<14*1024**3:raise RuntimeError('wait for >=14GiB free; no fit/model inference started')
    torch.cuda.set_per_process_memory_fraction(.3,device);torch.cuda.reset_peak_memory_stats(device)
    data=BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in fit[:8]])
    batch=next(iter(DataLoader(data,batch_size=8,num_workers=0)))
    model=AutomaticPromptLocator(center_aware=False).to(device).eval()
    ck=torch.load(controls['2026090711']['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
    start=time.monotonic();image=batch['image'].to(device)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
    output=logits[:,0].cpu().numpy();records=[]
    for case,x in zip(batch['case_id'],output,strict=True):
        y,info=cleanup_logits(x)
        assert np.isfinite(y).all() and int(y.argmax())==int(x.argmax()) and np.all((y>0)<=(x>0))
        records.append({'case_id':case,**info})
    peak=torch.cuda.max_memory_allocated(device)
    gates={'point_exact':True,'only_removes_positive_voxels':True,'finite':True,'batch8':len(records)==8,'memory_under12GiB':peak<=12*1024**3}
    files=['sammed3d/peak_protected_cleanup.py','scripts/peak_cleanup_protocol.py','scripts/smoke_peak_cleanup.py','tests/test_peak_protected_cleanup.py']
    p.write_new(out,{'status':'passed' if all(gates.values()) else 'failed','created_utc':datetime.now(timezone.utc).isoformat(),'gates':gates,'fit_cases':records,'free_gpu_bytes_before':free,'total_gpu_bytes':total,'peak_allocated_bytes':peak,'seconds':time.monotonic()-start,'files':{name:p.sha(p.ROOT/name) for name in files},'checkpoint':controls['2026090711']['checkpoint'],'access':{'fit_images':8,'development':False,'test':False,'confirmation':False,'external':False},'forwards':8})
    print(json.dumps({'gates':gates,'peak_gib':peak/1024**3,'seconds':time.monotonic()-start},indent=2),flush=True)
    assert all(gates.values())


if __name__=='__main__':main()
