"""Contract checks for signs, fixed split, scorer, and reference model compatibility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *
os.environ.update(env())
import numpy as np
import torch
from scripts.nnunet_resenc_20261013.finalize import metrics_row
from scripts.segresnet_two_ideas_common import paired
from scripts.segresnet_plain972_common import model_for

def main():
    verify_manifests()
    target=np.zeros((5,6,7),dtype=np.uint8);target[1:3,2:4,3:5]=1
    logits=np.where(target,2.,-2.).astype(np.float32)
    a=metrics_row('synthetic-1','synthetic',logits,target)
    assert a['locator_dsc']==1 and a['automatic_point_hits_gt']
    b=metrics_row('synthetic-1','synthetic',np.full_like(logits,-2),target)
    assert b['locator_dsc']==0
    assert abs(paired([a],[b])['scan_delta']-1)<1e-12
    logits[0,0,0]=-.2
    assert metrics_row('synthetic-1','synthetic',logits,target,.4)['fp']==1
    manifest=read(ROOT/'workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921/deployment/manifest.json')
    identities=[]
    torch.set_num_threads(1)
    for c in manifest['components']:
        path=ROOT/c['checkpoint_path'];assert sha(path)==c['checkpoint_sha256']
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model=model_for(torch.device('cpu'));model.load_state_dict(ckpt['model_state_dict'],strict=True)
        model.eval()
        with torch.inference_mode():out=model(torch.zeros((1,1,32,32,32))).segmentation_logits
        assert tuple(out.shape)==(1,1,32,32,32) and torch.isfinite(out).all()
        identities.append(c)
    write(R/'CPU_CHECKS.json',{'at':now(),'status':'passed','manifest_counts_and_patient_isolation':True,
      'paired_candidate_minus_reference_sign':True,'threshold_and_point':True,'reference_models_strictly_loadable':identities,
      'patient_pixel_access':False,'patient_model_forwards':0})

if __name__=='__main__':main()
