"""Reference row: the frozen original972 pure SegResNet e193 scored on the SAME native 1mm crop grid.

The baseline predicts on its 128^3 resized view; its logits are resampled back to the native crop
(trilinear, primary) and its mask by nearest (secondary), then scored against the native WT label.
Validation108 only; no test pixels.  Also re-derives the 128^3-grid Dice to prove the checkpoint and
cache are the ones behind the recorded 0.740032.
"""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from scripts.segresnet_native_res_common import *
from scripts.train_t1ce_wt_capg3d import BrainCropDataset


def main():
    verify();fast_cudnn();device=torch.device('cuda:0')
    if sha(BASELINE_CKPT)!=BASELINE_CKPT_SHA:raise RuntimeError('baseline checkpoint hash changed')
    data=json.loads((R/'DATA_AUDIT.json').read_text())
    grid=BrainCropDataset(CANONICAL['validation'],expected_split='validation',case_ids=data['validation_case_ids'])
    native=NativeCanvasDataset(VAL_CACHE,expected_split='validation',case_ids=data['validation_case_ids'])
    assert grid.case_ids==native.case_ids and len(grid)==108
    reported={r['case_id']:r for r in json.loads(BASELINE_VAL_REPORT.read_text())['per_case']}
    write(R/'BASELINE_NATIVE_PREACCESS.json',{'utc':now(),'checkpoint':str(BASELINE_CKPT.relative_to(ROOT)),'checkpoint_sha256':BASELINE_CKPT_SHA,'role':'reused fixed validation108/50 only','purpose':'score the frozen baseline on the native crop grid as the comparison row; no selection, no tuning','test_access':False,'freeze_sha256':sha(R/'FREEZE.json')})
    model=model_for(device);state=torch.load(BASELINE_CKPT,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state_dict'],strict=True);model.eval();del state
    rows=[];tick=time.monotonic()
    with torch.no_grad():
        for start in range(0,108,2):
            items=[grid[i] for i in range(start,min(start+2,108))]
            image=torch.stack([it['image'] for it in items]).to(device);label128=torch.stack([it['label'] for it in items]).to(device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
            d128=dice_per_case(logits,label128)
            for k,it in enumerate(items):
                cid=it['case_id'];geo=native.geometry[cid];nat=native[start+k];assert nat['case_id']==cid
                s=canvas_slices(geo);crop_label=nat['label'][(slice(None),)+s].to(device)
                extent=tuple(int(v) for v in geo['extent'])
                up=F.interpolate(logits[k:k+1],size=extent,mode='trilinear',align_corners=False)[0]
                mask_nn=F.interpolate((logits[k:k+1]>0).float(),size=extent,mode='nearest-exact')[0]
                dsc_tri=float(dice_per_case(up.unsqueeze(0),crop_label.unsqueeze(0))[0]);dsc_nn=float(dice_per_case((mask_nn*2-1).unsqueeze(0),crop_label.unsqueeze(0))[0])
                flat=int(torch.argmax(up[0].flatten()));h,w=extent[1],extent[2];z=flat//(h*w);rem=flat%(h*w);y=rem//w;x=rem%w
                rows.append({'case_id':cid,'subject_id':it['subject_id'],'locator_dsc':dsc_tri,'automatic_point':[int(z),int(y),int(x)],'automatic_point_hits_gt':bool(crop_label[0,z,y,x]>0.5),'crop_extent':list(extent),'dice_nearest_mask':dsc_nn,'dice_grid128_recomputed':float(d128[k]),'dice_grid128_reported':float(reported[cid]['locator_dsc']),'point_hit_grid128_reported':bool(reported[cid]['automatic_point_hits_gt'])})
    err=max(abs(r['dice_grid128_recomputed']-r['dice_grid128_reported']) for r in rows)
    if err>1e-3:raise RuntimeError(f'128-grid reproduction error {err} too large; wrong checkpoint or cache')
    nearest=[{**r,'locator_dsc':r['dice_nearest_mask']} for r in rows];grid128=[{**r,'locator_dsc':r['dice_grid128_recomputed'],'automatic_point_hits_gt':r['point_hit_grid128_reported']} for r in rows]
    out={'utc':now(),'method':'original972 pure SegResNet e193 (frozen baseline), 128^3 prediction resampled to the native crop grid','checkpoint_sha256':BASELINE_CKPT_SHA,'role':'reused fixed validation108/50','grid':'native 1mm image-only brain crop','primary':'trilinear logits upsampling, threshold 0','test_access':False,
         'native_trilinear_logits':summary_ci(rows),'native_nearest_mask':summary_ci(nearest),'grid128_recomputed':summary_ci(grid128),'grid128_reported_scan_dice':float(np.mean([r['dice_grid128_reported'] for r in rows])),'max_grid128_reproduction_error':err,'seconds':time.monotonic()-tick,'per_case':rows}
    write(R/'BASELINE_E193_NATIVE_VALIDATION.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='per_case'}),flush=True)


if __name__=='__main__':main()
