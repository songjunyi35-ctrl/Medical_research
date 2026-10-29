"""Check the new core24 control against all saved0020 predicted pixels; no GT."""
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import probability_for,single_cut

@torch.inference_mode()
def main():
    numeric_mode();torch.cuda.set_per_process_memory_fraction(.15)
    folder=R/'analyses/f4_0000_brain_phase';assert (folder/'DONE.json').exists()
    recipe=read(folder/'PLAN.json')['recipes'][2];original=R/'candidates/0020_brain_domain';records=read(original/'RAW_TEST270.json')['per_case'];params=read(original/'RECIPE.json')['decoder']
    index=read(CACHE/'index.json');images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS));assert [r['case_id'] for r in records]==index['case_ids'];rows=[]
    for i,reference in enumerate(records):
        guard();cid=reference['case_id'];image=np.array(images[i][slices(index['geometry'][cid])],np.float32)
        p=probability_for(cid,recipe['base'],recipe['base_source_job'],torch.from_numpy(image).cuda());point=list(map(int,np.unravel_index(int(p.argmax()),p.shape)))
        valid=supports(image)[0];codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy();cut=single_cut(np.bincount(codes[valid],minlength=65536),params);mask=(codes>=cut)&valid
        saved=np.load(original/'predictions'/(cid+'.npz'));expected=np.unpackbits(saved['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
        different=int((mask!=expected).sum());assert different==0 and point==reference['automatic_point'],(cid,different)
        rows.append(dict(case_id=cid,mask_different_voxels=different,point_equal=True))
    write(folder/'CONTROL_PIXELS_EQ.json',dict(at=now(),state='passed',scans=270,total_mask_different_voxels=0,all_points_equal=True,GT_access=False,neural_forwards_added=0,source_sha256=sha(Path(__file__)),reference_raw_sha256=sha(original/'RAW_TEST270.json'),per_case=rows))
    print(dict(state='passed',scans=270,total_mask_different_voxels=0,GT_access=False,neural_forwards=0))
if __name__=='__main__':main()
