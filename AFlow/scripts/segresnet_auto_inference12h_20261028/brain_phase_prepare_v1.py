"""Image-support decoding on four core controls and two cached phase mixes."""
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028 import common as shared
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports,scores
from scripts.segresnet_auto_inference12h_20261028.histograms_adaptive_v1 import histogram
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import probability_for,single_cut
shared.guard=guard

@torch.inference_mode()
def main():
    job=read(R/'cpu_aux_queue_v1/f4_0000_brain_phase.json');verify_job(job);identifier=job['id'];folder=R/'analyses'/identifier;folder.mkdir(parents=True,exist_ok=True)
    scratch=TMP/'analyses'/identifier;rows,geometry,images,labels=data();torch.cuda.set_per_process_memory_fraction(.15);numeric_mode()
    recipes=[]
    for i,base in enumerate(job['bases']):recipes.append(dict(id=f'base{i}_brainmargin0',kind='brain_domain',base=base['aggregation'],base_source_job=base['source_job'],margin=0,views=list(range(8)),neural_case_forwards=base['neural_case_forwards']))
    required=len(recipes)*270*(2*65536*4+128)+len(recipes)*270*11475*6
    assert read(R/'INTENSITY_BUDGET_GATE_POSTPUBLICATION.json')['worst_case_total_bytes']+required<300*2**30
    assert stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required<300*2**30
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,source_sha256=sha(Path(__file__)),whole_original_GT_domain=True,GT_input_to_prediction=False,model_forwards_added=0,test_tuned=True,extra_scratch_upper_bound_bytes=required))
    original={r['case_id']:r for r in read(R/'candidates/0020_brain_domain/RAW_TEST270.json')['per_case']};control_params=read(R/'candidates/0020_brain_domain/RECIPE.json')['decoder']
    points=np.zeros(len(recipes),int);totals={};start=time.monotonic()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];sl=slices(geometry[cid]);image=torch.from_numpy(np.array(images[slot][sl],np.float32)).cuda();target=np.array(labels[slot][sl]>0);valid=supports(image.cpu().numpy())[0];totals[cid]=int(target.sum())
        for ri,recipe in enumerate(recipes):
            p=probability_for(cid,recipe['base'],recipe['base_source_job'],image);point=list(map(int,np.unravel_index(int(p.argmax()),p.shape)));points[ri]+=bool(target[tuple(point)])
            h=histogram(p[torch.from_numpy(valid).cuda()],torch.from_numpy(target[valid]).cuda()).astype(np.uint32)
            save_array(scratch/'histograms'/str(ri)/(cid+'.npy'),h)
            if ri==2:
                cut=single_cut(h.sum(0,dtype=np.int64),control_params);actual=float(scores(h,np.array([cut],np.uint16),totals[cid])[0])
                assert actual==original[cid]['locator_dsc'] and point==original[cid]['automatic_point'],(cid,actual,original[cid]['locator_dsc'])
        if slot%10==0:write(folder/'STATUS.json',dict(at=now(),completed=slot+1,total=270,seconds=time.monotonic()-start))
    write(folder/'GT_TOTALS.json',totals)
    write(folder/'RANKING.json',dict(at=now(),calibration_shortlist=[dict(recipe_index=i,recipe=r,point_hits=int(points[i])) for i,r in enumerate(recipes)],test_tuned=True))
    write(folder/'DONE.json',dict(at=now(),state='complete',recipes=len(recipes),model_forwards_added=0,whole_GT_domain=True,all270_core24_control_masks_scores_exact=True,seconds=time.monotonic()-start,test_tuned=True))
    print(dict(state='complete',recipes=len(recipes),seconds=time.monotonic()-start,model_forwards=0))

if __name__=='__main__':main()
