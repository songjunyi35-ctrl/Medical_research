import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028 import common as shared
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports,gate
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v2 import aggregate
from scripts.segresnet_auto_inference12h_20261028.histograms_adaptive_v1 import histogram
shared.guard=guard


@torch.inference_mode()
def main():
    job=read(R/'cpu_aux_queue_v1/f1_0000_brain_domain.json');verify_job(job)
    identifier=job['id'];folder=R/'analyses'/identifier;folder.mkdir(parents=True,exist_ok=True);write(folder/'GATE.json',dict(at=now(),**gate()))
    scratch=TMP/'analyses'/identifier;scratch.mkdir(parents=True,exist_ok=True)
    rows,geometry,images,labels=data();recipes=[]
    for bi,base in enumerate(job['bases']):
        for margin in [-1,0,1,2]:recipes.append(dict(id=f'base{bi}_brainmargin{margin}',kind='brain_domain',base=base,margin=margin,views=base['views'],neural_case_forwards=base.get('neural_case_forwards',len(base['views']))))
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,brain_support='fill holes(image!=0), optional6-neighbor dilation; no support control',evaluation='ALL original native-crop GT including outside support',test_tuned=True))
    from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
    required=len(recipes)*270*(2*65536*4+128)+len(recipes)*270*11475*6
    while stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required+256*2**20>300*2**30:
        guard();write(R/'BRAIN_DOMAIN_STATUS.json',dict(at=now(),state='waiting_for_scratch'));time.sleep(10)
    totals={};point_hits=np.zeros(len(recipes),int);outside=[];start=time.monotonic()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];sl=slices(geometry[cid]);image=np.array(images[slot][sl],np.float32);target=np.array(labels[slot][sl]>0);totals[cid]=int(target.sum());masks=supports(image)
        logits={i:torch.from_numpy(np.load(TMP/'groups/r1_0000_core_mirror8/atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)}
        for bi,base in enumerate(job['bases']):
            p,point_prob=aggregate(logits,base,torch.from_numpy(image).cuda());point=list(map(int,np.unravel_index(int(point_prob.argmax()),point_prob.shape)));hit=bool(target[tuple(point)])
            for mi,margin in enumerate([-1,0,1,2]):
                ri=bi*4+mi;support=masks[margin];valid=torch.from_numpy(support).cuda();h=histogram(p[valid],torch.from_numpy(target[support]).cuda())
                save_array(scratch/'histograms'/str(ri)/(cid+'.npy'),h.astype(np.uint32));point_hits[ri]+=hit
        outside.append(dict(case_id=cid,GT_outside_input_support=int((target&~masks[0]).sum()),support_voxels=int(masks[0].sum()),total_voxels=target.size))
        if slot%20==0:write(R/'BRAIN_DOMAIN_STATUS.json',dict(at=now(),completed=slot+1,total=270,seconds=time.monotonic()-start))
    write(folder/'GT_TOTALS.json',totals);write(folder/'SUPPORT_DIAGNOSTIC.json',outside)
    shortlist=[dict(recipe_index=i,recipe=recipe,point_hits=int(point_hits[i])) for i,recipe in enumerate(recipes)]
    write(folder/'RANKING.json',dict(at=now(),calibration_shortlist=shortlist,test_tuned=True,not_ranked_until_calibration=True))
    write(folder/'DONE.json',dict(at=now(),state='complete',recipes=len(recipes),histogram_shortlist=len(recipes),model_forwards=0,test_tuned=True,seconds=time.monotonic()-start,GT_total_domain_unchanged=True))
if __name__=='__main__':main()
