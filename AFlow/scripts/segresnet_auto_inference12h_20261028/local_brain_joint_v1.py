"""Finite joint decoder/cavity grid; decisions only see predictions and image."""
import concurrent.futures,multiprocessing as mp
import numpy as np
import torch
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import probability_for,single_cut
from scripts.segresnet_auto_inference12h_20261028.fast_volume_cuts_v1 import fast_cuts
from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import apply as component_apply
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import apply as binary_apply

ID='h1_0000_local_brain_joint';FOLDER=R/'analyses'/ID;SCRATCH=TMP/'analyses'/ID
TEMPERATURES=[.75,.875,1.,1.125,1.25];BIASES=[.25,.375,.5,.625,.75];BETAS=[.4,.45,.5,.55,.6];SCALES=[.9,.95,1.,1.05,1.1];HOLES=[0,343,1000,3000]
COMPONENT=dict(id='c3_min512_protectFalse',connectivity=3,kind='minimum',value=512,protect_largest=False)
_Q=None


def parameters():return [dict(post_temperature=t,bias=b,beta=beta,volume_scale=s) for beta in BETAS for s in SCALES for t in TEMPERATURES for b in BIASES]


def cuts(counts):
    global _Q
    if _Q is None:
        p=np.linspace(1,0,65536,dtype=np.float64).clip(1e-7,1-1e-7);z=np.log(p)-np.log1p(-p)
        _Q=np.stack([1/(1+np.exp(-(z+b)/t)) for t in TEMPERATURES for b in BIASES])
    return fast_cuts(counts,_Q,BETAS,SCALES)


def predicted_structures(codes,valid,cut):
    # No GT enters either connected-component decision.
    filtered=component_apply((codes>=int(cut))&valid,COMPONENT)
    labels,n=ndi.label(~filtered,structure=ndi.generate_binary_structure(3,1));sizes=np.bincount(labels.ravel(),minlength=n+1)
    boundary=np.unique(np.concatenate([labels[0].ravel(),labels[-1].ravel(),labels[:,0].ravel(),labels[:,-1].ravel(),labels[:,:,0].ravel(),labels[:,:,-1].ravel()]))
    interior=np.ones(n+1,bool);interior[boundary]=False;interior[0]=False
    return filtered,labels,sizes,interior


def score_structures(structures,target):
    filtered,labels,sizes,interior=structures
    tp=int((filtered&target).sum());pred=int(filtered.sum());positive=int(target.sum());gt_counts=np.bincount(labels[target],minlength=len(sizes));out=[]
    for maximum in HOLES:
        fill=interior&(sizes<=maximum);num=tp+int(gt_counts[fill].sum());den=pred+int(sizes[fill].sum())+positive
        out.append(float(np.float32(2*num)/np.float32(den)) if den else 1.)
    return np.array(out,np.float32)


@torch.inference_mode()
def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
    index=read(CACHE/'index.json');images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS));base=read(R/'candidates/0020_brain_domain/RECIPE.json')
    one=sum(int(np.prod(g['extent'])) for g in index['geometry'].values())*2;required=one+625*4*270*4+64*2**20
    extra_f4=read(R/'analyses/f4_0000_brain_phase/PLAN.json')['extra_scratch_upper_bound_bytes']
    assert read(R/'INTENSITY_BUDGET_GATE_POSTPUBLICATION.json')['worst_case_total_bytes']+extra_f4+required<300*2**30
    assert stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required<300*2**30
    write(FOLDER/'PLAN.json',dict(at=now(),parameters=parameters(),holes=HOLES,component=COMPONENT,probability_source='0020 case24/T4, brainmargin0',base_recipe_sha256=sha(R/'candidates/0020_brain_domain/RECIPE.json'),source_sha256=sha(Path(__file__)),configurations=2500,GT_input_to_prediction=False,whole_native_crop_GT=True,neural_forwards_added=0,test_tuned=True))
    numeric_mode();torch.cuda.set_per_process_memory_fraction(.15);points={};files=[];start=time.monotonic()
    for i,cid in enumerate(index['case_ids']):
        guard();image=np.array(images[i][slices(index['geometry'][cid])],np.float32)
        p=probability_for(cid,base['aggregation']['base'],dict(handler='case_consistency_v1',core_source='r1_0000_core_mirror8'),torch.from_numpy(image).cuda());points[cid]=list(map(int,np.unravel_index(int(p.argmax()),p.shape)))
        code=(p.clamp(0,1)*65535+.5).long().cpu().numpy().astype(np.uint16);path=SCRATCH/'codes'/(cid+'.npy');save_array(path,code);files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        if i==0:
            valid=supports(image)[0];thresholds=cuts(np.bincount(code[valid],minlength=65536));probes=np.unique(thresholds)[::max(1,len(np.unique(thresholds))//16)][:16]
            started=time.monotonic()
            for cut in probes:predicted_structures(code,valid,int(cut))
            seconds=time.monotonic()-started;worst_seconds=seconds/max(len(probes),1)*625*270/8*1.5
            write(FOLDER/'RUNTIME_GATE.json',dict(at=now(),state='passed' if worst_seconds<5400 else 'over_cpu_budget',probes=len(probes),seconds=seconds,worst_case_projected_seconds=worst_seconds,workers=8,max_seconds=5400,GT_access=False))
            assert worst_seconds<5400,'CPU budget gate failed before full run'
            control=predicted_structures(code,valid,int(thresholds[312]));mask=control[0]|(control[3]&(control[2]<=1000))[control[1]]
            old=np.load(R/'candidates/0023_brain_holes/predictions'/(cid+'.npz'));expected=np.unpackbits(old['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
            assert np.array_equal(mask,expected) and points[cid]==old['point'].tolist()
            # Check the compact TP/FP/FN scoring against independently materialized masks.
            synthetic_target=(np.indices(code.shape).sum(0)%7==0)
            values=score_structures(control,synthetic_target)
            for hi,maximum in enumerate(HOLES):
                direct=binary_apply(control[0],control[0],dict(source='selected',operation='fill_holes',value=maximum),{'selected':control[0]});den=int(direct.sum())+int(synthetic_target.sum());score=float(np.float32(2*int((direct&synthetic_target).sum()))/np.float32(den)) if den else 1.
                assert score==float(values[hi])
            write(FOLDER/'GATE.json',dict(at=now(),state='passed',control_first_case_mask_xor=0,scoring_matches_direct_masks=True,unlabeled_cuts_only=True,GT_domain_unchanged=True))
        if i%10==0:write(FOLDER/'PREPARE_STATUS.json',dict(at=now(),completed=i+1,total=270,seconds=time.monotonic()-start))
    write(FOLDER/'POINTS.json',points);write(FOLDER/'CODE_FILES.json',files);write(FOLDER/'PREPARE_DONE.json',dict(at=now(),state='complete',scans=270,neural_forwards_added=0,seconds=time.monotonic()-start))


def case_task(slot):
    index=read(CACHE/'index.json');cid=index['case_ids'][slot];sl=slices(index['geometry'][cid])
    image=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS))[slot][sl];target=np.array(np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))[slot][sl]>0);code=np.load(SCRATCH/'codes'/(cid+'.npy'));valid=supports(image)[0]
    thresholds=cuts(np.bincount(code[valid],minlength=65536));unique,inverse=np.unique(thresholds,return_inverse=True);values=[]
    for threshold in unique:values.append(score_structures(predicted_structures(code,valid,int(threshold)),target))
    result=np.array(values,np.float32)[inverse]
    original=read(R/'candidates/0023_brain_holes/RAW_TEST270.json')['per_case'][slot];assert original['case_id']==cid and float(result[312,2])==original['locator_dsc']
    return slot,result,thresholds,len(unique)


def sweep():
    index=read(CACHE/'index.json');records=read(R/'candidates/0023_brain_holes/RAW_TEST270.json')['per_case'];matrix=np.zeros((625,4,270),np.float32);cut_matrix=np.zeros((625,270),np.uint16);unique_counts=[];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=mp.get_context('spawn')) as pool:
        for done,(slot,values,thresholds,unique) in enumerate(pool.map(case_task,range(270)),1):
            guard();matrix[:,:,slot]=values;cut_matrix[:,slot]=thresholds;unique_counts.append(unique)
            if done%5==0:write(FOLDER/'STATUS.json',dict(at=now(),completed=done,total=270,seconds=time.monotonic()-start))
    subjects=sorted({r['subject_id'] for r in records});groups=[[i for i,r in enumerate(records) if r['subject_id']==s] for s in subjects];mean=matrix.mean(2,dtype=np.float64);sub=np.stack([matrix[:,:,g].mean(2,dtype=np.float64) for g in groups],2).mean(2);params=parameters();ranking=[]
    for pi,param in enumerate(params):
        for hi,maximum in enumerate(HOLES):ranking.append(dict(parameter_index=pi,hole_index=hi,parameters=param,hole_maximum=maximum,component=COMPONENT,scan_dice=float(mean[pi,hi]),subject_dice=float(sub[pi,hi]),point_hits=260,neural_case_forwards=8))
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['parameter_index'],r['hole_index']))
    save_array(FOLDER/'ALL_PER_CASE_DICE.npy',matrix);save_array(FOLDER/'ALL_CUTS.npy',cut_matrix)
    write(FOLDER/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,test_tuned=True))
    write(FOLDER/'DONE.json',dict(at=now(),state='complete',configurations=2500,best=ranking[0],seconds=time.monotonic()-start,unique_cut_counts=unique_counts,all270_current_control_scores_exact=True,dice_sha256=sha(FOLDER/'ALL_PER_CASE_DICE.npy'),cuts_sha256=sha(FOLDER/'ALL_CUTS.npy'),neural_forwards_added=0,test_tuned=True));print(ranking[0])

if __name__=='__main__':
    if not (FOLDER/'PREPARE_DONE.json').exists():prepare()
    sweep()
