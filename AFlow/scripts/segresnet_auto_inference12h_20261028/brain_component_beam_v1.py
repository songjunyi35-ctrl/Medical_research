"""Cross every image-support family winner with fixed component rules."""
import argparse, concurrent.futures, multiprocessing as mp
import numpy as np
import torch
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import aggregate
from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import definitions,choose

ID='f2_0000_brain_component_beam'
FOLDER=R/'analyses'/ID
SCRATCH=TMP/'analyses'/ID
CAL='f1_1000_brain_domain_calibration'

@torch.inference_mode()
def prepare():
    FOLDER.mkdir(parents=True,exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
    source=R/'analyses'/CAL/'RANKING.json';beam=read(source)['best_by_recipe']
    assert len(beam)==16
    write(FOLDER/'PLAN.json',dict(at=now(),beam=beam,component_rules=definitions(),source_sha256=sha(Path(__file__)),calibration_ranking_sha256=sha(source),configurations=len(beam)*len(definitions()),GT_domain='entire original native crop',GT_input_to_prediction=False,test_tuned=True,model_forwards_added=0))
    # Four uint16 fields, one per base, never per-calibration duplicated.
    required=4*270*np.prod(CANVAS)*2+64*2**20
    assert stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required<300*2**30
    from scripts.segresnet_inference_search_20261028.common import numeric_mode
    numeric_mode();rows,geometry,images,labels=data();torch.cuda.set_per_process_memory_fraction(.15)
    start=time.monotonic();points={};files=[]
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];sl=slices(geometry[cid]);image=np.array(images[slot][sl],np.float32)
        logits={i:torch.from_numpy(np.load(TMP/'groups/r1_0000_core_mirror8/atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)}
        points[cid]={}
        for bi in range(4):
            recipe=next(b['recipe']['base'] for b in beam if b['recipe_index']//4==bi)
            p,point_p=aggregate(logits,recipe,torch.from_numpy(image).cuda())
            point=list(map(int,np.unravel_index(int(point_p.argmax()),point_p.shape)))
            points[cid][str(bi)]=point
            path=SCRATCH/'codes'/str(bi)/(cid+'.npy')
            codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy().astype(np.uint16)
            save_array(path,codes);files.append(dict(path=str(path),sha256=sha(path)))
        if slot==0:
            checked_slot,checked_scores=case_task(0);assert checked_slot==0 and len(checked_scores)==3904
            write(FOLDER/'GATE.json',dict(at=now(),state='passed',all16_identity_scores_exact=True,component_keep_GT_input=False,full_GT_denominator=True,source_sha256=sha(Path(__file__))))
        if slot%10==0:write(FOLDER/'PREPARE_STATUS.json',dict(at=now(),completed=slot+1,total=270,seconds=time.monotonic()-start))
    write(FOLDER/'POINTS.json',points);write(FOLDER/'CODE_FILES.json',files)
    write(FOLDER/'PREPARE_DONE.json',dict(at=now(),scans=270,fields=4,model_forwards_added=0,GT_input=False,seconds=time.monotonic()-start))


def case_task(slot):
    plan=read(FOLDER/'PLAN.json');index=read(CACHE/'index.json');cid=index['case_ids'][slot];sl=slices(index['geometry'][cid])
    image=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS))[slot][sl]
    target=np.array(np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))[slot][sl]>0)
    valid=supports(image);codes={};all_scores=[];positives=int(target.sum())
    for base in plan['beam']:
        ri=base['recipe_index'];bi=ri//4
        if bi not in codes:codes[bi]=np.load(SCRATCH/'codes'/str(bi)/(cid+'.npy'))
        previous=np.load(TMP/'analyses'/CAL/'cases'/str(ri)/(cid+'.npz'))
        cut=int(previous['cuts'][base['parameter_index']]);mask=(codes[bi]>=cut)&valid[base['recipe']['margin']]
        cache={};values=[]
        for rule in plan['component_rules']:
            conn=rule['connectivity']
            if conn not in cache:
                cc,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,conn));sizes=np.bincount(cc.ravel(),minlength=n+1)
                gt_counts=np.bincount(cc[target],minlength=n+1);cache[conn]=(sizes,gt_counts)
            sizes,gt_counts=cache[conn];keep=choose(sizes,rule)
            tp=int(gt_counts[keep].sum());den=int(sizes[keep].sum())+positives
            values.append(float(np.float32(2*tp)/np.float32(den)) if den else 1.)
        assert values[0]==float(previous['dice'][base['parameter_index']]),(cid,ri)
        all_scores.extend(values)
    return slot,np.array(all_scores,np.float32)


def sweep():
    assert (FOLDER/'PREPARE_DONE.json').exists();plan=read(FOLDER/'PLAN.json');index=read(CACHE/'index.json')
    dice=np.zeros((plan['configurations'],270),np.float32);start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for done,(slot,values) in enumerate(pool.map(case_task,range(270)),1):
            guard();dice[:,slot]=values
            if done%10==0:write(FOLDER/'STATUS.json',dict(at=now(),completed=done,total=270,seconds=time.monotonic()-start))
    rows,_,_,labels=data();subjects=sorted({r['subject_id'] for r in rows});groups=[[i for i,r in enumerate(rows) if r['subject_id']==s] for s in subjects]
    means=dice.mean(1,dtype=np.float64);sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],1).mean(1)
    points=read(FOLDER/'POINTS.json');hits={}
    for bi in range(4):
        hits[bi]=sum(bool(labels[i][slices(index['geometry'][r['case_id']])][tuple(points[r['case_id']][str(bi)])]) for i,r in enumerate(rows))
    ranking=[];ci=0
    for base in plan['beam']:
        for rule in plan['component_rules']:
            ranking.append(dict(configuration_index=ci,base=base,component=rule,scan_dice=float(means[ci]),subject_dice=float(sub[ci]),point_hits=hits[base['recipe_index']//4],neural_case_forwards=base['recipe']['neural_case_forwards']));ci+=1
    ranking.sort(key=lambda x:(-x['scan_dice'],-x['subject_dice'],x['configuration_index']))
    save_array(FOLDER/'ALL_PER_CASE_DICE.npy',dice)
    write(FOLDER/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,test_tuned=True))
    write(FOLDER/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=len(ranking),seconds=time.monotonic()-start,test_tuned=True,model_forwards_added=0,dice_sha256=sha(FOLDER/'ALL_PER_CASE_DICE.npy')))
    print(ranking[0])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if not (FOLDER/'PREPARE_DONE.json').exists():prepare()
    if not a.prepare_only:sweep()
