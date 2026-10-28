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

ID='f3_0000_brain_decoder_component_beam'
FOLDER=R/'analyses'/ID
SCRATCH=TMP/'analyses'/ID
CAL='f1_1000_brain_domain_calibration'

def prepare():
    import hashlib
    from scripts.segresnet_auto_inference12h_20261028.calibration_brain_domain_v1 import parameters
    FOLDER.mkdir(parents=True,exist_ok=True)
    source=R/'analyses'/CAL/'RANKING.json';source_best=read(source)['best_by_recipe'];params=parameters()
    assert (R/'analyses/f2_0000_brain_component_beam/PREPARE_DONE.json').exists()
    write(FOLDER/'SELECTION_FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),ranking_sha256=sha(source),distinct_cut_profiles_per_support_family=8,selection='global scan means, stable parameter-index tie; identical270 code-cut vectors deduplicated',test_tuned=True,GT_input_to_single_case_prediction=False))
    index=read(CACHE/'index.json');beam=[]
    for best in source_best:
        ri=best['recipe_index'];arrays=[np.load(TMP/'analyses'/CAL/'cases'/str(ri)/(cid+'.npz')) for cid in index['case_ids']]
        cuts=np.stack([v['cuts'] for v in arrays],1);dice=np.stack([v['dice'] for v in arrays],1)
        means=dice.mean(1,dtype=np.float64);order=np.argsort(-means,kind='stable');seen=set();selected=[]
        for pi in order:
            signature=hashlib.sha256(cuts[pi].tobytes()).hexdigest()
            if signature in seen:continue
            seen.add(signature);item=dict(best);item.update(parameter_index=int(pi),parameters=params[int(pi)],scan_dice=float(means[pi]),cut_vector_sha256=signature)
            # Original source subject metric belongs to its original winning cut only.
            item.pop('subject_dice',None);selected.append(item)
            if len(selected)==8:break
        assert len(selected)==8;beam.extend(selected)
        write(FOLDER/'SELECTION_STATUS.json',dict(at=now(),families=len(beam)//8,total_families=16))
        for v in arrays:v.close()
    write(FOLDER/'PLAN.json',dict(at=now(),beam=beam,component_rules=definitions(),source_sha256=sha(Path(__file__)),calibration_ranking_sha256=sha(source),configurations=len(beam)*len(definitions()),GT_domain='entire original native crop',GT_input_to_prediction=False,test_tuned=True,model_forwards_added=0))
    assert len(beam)==128
    write(FOLDER/'POINTS.json',read(R/'analyses/f2_0000_brain_component_beam/POINTS.json'))
    write(FOLDER/'PREPARE_DONE.json',dict(at=now(),scans=270,beam=128,model_forwards_added=0,additional_probability_maps=0))


def case_task(slot):
    plan=read(FOLDER/'PLAN.json');index=read(CACHE/'index.json');cid=index['case_ids'][slot];sl=slices(index['geometry'][cid])
    image=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS))[slot][sl]
    target=np.array(np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))[slot][sl]>0)
    valid=supports(image);codes={};all_scores=[];positives=int(target.sum())
    for base in plan['beam']:
        ri=base['recipe_index'];bi=ri//4
        if bi not in codes:codes[bi]=np.load(TMP/'analyses/f2_0000_brain_component_beam/codes'/str(bi)/(cid+'.npy'))
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
