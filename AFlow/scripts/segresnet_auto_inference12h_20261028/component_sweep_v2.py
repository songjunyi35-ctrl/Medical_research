"""Connected-component combinations for any audited globally selected base mask."""
import argparse
import concurrent.futures
import multiprocessing as mp
import numpy as np
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import definitions,choose
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import unpack,LABEL_DIR


def case_task(args):
    slot,base=args;index=read(LABEL_DIR/'index.json');cid=index['case_ids'][slot]
    labels=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS));y=np.array(labels[slot][slices(index['geometry'][cid])]>0)
    mask=unpack(Path(base)/'predictions'/(cid+'.npz'));cache={};scores=[]
    for recipe in definitions():
        conn=recipe['connectivity']
        if conn not in cache:
            components,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,conn));sizes=np.bincount(components.ravel(),minlength=n+1)
            positive=np.bincount(components[y],minlength=n+1);cache[conn]=(sizes,positive)
        sizes,positive=cache[conn];keep=choose(sizes,recipe);tp=int(positive[keep].sum());den=int(sizes[keep].sum())+int(y.sum())
        scores.append(float(np.float32(2*tp)/np.float32(den)) if den else 1.)
    return slot,np.array(scores,np.float32)


def main(base,identifier):
    base=Path(base).resolve();assert (base/'RESULT.json').exists()
    folder=R/'analyses'/identifier;folder.mkdir(parents=True,exist_ok=True);recipes=definitions()
    raw=read(base/'RAW_TEST270.json')['per_case'];index=read(LABEL_DIR/'index.json');assert [r['case_id'] for r in raw]==index['case_ids']
    result=read(base/'RESULT.json');cost=result.get('locator_forwards',5)
    write(folder/'FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),component_definition_sha256=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/component_sweep_v1.py'),base_folder=str(base.relative_to(ROOT)),base_raw_sha256=sha(base/'RAW_TEST270.json'),recipes=recipes,test_tuned=True,GT_input_to_choose=False))
    start=time.monotonic();dice=np.zeros((len(recipes),270),np.float32)
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for done,(slot,values) in enumerate(pool.map(case_task,[(i,str(base)) for i in range(270)],chunksize=1),1):
            guard();dice[:,slot]=values
            if done%25==0:write(folder/'STATUS.json',dict(at=now(),completed=done,total=270,seconds=time.monotonic()-start))
    assert np.array_equal(dice[0],np.array([r['locator_dsc'] for r in raw],np.float32))
    subjects=sorted({r['subject_id'] for r in raw});groups=[[i for i,r in enumerate(raw) if r['subject_id']==s] for s in subjects]
    mean=dice.mean(1,dtype=np.float64);sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],1).mean(1);hits=sum(r['automatic_point_hits_gt'] for r in raw)
    ranking=[dict(configuration_index=i,recipe=recipe,scan_dice=float(mean[i]),subject_dice=float(sub[i]),point_hits=hits,neural_case_forwards=cost) for i,recipe in enumerate(recipes)]
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['configuration_index']));np.save(folder/'ALL_PER_CASE_DICE.npy',dice)
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,base_folder=str(base.relative_to(ROOT)),test_tuned=True,model_forwards_added=0))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=len(recipes),seconds=time.monotonic()-start,test_tuned=True,base_folder=str(base.relative_to(ROOT)),dice_sha256=sha(folder/'ALL_PER_CASE_DICE.npy')))
    print(ranking[0])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--id',required=True);a=p.parse_args();main(a.base,a.id)
