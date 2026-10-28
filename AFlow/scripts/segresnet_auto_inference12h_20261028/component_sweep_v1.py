"""Global connected-component rules; selection observes270, predictor never sees GT."""
import concurrent.futures
import multiprocessing as mp
import csv
import numpy as np
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import unpack,LABEL_DIR


def definitions():
    out=[dict(id='identity',connectivity=3,kind='minimum',value=0,protect_largest=False)]
    for connectivity in [1,2,3]:
        for protect in [False,True]:
            for volume in [8,27,64,125,216,343,512,750,1000,1250,1500,2000,2500,3000,4000,6000,8000,12000,16000,24000,32000,48000,64000]:
                out.append(dict(id=f'c{connectivity}_min{volume}_protect{protect}',connectivity=connectivity,kind='minimum',value=volume,protect_largest=protect))
            for fraction in [.0005,.001,.002,.005,.01,.02,.03,.05,.075,.1,.15,.2,.3,.5,.75]:
                out.append(dict(id=f'c{connectivity}_rel{fraction}_protect{protect}',connectivity=connectivity,kind='relative',value=fraction,protect_largest=protect))
        for k in [1,2,3,4,5]:
            out.append(dict(id=f'c{connectivity}_largest{k}',connectivity=connectivity,kind='largest',value=k,protect_largest=False))
    return out


def choose(sizes,recipe):
    """Only predicted component sizes enter the deployment decision."""
    sizes=np.asarray(sizes).copy();sizes[0]=0;keep=np.zeros(len(sizes),bool)
    if recipe['kind']=='minimum':keep=sizes>=recipe['value']
    elif recipe['kind']=='relative':keep=sizes>=sizes.max(initial=0)*recipe['value']
    elif recipe['kind']=='largest':keep[np.argsort(sizes,kind='stable')[-recipe['value']:]]=True
    else:raise ValueError(recipe)
    if recipe['protect_largest'] and sizes.max(initial=0)>0:keep[int(sizes.argmax())]=True
    keep[0]=False;return keep


def apply(mask,recipe):
    components,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,recipe['connectivity']))
    sizes=np.bincount(components.ravel(),minlength=n+1)
    return choose(sizes,recipe)[components]


def case_task(slot):
    index=read(LABEL_DIR/'index.json');cid=index['case_ids'][slot]
    labels=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS));y=np.array(labels[slot][slices(index['geometry'][cid])]>0)
    mask=unpack(OLD/'predictions'/(cid+'.npz'));cache={};scores=[]
    for recipe in definitions():
        conn=recipe['connectivity']
        if conn not in cache:
            components,n=ndi.label(mask,structure=ndi.generate_binary_structure(3,conn))
            sizes=np.bincount(components.ravel(),minlength=n+1)
            positives=np.bincount(components[y],minlength=n+1)
            cache[conn]=(sizes,positives)
        sizes,positives=cache[conn];keep=choose(sizes,recipe)
        # GT is only used after the keep decision to evaluate its counts.
        tp=int(positives[keep].sum());pred=int(sizes[keep].sum());den=pred+int(y.sum())
        scores.append(float(np.float32(2*tp)/np.float32(den)) if den else 1.)
    return slot,np.array(scores,np.float32)


def main():
    folder=R/'analyses/1100_component_sweep';folder.mkdir(parents=True,exist_ok=True)
    recipes=definitions();write(folder/'FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),recipes=recipes,
          base_mask_folder=str((OLD/'predictions').relative_to(ROOT)),test_tuned=True,GT_input_to_choose=False))
    started=time.monotonic();dice=np.zeros((len(recipes),270),np.float32)
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for done,(slot,values) in enumerate(pool.map(case_task,range(270),chunksize=1),1):
            guard();dice[:,slot]=values
            if done%25==0:write(folder/'STATUS.json',dict(at=now(),completed=done,total=270,seconds=time.monotonic()-started))
    rows=read(OLD/'RAW_TEST270.json')['per_case'];subjects=sorted({r['subject_id'] for r in rows});groups=[[i for i,r in enumerate(rows) if r['subject_id']==s] for s in subjects]
    means=dice.mean(1,dtype=np.float64);sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],1).mean(1)
    ranking=[dict(configuration_index=i,recipe=recipe,scan_dice=float(means[i]),subject_dice=float(sub[i]),point_hits=255,neural_case_forwards=5) for i,recipe in enumerate(recipes)]
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['configuration_index']))
    np.save(folder/'ALL_PER_CASE_DICE.npy',dice)
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,test_tuned=True,model_forwards_added=0))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=len(recipes),seconds=time.monotonic()-started,best=ranking[0],test_tuned=True,dice_sha256=sha(folder/'ALL_PER_CASE_DICE.npy')))
if __name__=='__main__':main()
