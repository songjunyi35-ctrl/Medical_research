"""Signed-distance shape fusion of eight predictions; no labels enter fusion."""
import concurrent.futures,multiprocessing as mp,csv
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import signed_distance,LABEL_DIR
LABEL_DIR=CACHE
_INPUTS=TMP/'shape_fusion_inputs'
_DATA=None
LEVELS=np.array([-3.,-2.,-1.5,-1.,-.75,-.5,-.25,0.,.25,.5,.75,1.,1.5,2.,3.])


def definitions():
    rows=[]
    for threshold in ['t20','t50','t80']:
        for exponent in [-16.,-8.,0.,4.,8.,16.,32.]:rows.append(dict(id=f'{threshold}_sdf_agreement{exponent}',threshold=threshold,kind='weighted_mean',exponent=exponent))
        for mode in ['trim6','trim4','median','quantile25','quantile75','minimum','maximum']:rows.append(dict(id=f'{threshold}_sdf_{mode}',threshold=threshold,kind=mode))
    return rows


def unpack_case(path,threshold):
    with np.load(path) as data:
        shape=tuple(data['shape']);n=int(np.prod(shape));masks=np.unpackbits(data[threshold],axis=1)[:,:n].reshape(8,*shape).astype(bool)
        reference=np.unpackbits(data[threshold+'_reference'])[:n].reshape(shape).astype(bool)
        point=data['point'].tolist()
    return masks,reference,point


def field(masks,reference,recipe,cache):
    if 'distance' not in cache:cache['distance']=np.stack([signed_distance(mask) for mask in masks])
    distance=cache['distance'];mode=recipe['kind']
    if mode=='weighted_mean':
        tp=(masks&reference[None]).reshape(8,-1).sum(1).astype(np.float64);den=masks.reshape(8,-1).sum(1)+reference.sum()
        score=2*tp/np.maximum(den,1);value=score*recipe['exponent'];weights=np.exp(value-value.max());weights/=weights.sum()
        result=np.zeros_like(distance[0])
        for i in range(8):result+=distance[i]*np.float32(weights[i])
        return result
    if 'sorted' not in cache:cache['sorted']=np.sort(distance,axis=0)
    ordered=cache['sorted']
    if mode=='trim6':return ordered[1:7].mean(0)
    if mode=='trim4':return ordered[2:6].mean(0)
    if mode=='median':return (ordered[3]+ordered[4])*.5
    if mode=='quantile25':return ordered[1]*.25+ordered[2]*.75
    if mode=='quantile75':return ordered[5]*.75+ordered[6]*.25
    if mode=='minimum':return ordered[0]
    if mode=='maximum':return ordered[-1]
    raise ValueError(mode)


def case_task(slot):
    global _DATA
    if _DATA is None:
        index=read(LABEL_DIR/'index.json');labels=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS));_DATA=index,labels
    index,labels=_DATA;cid=index['case_ids'][slot];target=np.array(labels[slot][slices(index['geometry'][cid])]>0)
    values=[];previous=None;start=time.monotonic()
    for recipe in definitions():
        if recipe['threshold']!=previous:
            masks,reference,point=unpack_case(_INPUTS/(cid+'.npz'),recipe['threshold']);cache={};previous=recipe['threshold']
        p=field(masks,reference,recipe,cache)
        bins=np.searchsorted(LEVELS,p.ravel(),side='left');hist=np.bincount(bins+target.ravel().astype(np.int64)*16,minlength=32).reshape(2,16)
        tail=np.cumsum(hist[:,::-1],axis=1,dtype=np.int64)[:,::-1];tp=tail[1,1:];fp=tail[0,1:]
        score=(2*tp/np.maximum(tp+fp+target.sum(),1)).astype(np.float32);values.append(score)
    return slot,np.stack(values),bool(target[tuple(point)]),time.monotonic()-start


def main():
    assert read(R/'SHAPE_FUSION_INPUT_READY.json')['state']=='complete';identifier='1300_shape_distance_fusion';folder=R/'analyses'/identifier;folder.mkdir(parents=True,exist_ok=True)
    recipes=definitions();assert len(recipes)==42
    write(folder/'FREEZE.json',dict(at=now(),recipes=recipes,levels=LEVELS.tolist(),source_sha256=sha(Path(__file__)),input_receipt_sha256=sha(R/'SHAPE_FUSION_INPUT_READY.json'),GT_input_to_field=False,all270_same_global_rule=True,test_tuned=True,model_forwards_added=0))
    dice=np.zeros((42,15,270),np.float32);hits=np.zeros(270,bool);start=time.monotonic();times=[]
    with concurrent.futures.ProcessPoolExecutor(4,mp_context=mp.get_context('spawn')) as pool:
        for done,(slot,value,hit,elapsed) in enumerate(pool.map(case_task,range(270),chunksize=1),1):
            guard();dice[...,slot]=value;hits[slot]=hit;times.append(elapsed)
            if done%5==0:write(folder/'STATUS.json',dict(at=now(),completed=done,total=270,last_case_seconds=elapsed,seconds=time.monotonic()-start))
    np.save(folder/'ALL_PER_CASE_DICE.npy',dice)
    raw=read(OLD/'RAW_TEST270.json')['per_case'];groups=[[i for i,r in enumerate(raw) if r['subject_id']==s] for s in sorted({r['subject_id'] for r in raw})]
    mean=dice.mean(-1,dtype=np.float64);subject=np.stack([dice[...,g].mean(-1,dtype=np.float64) for g in groups],-1).mean(-1);ranking=[]
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','level','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for li,level in enumerate(LEVELS):
                ranking.append(dict(recipe_index=ri,level_index=li,recipe=recipe,level=float(level),scan_dice=float(mean[ri,li]),subject_dice=float(subject[ri,li]),point_hits=int(hits.sum()),neural_case_forwards=8));w.writerow([recipe['id'],level,mean[ri,li],subject[ri,li],int(hits.sum())])
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['recipe_index'],r['level_index']))
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,test_tuned=True))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=630,seconds=time.monotonic()-start,median_case_seconds=float(np.median(times)),dice_sha256=sha(folder/'ALL_PER_CASE_DICE.npy'),test_tuned=True,model_forwards_added=0))
    print(ranking[0])
if __name__=='__main__':main()
