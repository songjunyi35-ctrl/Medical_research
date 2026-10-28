"""CPU fallback: global postprocessing of retained predictions, no neural forward."""
import concurrent.futures
import multiprocessing as mp
import numpy as np
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028.common import *

BASE=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270'
LABEL_DIR=TMP/'cpu_native'
FOLDER=R/'analyses/1000_cpu_binary'
_LABELS=None
_INDEX=None


def recipes():
    out=[]
    for source in ['selected','union','intersection']:
        out.append(dict(id=source,source=source,operation='identity'))
        for conn in [1,3]:
            for radius in [1,2]:
                for op in ['opening','closing','erosion','dilation']:
                    out.append(dict(id=f'{source}_{op}_c{conn}_r{radius}',source=source,operation=op,connectivity=conn,radius=radius))
        for op in ['closing_opening','opening_closing']:
            out.append(dict(id=f'{source}_{op}',source=source,operation=op,connectivity=3,radius=1))
    for volume in [8,27,64,125,343,1000]:out.append(dict(id=f'min_component{volume}',source='selected',operation='min_component',value=volume))
    for fraction in [.001,.005,.01]:out.append(dict(id=f'rel_component{fraction}',source='selected',operation='relative_component',value=fraction))
    for k in [1,2,3]:out.append(dict(id=f'largest{k}',source='selected',operation='largest',value=k))
    for volume in [8,27,64,125,343,1000,3375]:out.append(dict(id=f'fill_small_hole{volume}',source='selected',operation='fill_holes',value=volume))
    for sigma in [.5,.75,1.,1.25,1.5,2.]:
        for threshold in np.round(np.arange(.2,.8001,.05),6):
            out.append(dict(id=f'binary_gaussian_s{sigma}_t{threshold}',source='selected',operation='gaussian',sigma=sigma,threshold=float(threshold)))
    for weight in [.1,.25,.5,.75,.9]:
        for shift in [-1.,-.5,0.,.5,1.]:out.append(dict(id=f'distance_blend{weight}_s{shift}',source='distance',operation='distance',weight=weight,shift=shift))
    return out


def unpack(path):
    saved=np.load(path);shape=tuple(saved['shape'])
    return np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)


def signed_distance(mask):
    if not mask.any():return np.full(mask.shape,-max(mask.shape),np.float32)
    if mask.all():return np.full(mask.shape,max(mask.shape),np.float32)
    return (ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)).astype(np.float32)


def apply(selected,original,recipe,cache):
    source=recipe['source']
    if source not in cache:
        if source=='union':cache[source]=selected|original
        elif source=='intersection':cache[source]=selected&original
        elif source=='selected':cache[source]=selected
    op=recipe['operation']
    if op=='distance':
        if 'sdf_selected' not in cache:
            cache['sdf_selected']=signed_distance(selected);cache['sdf_original']=signed_distance(original)
        return (recipe['weight']*cache['sdf_original']+(1-recipe['weight'])*cache['sdf_selected'])>recipe['shift']
    mask=cache[source]
    if op=='identity':return mask
    if op in ['opening','closing','erosion','dilation','opening_closing','closing_opening']:
        structure=ndi.iterate_structure(ndi.generate_binary_structure(3,recipe['connectivity']),recipe['radius'])
        result=mask
        for step in op.split('_'):result=getattr(ndi,'binary_'+step)(result,structure=structure,border_value=0)
        return result
    if op=='gaussian':
        key=f"gaussian{recipe['sigma']}"
        if key not in cache:cache[key]=ndi.gaussian_filter(mask.astype(np.float32),recipe['sigma'],mode='nearest')
        return cache[key]>recipe['threshold']
    if op in ['min_component','relative_component','largest']:
        if 'foreground_components' not in cache:
            labels,n=ndi.label(mask,structure=np.ones((3,3,3),np.uint8));sizes=np.bincount(labels.ravel());sizes[0]=0
            cache['foreground_components']=(labels,sizes)
        labels,sizes=cache['foreground_components'];keep=np.zeros(len(sizes),bool)
        if op=='min_component':keep=sizes>=recipe['value']
        elif op=='relative_component':keep=sizes>=sizes.max(initial=0)*recipe['value']
        else:keep[np.argsort(sizes)[-recipe['value']:]]=True
        keep[0]=False;return keep[labels]
    if op=='fill_holes':
        if 'background_components' not in cache:
            labels,n=ndi.label(~mask,structure=ndi.generate_binary_structure(3,1));sizes=np.bincount(labels.ravel())
            boundary=np.unique(np.concatenate([labels[0].ravel(),labels[-1].ravel(),labels[:,0].ravel(),labels[:,-1].ravel(),labels[:,:,0].ravel(),labels[:,:,-1].ravel()]))
            interior=np.ones(len(sizes),bool);interior[boundary]=False;interior[0]=False
            cache['background_components']=(labels,sizes,interior)
        labels,sizes,interior=cache['background_components'];fill=interior&(sizes<=recipe['value'])
        return mask|fill[labels]
    raise ValueError(op)


def case_task(slot):
    global _LABELS,_INDEX
    if _INDEX is None:
        _INDEX=read(LABEL_DIR/'index.json');_LABELS=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))
    cid=_INDEX['case_ids'][slot];g=_INDEX['geometry'][cid]
    selected=unpack(OLD/'predictions'/(cid+'.npz'));original=unpack(BASE/'predictions'/(cid+'.npz'))
    target=np.array(_LABELS[slot][slices(g)]>0)
    values=[];cache={'selected':selected}
    for recipe in recipes():
        mask=apply(selected,original,recipe,cache)
        tp=int((mask&target).sum());pred=int(mask.sum());den=pred+int(target.sum())
        values.append(float(np.float32(2*tp)/np.float32(den)) if den else 1.)
    path=TMP/'cpu_binary_cases'/(cid+'.npy');save_array(path,np.array(values,np.float32))
    return slot,cid


def main():
    import time
    guard();FOLDER.mkdir(parents=True,exist_ok=True)
    expected=read(OLD/'CACHE_READY.json')['payload_sha256']
    assert sha(LABEL_DIR/'labels.u8')==expected['labels'] and sha(LABEL_DIR/'index.json')==expected['index']
    manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    assert read(LABEL_DIR/'index.json')['case_ids']==[d['case_id'] for d in rows]
    definitions=recipes();write(FOLDER/'PLAN.json',dict(at=now(),recipes=definitions,test_tuned=True,model_forward=False,point_source='unchanged selected five-view argmax'))
    pending=[i for i,d in enumerate(rows) if not (TMP/'cpu_binary_cases'/(d['case_id']+'.npy')).exists()]
    done=270-len(pending);begun=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for slot,cid in pool.map(case_task,pending,chunksize=1):
            guard();done+=1
            if done%10==0 or done==270:write(R/'CPU_FALLBACK_STATUS.json',dict(at=now(),completed=done,total=270,configurations=len(definitions),seconds=time.monotonic()-begun,GPU_used=False,trigger='single_GPU_resource_outage'))
    dice=np.stack([np.load(TMP/'cpu_binary_cases'/(d['case_id']+'.npy')) for d in rows],axis=1)
    baseline=np.array([d['locator_dsc'] for d in read(OLD/'RAW_TEST270.json')['per_case']])
    assert np.array_equal(dice[0],baseline)
    subjects=sorted({d['subject_id'] for d in rows});groups=[[i for i,d in enumerate(rows) if d['subject_id']==s] for s in subjects]
    means=dice.mean(1,dtype=np.float64);sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],1).mean(1)
    ranking=[dict(recipe=recipe,scan_dice=float(means[i]),subject_dice=float(sub[i]),point_hits=255,configuration_index=i,
                  neural_case_forwards=6 if recipe['source']!='selected' else 5) for i,recipe in enumerate(definitions)]
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],d['neural_case_forwards'],d['recipe']['id']))
    np.save(FOLDER/'ALL_PER_CASE_DICE.npy',dice)
    write(FOLDER/'RANKING.json',dict(at=now(),best=ranking[0],all=ranking,configurations=len(definitions),test_tuned=True,model_forwards_added=0,
          historical_baseline=.7809951982288449,unchanged_baseline_exact=True,seconds=time.monotonic()-begun))
    write(FOLDER/'DONE.json',dict(at=now(),state='complete',best=ranking[0],model_forwards_added=0,test_tuned=True))


if __name__=='__main__':main()
