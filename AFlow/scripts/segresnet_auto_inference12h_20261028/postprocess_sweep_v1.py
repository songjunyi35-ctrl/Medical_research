"""Global binary morphology and cavity rules composed with one audited base."""
import argparse,concurrent.futures,multiprocessing as mp
import numpy as np
from scipy import ndimage as ndi
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import recipes as original_recipes,unpack,LABEL_DIR
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_fast_v1 import fast_apply


def definitions():
    out=original_recipes()
    for conn in [1,3]:
        for maximum in [5000,8000,16000,32000,64000,2**31-1]:
            out.append(dict(id=f'fill_enclosed_c{conn}_max{maximum}',source='selected',operation='enclosed_cavity',connectivity=conn,value=maximum,relative=False))
        for fraction in [.025,.05,.1,.2,.5,1.]:
            out.append(dict(id=f'fill_enclosed_c{conn}_relative{fraction}',source='selected',operation='enclosed_cavity',connectivity=conn,value=fraction,relative=True))
    return out


def apply_global(selected,original,recipe,cache):
    if recipe['operation']!='enclosed_cavity':return fast_apply(selected,original,recipe,cache)
    conn=recipe['connectivity'];key=f'cavity_components_{conn}'
    if key not in cache:
        components,n=ndi.label(~selected,structure=ndi.generate_binary_structure(3,conn));sizes=np.bincount(components.ravel(),minlength=n+1)
        exterior=np.unique(np.concatenate([components[0].ravel(),components[-1].ravel(),components[:,0].ravel(),components[:,-1].ravel(),components[:,:,0].ravel(),components[:,:,-1].ravel()]))
        interior=np.ones(n+1,bool);interior[exterior]=False;interior[0]=False;cache[key]=(components,sizes,interior)
    components,sizes,interior=cache[key]
    maximum=selected.sum()*recipe['value'] if recipe['relative'] else recipe['value']
    fill=interior&(sizes<=maximum);return selected|fill[components]


def inferred_views(recipe_path):
    recipe=read(recipe_path)
    if 'aggregation' in recipe:return recipe['aggregation']['views']
    if 'base_recipe_path' in recipe:return inferred_views(ROOT/recipe['base_recipe_path'])
    raise ValueError('unknown base recipe views')


def case_task(args):
    slot,base=args;index=read(LABEL_DIR/'index.json');cid=index['case_ids'][slot]
    labels=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS));target=np.array(labels[slot][slices(index['geometry'][cid])]>0)
    selected=unpack(Path(base)/'predictions'/(cid+'.npz'));original=unpack(R/'control_from_core_flip0/predictions'/(cid+'.npz'))
    cache={'selected':selected};positive=int(target.sum());scores=[]
    for recipe in definitions():
        mask=apply_global(selected,original,recipe,cache);tp=int((mask&target).sum());den=int(mask.sum())+positive
        scores.append(float(np.float32(2*tp)/np.float32(den)) if den else 1.)
    return slot,np.array(scores,np.float32)


def main(base,identifier):
    base=Path(base).resolve();assert (base/'RESULT.json').exists() and (R/'control_from_core_flip0/DONE.json').exists()
    folder=R/'analyses'/identifier;folder.mkdir(parents=True,exist_ok=True);recipes=definitions();raw=read(base/'RAW_TEST270.json')['per_case']
    index=read(LABEL_DIR/'index.json');assert [r['case_id'] for r in raw]==index['case_ids'];cost=read(base/'RESULT.json')['locator_forwards'];views=inferred_views(base/'RECIPE.json')
    write(folder/'FREEZE.json',dict(at=now(),recipes=recipes,source_sha256=sha(Path(__file__)),base_folder=str(base.relative_to(ROOT)),base_raw_sha256=sha(base/'RAW_TEST270.json'),original_control_sha256=sha(R/'control_from_core_flip0/DONE.json'),base_views=views,original_view_reused_without_extra_forward=0 in views,test_tuned=True,GT_input_to_postprocess=False))
    start=time.monotonic();dice=np.zeros((len(recipes),270),np.float32)
    with concurrent.futures.ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        for done,(slot,values) in enumerate(pool.map(case_task,[(i,str(base)) for i in range(270)],chunksize=1),1):
            guard();dice[:,slot]=values
            if done%10==0:write(folder/'STATUS.json',dict(at=now(),completed=done,total=270,seconds=time.monotonic()-start))
    assert np.array_equal(dice[0],np.array([r['locator_dsc'] for r in raw],np.float32))
    subjects=sorted({r['subject_id'] for r in raw});groups=[[i for i,r in enumerate(raw) if r['subject_id']==s] for s in subjects]
    mean=dice.mean(1,dtype=np.float64);sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],1).mean(1);hits=sum(r['automatic_point_hits_gt'] for r in raw)
    ranking=[]
    for i,recipe in enumerate(recipes):
        uses_original=recipe['source'] in ['union','intersection','distance'];neural=cost+int(uses_original and 0 not in views)
        ranking.append(dict(configuration_index=i,recipe=recipe,scan_dice=float(mean[i]),subject_dice=float(sub[i]),point_hits=hits,neural_case_forwards=neural))
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['configuration_index']));np.save(folder/'ALL_PER_CASE_DICE.npy',dice)
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,base_folder=str(base.relative_to(ROOT)),test_tuned=True,model_forwards_added=0))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=len(recipes),seconds=time.monotonic()-start,test_tuned=True,base_folder=str(base.relative_to(ROOT)),dice_sha256=sha(folder/'ALL_PER_CASE_DICE.npy')))
    print(ranking[0])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--id',required=True);a=p.parse_args();main(a.base,a.id)
