"""Mix one transformed eight-view group with the fixed five-view probability map."""
import csv
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import threshold_scores,summary_arrays
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram


def definitions(group,view_count=8,base_reused=False):
    return [dict(id=f'{group}_{kind}_T{temp}_a{alpha}',kind='group_mix',group=group,representation=kind,
                 group_temperature=temp,alpha=alpha,views=list(range(8)),base_views=[2,3,4,6,7],
                 neural_case_forwards=view_count if alpha==1 or base_reused else view_count+5)
            for kind in ['mean_probability','mean_logit'] for temp in [.75,1.,1.5,2.] for alpha in [.1,.25,.5,.75,1.]]


def prepare(cid,job):
    folder=TMP/'groups'/job['group_source']
    p=torch.from_numpy(np.load(folder/'mean_probability'/(cid+'.npy'))).cuda()
    z=torch.from_numpy(np.load(folder/'mean_logit'/(cid+'.npy'))).cuda()
    base=torch.zeros_like(p)
    for i in [2,3,4,6,7]:
        atom=torch.from_numpy(np.load(TMP/'groups'/job['core_source']/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda()
        base.add_(atom.sigmoid(),alpha=.2)
    return dict(base=base,mean_probability=p,mean_logit=z)


def probability(cache,recipe):
    value=cache[recipe['representation']]
    if recipe['representation']=='mean_probability':
        value=torch.logit(value.clamp(1e-7,1-1e-7))
    p=(value/recipe['group_temperature']).sigmoid()
    return cache['base']*(1-recipe['alpha'])+p*recipe['alpha']


@torch.inference_mode()
def execute(job):
    verify_job(job);folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometry,_,labels=data();source_job=read(R/'queue_v2'/(job['group_source']+'.json'));source_done=read(R/'jobs'/job['group_source']/'DONE.json')
    cost=source_done.get('deployment_mean_case_forwards',len(source_job['views']))
    recipes=definitions(job['group_source'],cost,job.get('group_cost_includes_base',False))
    for recipe in recipes:recipe.update(group_views=source_job['views'],group_spec=source_job['spec'],base_shared_with_group=job.get('group_cost_includes_base',False))
    thresholds=np.round(np.arange(.025,.9751,.025),6).tolist()
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,bins=65536,selected_on_test270=True))
    n=len(recipes);nt=len(thresholds)
    dp=scratch/'dice.npy';hp=scratch/'hits.npy'
    dice=np.lib.format.open_memmap(dp,mode='r+' if dp.exists() else 'w+',dtype=np.float32,shape=(n,nt,270))
    hits=np.lib.format.open_memmap(hp,mode='r+' if hp.exists() else 'w+',dtype=bool,shape=(n,270))
    start=time.monotonic()
    for slot,row in enumerate(rows):
        guard();marker=folder/f'case{slot:03}.json'
        if marker.exists():continue
        cid=row['case_id'];cache=prepare(cid,job);target=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda()
        for ri,recipe in enumerate(recipes):
            p=probability(cache,recipe);h=histogram(p,target)
            dice[ri,:,slot]=threshold_scores(h,thresholds)[0];hits[ri,slot]=bool(target.flatten()[int(p.argmax())])
        dice.flush();hits.flush();write(marker,dict(at=now(),case_id=cid))
        if slot%8==0:status('group_fusion_scoring',job=job['id'],completed=slot+1,total=270,configurations=n*nt,seconds=time.monotonic()-start)
    means,subjects=summary_arrays(np.asarray(dice),rows);ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=int(np.argmax(means[ri]));ranking.append(dict(recipe_index=ri,recipe=recipe,threshold=thresholds[ti],threshold_index=ti,
          scan_dice=float(means[ri,ti]),subject_dice=float(subjects[ri,ti]),point_hits=int(hits[ri].sum())))
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],d['recipe']['neural_case_forwards'],d['recipe_index']))
    chosen=ranking[:4]
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,calibration_shortlist=chosen,total_configurations=n*nt,test_tuned=True))
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,threshold in enumerate(thresholds):w.writerow([recipe['id'],threshold,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];cache=prepare(cid,job);target=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda()
        for item in chosen:
            out=scratch/'histograms'/str(item['recipe_index'])/(cid+'.npy')
            if not out.exists():save_array(out,histogram(probability(cache,item['recipe']),target).astype(np.uint32))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=n*nt,recipes=n,histogram_shortlist=len(chosen),
          seconds=time.monotonic()-start,best=ranking[0],model_forwards=0,test_tuned=True,dice_sha256=sha(dp),hits_sha256=sha(hp)))
