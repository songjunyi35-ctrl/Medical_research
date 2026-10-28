"""Mix one transformed eight-view group with the fixed five-view probability map."""
import csv
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import threshold_scores,summary_arrays
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram


def definitions(job,view_count=None,base_reused=False):
    n=job['prefix'];out=[]
    for phase_weight in [.25,.5,.75]:
        for temperature in [.25,.5,1.,2.,4.]:
            for base in ['base7','base8raw','base8soft']:
                count=7 if base=='base7' else 8
                for alpha in [0.,.025,.05,.1,.2,.35,.5,.75,1.]:
                    if alpha==1. and base!='base7':continue
                    if alpha==0. and not (n==64 and phase_weight==.5 and temperature==1.):continue
                    out.append(dict(id=f'cross_phase{n}_w{phase_weight}_T{temperature}_{base}_a{alpha}',kind='group_mix',group=job['phase0_source'],group_sources=[job['phase0_source'],job['phase7_source']],phase_weight=phase_weight,base=base,group_temperature=temperature,alpha=alpha,views=list(range(8)),neural_case_forwards=count if alpha==0 else (2*n if alpha==1 else 2*n+count)))
    return out


def prepare(cid,job):
    p0=torch.from_numpy(np.load(TMP/'groups'/job['phase0_source']/'mean_probability'/(cid+'.npy'))).cuda()
    p7=torch.from_numpy(np.load(TMP/'groups'/job['phase7_source']/'mean_probability'/(cid+'.npy'))).cuda()
    logits=[torch.from_numpy(np.load(TMP/'groups'/job['core_source']/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
    stack=torch.stack([z.sigmoid() for z in logits]);reference=stack.mean(0)>.2;votes=stack>.2
    tp=(votes&reference[None]).flatten(1).sum(1).double();den=votes.flatten(1).sum(1).double()+reference.sum();score=2*tp/den.clamp_min(1);weights=torch.softmax(score*16.,0)
    raw=torch.zeros_like(logits[0]);base7=torch.zeros_like(raw)
    for i in range(8):
        raw.add_(logits[i],alpha=float(weights[i]))
        if i in [0,1,3,4,5,6,7]:base7.add_((logits[i]/4).sigmoid(),alpha=1/7)
    return dict(p0=p0,p7=p7,base7=base7,base8raw=raw.sigmoid(),base8soft=(raw/4).sigmoid())


def probability(cache,recipe):
    base=cache[recipe['base']]
    if recipe['alpha']==0:return base
    group=cache['p0']*recipe['phase_weight']+cache['p7']*(1-recipe['phase_weight'])
    p=torch.sigmoid(torch.logit(group.clamp(1e-7,1-1e-7))/recipe['group_temperature'])
    return base*(1-recipe['alpha'])+p*recipe['alpha']


def wait_scratch(required,folder):
    from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
    while True:
        guard();used=stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']
        if used+required+256*2**20<=300*2**30:return
        status('cross_phase_waiting_for_scratch',job=folder.name,used_gib=used/2**30);time.sleep(10)


@torch.inference_mode()
def execute(job):
    verify_job(job);folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometry,_,labels=data();recipes=definitions(job)
    thresholds=np.round(np.arange(.025,.9751,.025),6).tolist()
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,bins=65536,selected_on_test270=True))
    n=len(recipes);nt=len(thresholds)
    dp=scratch/'dice.npy';hp=scratch/'hits.npy'
    if not dp.exists():wait_scratch(n*nt*270*4+n*270+1024,folder)
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
    chosen=ranking[:8]
    for row in ranking:
        if row['recipe']['alpha']==0 and row['recipe_index'] not in {x['recipe_index'] for x in chosen}:chosen.append(row)
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,calibration_shortlist=chosen,total_configurations=n*nt,test_tuned=True))
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,threshold in enumerate(thresholds):w.writerow([recipe['id'],threshold,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    remaining=sum(not (scratch/'histograms'/str(x['recipe_index'])/(row['case_id']+'.npy')).exists() for x in chosen for row in rows)
    wait_scratch(remaining*(2*65536*4+128)+len(chosen)*270*11475*6,folder)
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];cache=prepare(cid,job);target=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda()
        for item in chosen:
            out=scratch/'histograms'/str(item['recipe_index'])/(cid+'.npy')
            if not out.exists():save_array(out,histogram(probability(cache,item['recipe']),target).astype(np.uint32))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=n*nt,recipes=n,histogram_shortlist=len(chosen),
          seconds=time.monotonic()-start,best=ranking[0],model_forwards=0,test_tuned=True,dice_sha256=sha(dp),hits_sha256=sha(hp)))
