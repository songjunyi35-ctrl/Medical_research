"""Global, test-tuned view combinations; labels only enter the scoring functions."""
import csv
import time
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import fuse_score, volume_decode


def definitions():
    out=[]
    for temperature in [.5,.75,1.,1.5,2.,3.,4.]:
        for bits in range(1,256):
            views=[i for i in range(8) if bits&(1<<i)]
            out.append(dict(id=f'meanp_T{temperature}_subset{bits}',kind='probability_mean',views=views,view_temperature=temperature))
    for bits in range(1,256):
        out.append(dict(id=f'logit_subset{bits}',kind='logit_mean',views=[i for i in range(8) if bits&(1<<i)]))
    for r in [-4.,-2.,-1.,0.,.25,.5,2.,4.,8.]:
        for norm in [False,True]:
            out.append(dict(id=f'power{r}_norm{norm}',kind='power',power=r,normalized=norm,views=list(range(8))))
    for q in [.1,.25,.5,.75,.9]:out.append(dict(id=f'quantile{q}',kind='quantile',q=q,views=list(range(8))))
    for c in [-2.,-1.,-.5,-.25,.25,.5,1.,2.]:out.append(dict(id=f'mean_std{c}',kind='mean_std',coefficient=c,views=list(range(8))))
    for exponent in [.25,.5,1.,2.]:out.append(dict(id=f'confidence_weight{exponent}',kind='confidence',exponent=exponent,views=list(range(8))))
    out.append(dict(id='trimmed_mean6',kind='trim',trim=1,views=list(range(8))))
    out.append(dict(id='trimmed_mean4',kind='trim',trim=2,views=list(range(8))))
    return out


def prepare(logits):
    temps={t:[torch.sigmoid(z/t) for z in logits] for t in [.5,.75,1.,1.5,2.,3.,4.]}
    stack=torch.stack(temps[1.])
    return dict(logits=logits,temperatures=temps,stack=stack,mean=stack.mean(0),std=stack.std(0,unbiased=False))


def probability(cache,recipe):
    kind=recipe['kind'];views=recipe['views']
    if kind in ('probability_mean','logit_mean'):
        values=(cache['temperatures'][recipe['view_temperature']] if kind=='probability_mean' else cache['logits'])
        value=torch.zeros_like(values[0])
        for i in views:value.add_(values[i],alpha=1/len(views))
        return value if kind=='probability_mean' else value.sigmoid()
    if kind=='mean_std':return (cache['mean']+recipe['coefficient']*cache['std']).clamp(0,1)
    if kind=='quantile':return torch.quantile(cache['stack'],recipe['q'],dim=0)
    if kind=='trim':return cache['stack'].sort(dim=0).values[recipe['trim']:-recipe['trim']].mean(0)
    if kind=='confidence':
        p=cache['stack'];weights=(p*(1-p)+1e-5).pow(-recipe['exponent'])
        return (weights*p).sum(0)/weights.sum(0)
    if kind=='power':
        p=cache['stack'].clamp(1e-6,1-1e-6);r=recipe['power']
        mean=lambda x:x.log().mean(0).exp() if r==0 else x.pow(r).mean(0).pow(1/r)
        foreground=mean(p)
        return foreground/(foreground+mean(1-p)) if recipe['normalized'] else foreground
    raise ValueError(kind)


def histogram(p,target,bins):
    codes=(p.clamp(0,1).flatten()*(bins-1)+.5).long()
    hist=torch.bincount(codes+target.flatten().long()*bins,minlength=2*bins).reshape(2,bins)
    return hist.cpu().numpy()


def threshold_scores(hist,thresholds):
    # The new decoder is explicitly quantized; exact masks use these same bins.
    bins=hist.shape[1]
    tail=np.cumsum(hist[:,::-1],axis=1,dtype=np.int64)[:,::-1]
    indices=np.floor(np.array(thresholds)*(bins-1)).astype(int)+1
    indices=np.clip(indices,0,bins-1)
    tp=tail[1,indices];fp=tail[0,indices];positive=hist[1].sum()
    denominator=tp+fp+positive
    return (2*tp/np.maximum(denominator,1)).astype(np.float32),tp,fp


def summary_arrays(dice,rows):
    subjects=sorted({d['subject_id'] for d in rows})
    groups=[[i for i,d in enumerate(rows) if d['subject_id']==s] for s in subjects]
    return dice.mean(-1,dtype=np.float64),np.stack([dice[...,g].mean(-1,dtype=np.float64) for g in groups],-1).mean(-1)


@torch.inference_mode()
def execute(job):
    verify_job(job)
    source=job['depends_on'][0];folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometries,_,labels=data();recipes=definitions();thresholds=np.round(np.arange(.025,.9751,.025),6).tolist()
    assert len(recipes)==job['recipe_count']
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,bins=65536,
          calibration_shortlist='top24 unique recipes plus best recipe per aggregation family',test_tuned=True))
    n=len(recipes);nt=len(thresholds);bins=65536
    path=scratch/'dice.npy'
    dice=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(n,nt,270))
    path=scratch/'hits.npy';hits=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=bool,shape=(n,270))
    old={d['case_id']:d for d in read(OLD/'RAW_TEST270.json')['per_case']}
    baseline=[];started=time.monotonic()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];marker=folder/f'case{slot:03}.json'
        if marker.exists():continue
        logits=[torch.from_numpy(np.load(TMP/'groups'/source/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
        y=np.array(labels[slot][slices(geometries[cid])]>0);target=torch.from_numpy(y).cuda()
        # Reproduce the old fixed candidate with full-precision original decoding.
        exact=fuse_score({f'scale1_flip{i}':logits[i] for i in range(8)},read(OLD/'SELECTED_RECIPE.json')['recipe'])
        mask,k=volume_decode(exact.cpu().numpy(),2.)
        tp=int((mask&y).sum());score=float(np.float32(2*tp)/np.float32(k+y.sum()))
        point=list(map(int,np.unravel_index(int(exact.argmax()),exact.shape)))
        saved=np.load(ROOT/old[cid].get('prediction_path','missing')) if 'prediction_path' in old[cid] else np.load(OLD/'predictions'/(cid+'.npz'))
        original=np.unpackbits(saved['mask_bits'])[:int(np.prod(mask.shape))].reshape(mask.shape).astype(bool)
        audit=dict(case_id=cid,score=score,dice_difference=score-old[cid]['locator_dsc'],
                   mask_different_voxels=int((mask!=original).sum()),point_equal=point==old[cid]['automatic_point'])
        assert audit['mask_different_voxels']==0 and audit['point_equal'] and audit['dice_difference']==0, audit
        write(folder/'baseline_cases'/f'{slot:03}.json',audit)
        cache=prepare(logits)
        for ri,recipe in enumerate(recipes):
            p=probability(cache,recipe);hist=histogram(p,target,bins)
            scores,_,_=threshold_scores(hist,thresholds);dice[ri,:,slot]=scores
            hits[ri,slot]=bool(target.flatten()[int(p.argmax())])
        dice.flush();hits.flush();write(marker,dict(at=now(),case_id=cid,recipes=n,thresholds=nt))
        write(R/'ANALYSIS_STATUS.json',dict(at=now(),analysis=job['id'],completed=slot+1,total=270,
              recipes=n,configurations=n*nt,seconds=time.monotonic()-started,test_tuned=True))
        del cache,logits,target,exact,mask,p
    means,subjects=summary_arrays(np.asarray(dice),rows)
    ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=min(range(nt),key=lambda i:(-means[ri,i],-subjects[ri,i],i))
        ranking.append(dict(recipe_index=ri,recipe=recipe,threshold=thresholds[ti],threshold_index=ti,
              scan_dice=float(means[ri,ti]),subject_dice=float(subjects[ri,ti]),point_hits=int(hits[ri].sum())))
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],len(d['recipe']['views']),d['recipe']['id']))
    chosen=ranking[:24];families={d['recipe']['kind'] for d in ranking}
    for family in sorted(families):
        row=next(d for d in ranking if d['recipe']['kind']==family)
        if row['recipe_index'] not in {v['recipe_index'] for v in chosen}:chosen.append(row)
    for name in ['meanp_T1.0_subset220','meanp_T1.0_subset255','meanp_T2.0_subset255']:
        row=next(d for d in ranking if d['recipe']['id']==name)
        if row['recipe_index'] not in {v['recipe_index'] for v in chosen}:chosen.append(row)
    write(folder/'RANKING.json',dict(at=now(),best_by_recipe=ranking,calibration_shortlist=chosen,
          total_configurations=n*nt,best=ranking[0],model_forwards=0,test_tuned=True))
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,t in enumerate(thresholds):w.writerow([recipe['id'],t,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    # Second GPU pass saves only the selected high-resolution histograms.
    for slot,row in enumerate(rows):
        cid=row['case_id'];logits=[torch.from_numpy(np.load(TMP/'groups'/source/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
        target=torch.from_numpy(np.array(labels[slot][slices(geometries[cid])]>0)).cuda();cache=prepare(logits)
        for item in chosen:
            ri=item['recipe_index'];out=scratch/'histograms'/str(ri)/(cid+'.npy')
            if not out.exists():save_array(out,histogram(probability(cache,recipes[ri]),target,bins).astype(np.uint32))
        del cache,logits,target
    audits=[read(folder/'baseline_cases'/f'{i:03}.json') for i in range(270)]
    write(folder/'BASELINE_REPRODUCED.json',dict(at=now(),status='passed',scans=270,
          max_dice_difference=max(abs(d['dice_difference']) for d in audits),all_masks_and_points_exact=True,
          baseline_scan_dice=float(np.mean([d['score'] for d in audits]))))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=n*nt,recipes=n,
          histogram_shortlist=len(chosen),seconds=time.monotonic()-started,best=ranking[0],model_forwards=0,test_tuned=True,
          dice_sha256=sha(scratch/'dice.npy'),hits_sha256=sha(scratch/'hits.npy')))
