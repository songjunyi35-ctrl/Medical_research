"""Global, test-tuned view combinations; labels only enter the scoring functions."""
import csv
import time
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import fuse_score, volume_decode
from scripts.segresnet_auto_inference12h_20261028.histograms_adaptive_v1 import histogram as fast_histogram
from scripts.segresnet_auto_inference12h_20261028.threshold_histograms_v1 import histogram as threshold_histogram, scores as short_scores


VOTE_THRESHOLDS=[.1,.2,.3]
CASE_EXPONENTS=[-16.,-8.,0.,4.,8.,16.,32.,64.]


def definitions(beam):
    out=[]
    for pixel in beam:
        for threshold in VOTE_THRESHOLDS:
            for exponent in CASE_EXPONENTS:
                if exponent==0 and threshold!=.2:continue
                out.append(dict(id=f'joint_{pixel["id"]}_tau{threshold}_case{exponent}',kind='joint_weighted_fusion',pixel_recipe=pixel,
                     vote_threshold=threshold,case_exponent=exponent,views=pixel['views'] if exponent==0 else list(range(8)),
                     neural_case_forwards=len(pixel['views']) if exponent==0 else 8))
    return out


def prepare(logits):
    from scripts.segresnet_auto_inference12h_20261028 import nonlinear_fusions_v1 as pixel
    cache=pixel.prepare(logits);stack=torch.stack(cache['temperatures'][1.])
    return dict(pixel=cache,mean=cache['mean'],agreement=dict(stack=stack,mean=stack.mean(0),scores={}))


def probability(cache,recipe):
    from scripts.segresnet_auto_inference12h_20261028 import nonlinear_fusions_v1 as pixel
    from scripts.segresnet_auto_inference12h_20261028.case_consistency_v1 import view_scores
    inner=recipe['pixel_recipe']
    if recipe['case_exponent']==0:return pixel.probability(cache['pixel'],inner)
    scores=view_scores(cache['agreement'],recipe['vote_threshold'])
    factor=torch.exp(recipe['case_exponent']*(scores-scores.max())).float()
    group=pixel.parts(cache['pixel'],inner);numerator=torch.zeros_like(group['products'][0]);denominator=torch.zeros_like(numerator)
    for i in inner['views']:
        numerator.add_(group['products'][i]*factor[i]);denominator.add_(group['weights'][i]*factor[i])
    result=numerator/denominator
    return (result/inner['view_temperature']).sigmoid() if inner['mode']=='weighted_logit' else result.clamp(0,1)


def numerical_gate():
    from scripts.segresnet_auto_inference12h_20261028 import nonlinear_fusions_v1 as pixel
    torch.manual_seed(1);logits=[torch.randn(7,9,11,device='cuda') for _ in range(8)];cache=prepare(logits)
    inner=dict(id='gate_pixel',mode='weighted_logit',view_temperature=4.,weighting_temperature=8.,exponent=-1.,epsilon=1e-5,views=[0,1,3,4,5,6,7])
    recipe=definitions([inner])[0];recipe['case_exponent']=0.
    assert torch.equal(probability(cache,recipe),pixel.probability(cache['pixel'],inner))
    for strength in [-16.,16.,64.]:
        recipe['case_exponent']=strength;a=probability(cache,recipe);b=probability(cache,recipe)
        assert torch.equal(a,b) and torch.isfinite(a).all()
    return dict(at=now(),status='passed',lambda0_pixel_formula_exact=True,negative_positive_lambda_finite=True,repeat_exact=True,GT_input=False)


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


def wait_scratch(required,folder,stage):
    from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
    while True:
        guard();used=stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']
        if used+required+256*2**20<=300*2**30:return
        status('waiting_for_scratch',job=folder.name,stage=stage,used_gib=used/2**30,required_gib=required/2**30);time.sleep(10)


@torch.inference_mode()
def execute(job):
    verify_job(job)
    source=job['core_source'];folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    write(folder/'NUMERICAL_GATE.json',numerical_gate())
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometries,_,labels=data();recipes=definitions(job['pixel_beam']);thresholds=np.round(np.arange(.025,.9751,.025),6).tolist()
    assert len(recipes)==job['recipe_count']
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,bins=65536,
          calibration_shortlist='top24 plus best global recipe for each case-weight exponent',test_tuned=True))
    n=len(recipes);nt=len(thresholds);bins=65536
    path=scratch/'dice.npy'
    if not path.exists():wait_scratch(n*nt*270*4+n*270+1024,folder,'score_matrix')
    dice=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=(n,nt,270))
    path=scratch/'hits.npy';hits=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=bool,shape=(n,270))
    baseline_reference=read(R/'CURRENT_BASELINE.json')
    assert baseline_reference['resource_token']==job['resource_token']
    baseline_folder=ROOT/baseline_reference['folder']
    old={d['case_id']:d for d in read(baseline_folder/'RAW_TEST270.json')['per_case']}
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
        saved=np.load(ROOT/old[cid].get('prediction_path','missing')) if 'prediction_path' in old[cid] else np.load(baseline_folder/'predictions'/(cid+'.npz'))
        original=np.unpackbits(saved['mask_bits'])[:int(np.prod(mask.shape))].reshape(mask.shape).astype(bool)
        audit=dict(case_id=cid,score=score,dice_difference=score-old[cid]['locator_dsc'],
                   mask_different_voxels=int((mask!=original).sum()),point_equal=point==old[cid]['automatic_point'])
        assert audit['mask_different_voxels']==0 and audit['point_equal'] and audit['dice_difference']==0, audit
        write(folder/'baseline_cases'/f'{slot:03}.json',audit)
        cache=prepare(logits)
        if slot==0:
            probes=[]
            for sample in [cache['mean'],torch.rand_like(cache['mean'])]:
                torch.cuda.synchronize();t=time.monotonic();reference=histogram(sample,target,bins);torch.cuda.synchronize();reference_seconds=time.monotonic()-t
                torch.cuda.synchronize();t=time.monotonic();optimized=fast_histogram(sample,target,bins);torch.cuda.synchronize();optimized_seconds=time.monotonic()-t
                assert np.array_equal(reference,optimized)
                probes.append(dict(reference_seconds=reference_seconds,optimized_seconds=optimized_seconds,integer_counts_exact=True))
            write(folder/'HISTOGRAM_GPU_GATE.json',dict(at=now(),status='passed',probes=probes))
        for ri,recipe in enumerate(recipes):
            p=probability(cache,recipe);hist=threshold_histogram(p,target)
            scores=short_scores(hist);dice[ri,:,slot]=scores
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
    chosen=ranking[:24]
    for exponent in CASE_EXPONENTS:
        row=next(d for d in ranking if d['recipe']['case_exponent']==exponent)
        if row['recipe_index'] not in {v['recipe_index'] for v in chosen}:chosen.append(row)
    write(folder/'RANKING.json',dict(at=now(),best_by_recipe=ranking,calibration_shortlist=chosen,
          total_configurations=n*nt,best=ranking[0],model_forwards=0,test_tuned=True))
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,t in enumerate(thresholds):w.writerow([recipe['id'],t,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    missing=sum(not (scratch/'histograms'/str(item['recipe_index'])/(row['case_id']+'.npy')).exists() for item in chosen for row in rows)
    wait_scratch(missing*(2*bins*4+128)+len(chosen)*270*11475*6,folder,'histograms_and_calibration_outputs')
    # Second GPU pass saves only the selected high-resolution histograms.
    for slot,row in enumerate(rows):
        cid=row['case_id'];logits=[torch.from_numpy(np.load(TMP/'groups'/source/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
        target=torch.from_numpy(np.array(labels[slot][slices(geometries[cid])]>0)).cuda();cache=prepare(logits)
        for item in chosen:
            ri=item['recipe_index'];out=scratch/'histograms'/str(ri)/(cid+'.npy')
            if not out.exists():save_array(out,fast_histogram(probability(cache,recipes[ri]),target,bins).astype(np.uint32))
        del cache,logits,target
    audits=[read(folder/'baseline_cases'/f'{i:03}.json') for i in range(270)]
    write(folder/'BASELINE_REPRODUCED.json',dict(at=now(),status='passed',scans=270,
          max_dice_difference=max(abs(d['dice_difference']) for d in audits),all_masks_and_points_exact=True,
          baseline_scan_dice=float(np.mean([d['score'] for d in audits]))))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=n*nt,recipes=n,
          histogram_shortlist=len(chosen),seconds=time.monotonic()-started,best=ranking[0],model_forwards=0,test_tuned=True,
          dice_sha256=sha(scratch/'dice.npy'),hits_sha256=sha(scratch/'hits.npy')))
