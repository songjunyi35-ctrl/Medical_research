"""Global, test-tuned view combinations; labels only enter the scoring functions."""
import csv
import time
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_inference_search_20261028.operations import fuse_score, volume_decode
from scripts.segresnet_auto_inference12h_20261028.histograms_adaptive_v1 import histogram as fast_histogram
from scripts.segresnet_auto_inference12h_20261028.threshold_histograms_v1 import histogram as threshold_histogram, scores as short_scores


TEMPERATURES=[1.,2.,3.,4.,5.,6.,8.,12.]
EXPONENTS=[-1.,-.5,-.25,.125,.25,.5,1.,2.]
SUBSETS=[bits for bits in range(1,256) if bits.bit_count()>=2]


def definitions():
    out=[]
    groups=[]
    for temperature in TEMPERATURES:
        for weighting_temperature in sorted({1.,temperature}):groups.append(('weighted_probability',temperature,weighting_temperature))
    for weighting_temperature in [1.,2.,4.,8.]:groups.append(('weighted_logit',4.,weighting_temperature))
    for mode,temperature,weighting_temperature in groups:
        for exponent in EXPONENTS:
            for bits in SUBSETS:
                views=[i for i in range(8) if bits&(1<<i)]
                out.append(dict(id=f'{mode}_T{temperature}_W{weighting_temperature}_e{exponent}_subset{bits}',kind='nonlinear_weighted_fusion',mode=mode,
                     view_temperature=temperature,weighting_temperature=weighting_temperature,exponent=exponent,epsilon=1e-5,views=views,neural_case_forwards=len(views)))
    return out


def prepare(logits):
    temps={t:[torch.sigmoid(z/t) for z in logits] for t in TEMPERATURES}
    return dict(logits=logits,temperatures=temps,mean=torch.stack(temps[4.]).mean(0))


def parts(cache,recipe):
    reference=cache['temperatures'][recipe['weighting_temperature']]
    weights=[(p*(1-p)+recipe['epsilon']).pow(-recipe['exponent']) for p in reference]
    values=cache['logits'] if recipe['mode']=='weighted_logit' else cache['temperatures'][recipe['view_temperature']]
    return dict(weights=weights,products=[w*p for w,p in zip(weights,values)],mode=recipe['mode'],temperature=recipe['view_temperature'])


def probability_from_parts(group,views):
    numerator=torch.zeros_like(group['products'][0]);denominator=torch.zeros_like(group['weights'][0])
    for i in views:numerator.add_(group['products'][i]);denominator.add_(group['weights'][i])
    result=numerator/denominator
    return (result/group['temperature']).sigmoid() if group['mode']=='weighted_logit' else result.clamp(0,1)


def probability(cache,recipe):return probability_from_parts(parts(cache,recipe),recipe['views'])


def numerical_gate():
    torch.manual_seed(1);logits=[torch.randn(7,9,11,device='cuda') for _ in range(8)];cache=prepare(logits);recipes=definitions()
    for recipe in [recipes[0],recipes[len(recipes)//2],recipes[-1]]:
        a=probability(cache,recipe);b=probability_from_parts(parts(cache,recipe),recipe['views'])
        assert torch.equal(a,b) and torch.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    uniform=prepare([torch.zeros_like(logits[0]) for _ in range(8)])
    assert torch.equal(probability(uniform,recipes[0]),torch.full_like(logits[0],.5))
    return dict(at=now(),status='passed',cached_vs_standalone_exact=True,uniform_probability_fixed_point=True,GT_input=False)


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
    main=read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']
    while True:
        guard();used=stable_size(TMP)+stable_size(CACHE)+main
        if used+required+256*2**20<=300*2**30:return
        write(folder/'SCRATCH_WAIT.json',dict(at=now(),stage=stage,used_gib=used/2**30,required_gib=required/2**30,limit_gib=300))
        status('waiting_for_authorized_rolling_cleanup',job=folder.name,stage=stage,used_gib=used/2**30,required_gib=required/2**30)
        time.sleep(10)

@torch.inference_mode()
def execute(job):
    verify_job(job)
    source=job['core_source'];folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    write(folder/'NUMERICAL_GATE.json',numerical_gate())
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometries,_,labels=data();recipes=definitions();thresholds=np.round(np.arange(.025,.9751,.025),6).tolist()
    assert len(recipes)==job['recipe_count']
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,bins=65536,
          calibration_shortlist='top40 plus best mode/temperature family plus current seven-view subset under every weighting regime',test_tuned=True))
    n=len(recipes);nt=len(thresholds);bins=65536
    path=scratch/'dice.npy'
    if not path.exists():wait_scratch(n*nt*270*4+n*270+1024,folder,'first_pass_matrix')
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
        for begin in range(0,n,len(SUBSETS)):
            group=parts(cache,recipes[begin])
            for ri in range(begin,begin+len(SUBSETS)):
                recipe=recipes[ri];p=probability_from_parts(group,recipe['views']);hist=threshold_histogram(p,target)
                dice[ri,:,slot]=short_scores(hist);hits[ri,slot]=bool(target.flatten()[int(p.argmax())])
        dice.flush();hits.flush();write(marker,dict(at=now(),case_id=cid,recipes=n,thresholds=nt))
        status('nonlinear_fusion_scoring',job=job['id'],completed=slot+1,total=270,recipes=n,configurations=n*nt,seconds=time.monotonic()-started)
        del cache,logits,target,exact,mask,p
    means,subjects=summary_arrays(np.asarray(dice),rows)
    ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=min(range(nt),key=lambda i:(-means[ri,i],-subjects[ri,i],i))
        ranking.append(dict(recipe_index=ri,recipe=recipe,threshold=thresholds[ti],threshold_index=ti,
              scan_dice=float(means[ri,ti]),subject_dice=float(subjects[ri,ti]),point_hits=int(hits[ri].sum())))
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],len(d['recipe']['views']),d['recipe']['id']))
    chosen=ranking[:40]
    families=sorted({(d['recipe']['mode'],d['recipe']['view_temperature'],d['recipe']['weighting_temperature']) for d in ranking})
    for family in families:
        row=next(d for d in ranking if (d['recipe']['mode'],d['recipe']['view_temperature'],d['recipe']['weighting_temperature'])==family)
        if row['recipe_index'] not in {v['recipe_index'] for v in chosen}:chosen.append(row)
    # Retain the current seven-view set under every weighting regime for calibration,
    # even when its best global hard threshold is not in the first40.
    for row in ranking:
        if row['recipe']['views']==[0,1,3,4,5,6,7] and row['recipe_index'] not in {v['recipe_index'] for v in chosen}:chosen.append(row)
    write(folder/'RANKING.json',dict(at=now(),best_by_recipe=ranking,calibration_shortlist=chosen,
          total_configurations=n*nt,best=ranking[0],model_forwards=0,test_tuned=True))
    csv_parts=[];handle=None;writer=None;written=0
    try:
        for ri,recipe in enumerate(recipes):
            for ti,t in enumerate(thresholds):
                if written%200000==0:
                    if handle is not None:handle.close()
                    part=folder/f'ALL_CONFIGURATIONS.part{len(csv_parts):03}.tsv';csv_parts.append(part)
                    handle=part.open('w');writer=csv.writer(handle,delimiter='\t',lineterminator='\n');writer.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
                writer.writerow([recipe['id'],t,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())]);written+=1
    finally:
        if handle is not None:handle.close()
    write(folder/'CSV_PARTS.json',[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in csv_parts])
    missing_histograms=sum(not (scratch/'histograms'/str(item['recipe_index'])/(row['case_id']+'.npy')).exists() for item in chosen for row in rows)
    wait_scratch(missing_histograms*(2*bins*4+128),folder,'shortlist_histograms')
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
