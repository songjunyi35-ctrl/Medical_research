"""Local bilateral Potts mean-field candidate; explicitly not a fully connected CRF."""
import itertools
import csv
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import threshold_scores,summary_arrays
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram

TAUS=[.5,1.,2.]
SIGMAS=[.125,.25,.5,1.,2.]
STRENGTHS=[.25,.5,1.,2.,4.,8.]
ITERATIONS=[1,3,5,10]


def definitions():
    return [dict(id=f'local_c{conn}_s{sigma}_t{tau}_l{strength}_i{iterations}',kind='local_bilateral_mrf',
                 connectivity=conn,intensity_sigma=sigma,unary_temperature=tau,strength=strength,iterations=iterations,
                 views=[2,3,4,6,7],neural_case_forwards=5)
            for conn in [1,3] for sigma in SIGMAS for tau in TAUS for strength in STRENGTHS for iterations in ITERATIONS]


def pairs(shape,connectivity):
    result=[]
    for offset in itertools.product([-1,0,1],repeat=3):
        norm=sum(v*v for v in offset)
        if norm==0 or (connectivity==1 and norm!=1):continue
        a=[];b=[]
        for n,d in zip(shape,offset):
            a.append(slice(max(0,-d),min(n,n-d)))
            b.append(slice(max(0,d),min(n,n+d)))
        result.append((tuple(a),tuple(b),norm))
    return result


def edges(image,sigma,connectivity):
    values=[];denominator=torch.zeros_like(image)
    for a,b,norm in pairs(image.shape,connectivity):
        weight=torch.exp(-.5*((image[a]-image[b])/sigma).square())*float(np.exp(-.5*norm))
        denominator[a].add_(weight);values.append((a,b,weight))
    return values,denominator.clamp_min(1e-10)


def refine_stream(p,image,connectivity,sigma,tau,strengths=STRENGTHS,iterations=ITERATIONS):
    """Only probabilities and input MRI are accessed; no GT or case identity."""
    links,denominator=edges(image,sigma,connectivity)
    unary=torch.logit(p.clamp(1e-7,1-1e-7))/tau
    q=(unary[None].expand(len(strengths),*p.shape)).sigmoid()
    weights=torch.tensor(strengths,device=p.device,dtype=p.dtype)[:,None,None,None]
    for iteration in range(1,max(iterations)+1):
        message=torch.zeros_like(q)
        for a,b,w in links:message[(slice(None),)+a].add_((2*q[(slice(None),)+b]-1)*w[None])
        q=(unary[None]+weights*message/denominator[None]).sigmoid()
        if iteration in iterations:yield iteration,q


def refine(p,image,recipe):
    return next(refine_stream(p,image,recipe['connectivity'],recipe['intensity_sigma'],recipe['unary_temperature'],
                strengths=[recipe['strength']],iterations=[recipe['iterations']]))[1][0]


def base_probability(cid,source):
    value=None
    for flip in [2,3,4,6,7]:
        z=torch.from_numpy(np.load(TMP/'groups'/source/'atoms'/f'flip{flip}'/(cid+'.npy'))).cuda()
        if value is None:value=torch.zeros_like(z)
        value.add_(z.sigmoid(),alpha=.2)
    return value


def gate():
    torch.manual_seed(20261028);p=torch.sigmoid(torch.randn(7,9,11,device='cuda'));image=torch.randn_like(p)
    for conn in [1,3]:
        recipe=dict(connectivity=conn,intensity_sigma=.5,unary_temperature=1.,strength=0.,iterations=3)
        a=refine(p,image,recipe);b=refine(p,image,recipe)
        assert torch.equal(a,b) and torch.allclose(a,p,rtol=1e-6,atol=1e-7)
        assert torch.equal(a>.5,p>.5)
    uniform=torch.full_like(p,.5)
    recipe=dict(connectivity=3,intensity_sigma=.5,unary_temperature=1.,strength=4.,iterations=10)
    assert torch.equal(refine(uniform,image,recipe),uniform)
    # Batch-strength execution must match separate executions exactly.
    recipe.update(strength=.5,iterations=3)
    single=refine(p,image,recipe)
    batched=next(refine_stream(p,image,3,.5,1.,strengths=[.5,1.],iterations=[3]))[1][0]
    assert torch.equal(single,batched)
    return dict(at=now(),status='passed',zero_strength_identity_threshold=True,deterministic=True,
                uniform_fixed_point=True,batched_strength_exact=True,patient_model_forward=False)


@torch.inference_mode()
def execute(job):
    verify_job(job);folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    write(folder/'GATE.json',gate());scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometry,images,labels=data();recipes=definitions();lookup={d['id']:i for i,d in enumerate(recipes)}
    thresholds=np.round(np.arange(.025,.9751,.025),6).tolist();n=len(recipes);nt=len(thresholds)
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,source=job['core_source'],
          implementation='local bilateral Potts synchronous mean field; not DenseCRF',test_tuned=True))
    dp=scratch/'dice.npy';hp=scratch/'hits.npy'
    dice=np.lib.format.open_memmap(dp,mode='r+' if dp.exists() else 'w+',dtype=np.float32,shape=(n,nt,270))
    hits=np.lib.format.open_memmap(hp,mode='r+' if hp.exists() else 'w+',dtype=bool,shape=(n,270))
    start=time.monotonic()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];marker=folder/f'case{slot:03}.json'
        if marker.exists():continue
        sl=slices(geometry[cid]);image=torch.from_numpy(np.array(images[slot][sl])).cuda();target=torch.from_numpy(np.array(labels[slot][sl]>0)).cuda()
        p=base_probability(cid,job['core_source'])
        # The automatic point remains the pre-refinement five-view argmax.
        point_hit=bool(target.flatten()[int(p.argmax())])
        for conn in [1,3]:
            for sigma in SIGMAS:
                for tau in TAUS:
                    for iteration,q in refine_stream(p,image,conn,sigma,tau):
                        for si,strength in enumerate(STRENGTHS):
                            name=f'local_c{conn}_s{sigma}_t{tau}_l{strength}_i{iteration}';ri=lookup[name]
                            h=histogram(q[si],target);dice[ri,:,slot]=threshold_scores(h,thresholds)[0];hits[ri,slot]=point_hit
        dice.flush();hits.flush();write(marker,dict(at=now(),case_id=cid))
        status('local_mrf_scoring',job=job['id'],completed=slot+1,total=270,seconds=time.monotonic()-start,configurations=n*nt)
    means,subjects=summary_arrays(np.asarray(dice),rows);ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=int(np.argmax(means[ri]));ranking.append(dict(recipe_index=ri,recipe=recipe,threshold=thresholds[ti],threshold_index=ti,
          scan_dice=float(means[ri,ti]),subject_dice=float(subjects[ri,ti]),point_hits=int(hits[ri].sum())))
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],d['recipe_index']));chosen=ranking[:8]
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,calibration_shortlist=chosen,total_configurations=n*nt,test_tuned=True))
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,t in enumerate(thresholds):w.writerow([recipe['id'],t,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];sl=slices(geometry[cid]);image=torch.from_numpy(np.array(images[slot][sl])).cuda()
        target=torch.from_numpy(np.array(labels[slot][sl]>0)).cuda();p=base_probability(cid,job['core_source'])
        for item in chosen:
            out=scratch/'histograms'/str(item['recipe_index'])/(cid+'.npy')
            if not out.exists():save_array(out,histogram(refine(p,image,item['recipe']),target).astype(np.uint32))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=n*nt,recipes=n,histogram_shortlist=len(chosen),
          seconds=time.monotonic()-start,best=ranking[0],model_forwards=0,test_tuned=True,dice_sha256=sha(dp),hits_sha256=sha(hp)))
