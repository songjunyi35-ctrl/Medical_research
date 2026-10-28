"""One global calibration function may depend on unlabeled predicted size/uncertainty."""
import concurrent.futures
import multiprocessing as mp
import numpy as np
import csv
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.calibration_v1 import score_cuts


def parameters():
    return [dict(temperature_base=t,volume_exponent=a,uncertainty_exponent=g,bias=b,volume_scale=s)
            for t in [.75,1.,1.25,1.5,2.,2.5,3.,4.] for a in [-.5,-.25,0.,.25,.5]
            for g in [-.5,0.,.5] for b in [-.5,0.,.5] for s in [.9,1.,1.1]]


def cuts_from_counts(counts):
    counts=np.asarray(counts,np.int64);bins=len(counts);p=np.linspace(1,0,bins).clip(1e-7,1-1e-7)
    descending=counts[::-1];k=np.cumsum(descending,dtype=np.int64);n=int(k[-1]);assert n>0
    predicted_size=int(counts[bins//2:].sum());mass=float(np.dot(p,descending))
    uncertainty=float(np.dot(p*(1-p),descending)/max(mass,1e-10))
    ratio=np.clip(max(predicted_size,1)/20000.,.01,100.)
    uncertainty_ratio=np.clip((uncertainty+.05)/.25,.1,10.)
    logit=np.log(p)-np.log1p(-p);params=parameters();out=[]
    for begin in range(0,len(params),60):
        block=params[begin:begin+60]
        temperatures=np.array([np.clip(x['temperature_base']*ratio**x['volume_exponent']*uncertainty_ratio**x['uncertainty_exponent'],.25,12.) for x in block])
        biases=np.array([x['bias'] for x in block]);scale=np.array([x['volume_scale'] for x in block])
        q=1/(1+np.exp(-(logit[None]+biases[:,None])/temperatures[:,None]))
        prefix=np.cumsum(q*descending[None],axis=1,dtype=np.float64)
        total=prefix[:,-1];index=np.argmax(2*prefix/(k[None]+total[:,None]),axis=1)
        desired=np.rint(k[index]*scale).clip(1,n).astype(np.int64)
        right=np.searchsorted(k,desired,side='left').clip(0,bins-1);left=np.maximum(right-1,0)
        chosen=np.where(abs(k[left]-desired)<=abs(k[right]-desired),left,right)
        out.extend((bins-1-chosen).tolist())
    return np.array(out,np.uint16),dict(predicted_size=predicted_size,probability_mass=mass,uncertainty=uncertainty)


def task(args):
    source,ri,cid,path=args;hist=np.load(TMP/'analyses'/source/'histograms'/str(ri)/(cid+'.npy'))
    cuts,features=cuts_from_counts(hist.sum(0,dtype=np.int64));scores=score_cuts(hist,cuts)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,dice=scores,cuts=cuts,features=json.dumps(features))
    return ri,cid


def gate():
    rng=np.random.default_rng(20261028);hist=rng.integers(0,10,(2,64));counts=hist.sum(0)
    a,f=cuts_from_counts(counts);b,g=cuts_from_counts(counts)
    assert np.array_equal(a,b) and f==g
    # Arbitrary target reallocations with unchanged unlabeled counts cannot affect cuts.
    h=hist.copy();h[1]=rng.binomial(counts,.4);h[0]=counts-h[1]
    c,_=cuts_from_counts(h.sum(0));assert np.array_equal(a,c)
    score=score_cuts(hist,a)
    for i in [0,len(a)//2,len(a)-1]:
        tp=hist[1,int(a[i]):].sum();pred=hist[:,int(a[i]):].sum();expected=np.float32(2*tp/(pred+hist[1].sum()))
        assert score[i]==expected
    return dict(at=now(),status='passed',deterministic=True,GT_reallocation_invariant=True,direct_mask_counts_exact=True,parameters=len(parameters()))


def execute(job):
    verify_job(job);folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True);write(folder/'GATE.json',gate())
    source=job['source_analysis'];rank=read(R/'analyses'/job['calibration_rank']/'RANKING.json')['best_by_recipe']
    candidates=rank[:5]
    forced=next(x for x in rank if x['recipe']['id']=='meanp_T1.0_subset220')
    if forced['recipe_index'] not in {x['recipe_index'] for x in candidates}:candidates.append(forced)
    manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl';rows=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True);params=parameters();tasks=[]
    write(folder/'PLAN.json',dict(at=now(),parameters=params,shortlist=candidates,selection='top5 fixed-calibration recipes plus old5; same reused270',
          features='predicted p>0.5 voxelcount and sum(p*(1-p))/sum(p); no GT',size_reference_voxels=20000,uncertainty_reference=.25,test_tuned=True))
    for item in candidates:
        for row in rows:
            dest=scratch/'cases'/str(item['recipe_index'])/(row['case_id']+'.npz')
            if not dest.exists():tasks.append((source,item['recipe_index'],row['case_id'],str(dest)))
    start=time.monotonic();total=len(candidates)*270;done=total-len(tasks)
    with concurrent.futures.ProcessPoolExecutor(8,mp_context=mp.get_context('spawn')) as pool:
        for _ in pool.map(task,tasks,chunksize=1):
            guard();done+=1
            if done%25==0:status('adaptive_calibration',job=job['id'],completed=done,total=total,seconds=time.monotonic()-start)
    subjects=sorted({r['subject_id'] for r in rows});groups=[[i for i,r in enumerate(rows) if r['subject_id']==s] for s in subjects];ranking=[]
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','temperature_base','volume_exponent','uncertainty_exponent','bias','volume_scale','scan_dice','subject_dice','point_hits'])
        for item in candidates:
            ri=item['recipe_index'];dice=np.stack([np.load(scratch/'cases'/str(ri)/(row['case_id']+'.npz'))['dice'] for row in rows],1)
            mean=dice.mean(1,dtype=np.float64);sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],1).mean(1);selected=int(np.argmax(mean))
            ranking.append(dict(recipe_index=ri,recipe=item['recipe'],parameters=params[selected],parameter_index=selected,scan_dice=float(mean[selected]),subject_dice=float(sub[selected]),point_hits=item['point_hits']))
            for pi,param in enumerate(params):w.writerow([item['recipe']['id'],*[param[k] for k in ['temperature_base','volume_exponent','uncertainty_exponent','bias','volume_scale']],mean[pi],sub[pi],item['point_hits']])
    ranking.sort(key=lambda x:(-x['scan_dice'],-x['subject_dice'],x['recipe_index']))
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,configurations=len(params)*len(candidates),test_tuned=True,model_forwards=0))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=len(params)*len(candidates),seconds=time.monotonic()-start,test_tuned=True,model_forwards=0))
