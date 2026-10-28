"""Label-free histogram-volume decisions, with labels used only afterward to score."""
import concurrent.futures
import csv
import multiprocessing as mp
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *

TEMPERATURES=[.625,.75,.875,1.,1.125,1.25,1.5,1.75,2.]
BIASES=[-1.5,-1.25,-1.125,-1.,-.875,-.75,-.625,-.5,-.375,-.25,-.125,0.,.125,.25,.5]
BETAS=[.85,.9,.95,1.,1.05,1.1,1.15,1.2,1.25,1.3]
VOLUME_SCALES=[.85,.9,.95,1.,1.05,1.1,1.15]
_Q={}


def parameters():
    return [dict(post_temperature=t,bias=b,beta=beta,volume_scale=scale)
            for beta in BETAS for scale in VOLUME_SCALES for t in TEMPERATURES for b in BIASES]


def calibrated_bins(bins):
    if bins not in _Q:
        p=np.linspace(1,0,bins,dtype=np.float64).clip(1e-7,1-1e-7)
        logit=np.log(p)-np.log1p(-p)
        q=np.stack([1/(1+np.exp(-(logit+b)/t)) for t in TEMPERATURES for b in BIASES])
        q.flags.writeable=False;_Q[bins]=q
    return _Q[bins]


def all_cuts(counts):
    """Input is the unlabeled probability histogram, never positive/GT counts."""
    counts=np.asarray(counts,dtype=np.int64)
    bins=len(counts);n=int(counts.sum());assert n>0
    predicted=np.cumsum(counts[::-1],dtype=np.int64)
    mass=np.cumsum(calibrated_bins(bins)*counts[None,::-1],axis=1,dtype=np.float64)
    total=mass[:,-1]
    blocks=[]
    for beta in BETAS:
        objective=2*mass/(predicted[None,:]+beta**2*total[:,None])
        index=np.argmax(objective,axis=1)
        base_k=predicted[index]
        for scale in VOLUME_SCALES:
            target=np.rint(base_k*scale).clip(1,n).astype(np.int64)
            right=np.searchsorted(predicted,target,side='left').clip(0,bins-1)
            left=np.maximum(right-1,0)
            closer=np.where(abs(predicted[left]-target)<=abs(predicted[right]-target),left,right)
            blocks.append((bins-1-closer).astype(np.uint16))
    result=np.concatenate(blocks)
    assert len(result)==len(parameters())
    return result


def score_cuts(hist,cuts):
    positive=np.cumsum(np.asarray(hist[1,::-1],np.int64))
    predicted=np.cumsum(np.asarray(hist[:,::-1].sum(0),np.int64))
    indices=hist.shape[1]-1-cuts.astype(np.int64)
    tp=positive[indices];den=predicted[indices]+int(hist[1].sum())
    return (2*tp/np.maximum(den,1)).astype(np.float32)


def case_task(task):
    source,ri,cid,destination=task
    hist=np.load(TMP/'analyses'/source/'histograms'/str(ri)/(cid+'.npy'))
    assert hist.shape==(2,65536)
    cuts=all_cuts(hist.sum(0,dtype=np.int64))
    scores=score_cuts(hist,cuts)
    target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,dice=scores,cuts=cuts)
    os.replace(temp,target)
    return ri,cid,float(scores.max())


def execute(job):
    verify_job(job)
    source=job['source_analysis'];source_dir=R/'analyses'/source
    assert (source_dir/'DONE.json').is_file()
    shortlist=read(source_dir/'RANKING.json')['calibration_shortlist']
    shortlist=[item for item in shortlist if item['recipe_index'] in job['shortlist_recipe_indices']]
    manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    params=parameters();assert len(params)==9450
    write(folder/'PLAN.json',dict(at=now(),parameters=params,source_analysis=source,
          shortlisted_recipes=[d['recipe_index'] for d in shortlist],bins=65536,test_tuned=True,
          decision_uses_unlabeled_histogram_only=True,quantized_decoder_not_exact_RankDice=True))
    tasks=[]
    for item in shortlist:
        ri=item['recipe_index']
        for row in rows:
            dest=scratch/'cases'/str(ri)/(row['case_id']+'.npz')
            if not dest.exists():tasks.append((source,ri,row['case_id'],str(dest)))
    started=time.monotonic();total=len(shortlist)*270;done=total-len(tasks)
    with concurrent.futures.ProcessPoolExecutor(min(8,job.get('cpu_workers',8)),mp_context=mp.get_context('spawn')) as pool:
        for ri,cid,best in pool.map(case_task,tasks,chunksize=1):
            guard();done+=1
            if done%25==0 or done==total:
                write(R/'CALIBRATION_STATUS.json',dict(at=now(),analysis=job['id'],completed=done,total=total,
                      configurations=len(shortlist)*len(params),seconds=time.monotonic()-started,test_tuned=True))
    subjects=sorted({d['subject_id'] for d in rows})
    groups=[[i for i,d in enumerate(rows) if d['subject_id']==subject] for subject in subjects]
    ranking=[]
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['recipe','post_temperature','bias','beta','volume_scale','scan_dice','subject_dice','point_hits'])
        for item in shortlist:
            ri=item['recipe_index']
            arrays=[np.load(scratch/'cases'/str(ri)/(row['case_id']+'.npz')) for row in rows]
            dice=np.stack([a['dice'] for a in arrays],axis=1)
            mean=dice.mean(1,dtype=np.float64)
            sub=np.stack([dice[:,g].mean(1,dtype=np.float64) for g in groups],axis=1).mean(1)
            selected=int(np.argmax(mean))
            ranking.append(dict(recipe_index=ri,recipe=item['recipe'],parameters=params[selected],
                  parameter_index=selected,scan_dice=float(mean[selected]),subject_dice=float(sub[selected]),
                  point_hits=item['point_hits']))
            for pi,param in enumerate(params):
                writer.writerow([item['recipe']['id'],*[param[k] for k in ['post_temperature','bias','beta','volume_scale']],mean[pi],sub[pi],item['point_hits']])
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],len(d['recipe']['views']),d['recipe']['id']))
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,
          configurations=len(shortlist)*len(params),test_tuned=True,model_forwards=0))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=len(shortlist)*len(params),
          seconds=time.monotonic()-started,test_tuned=True,model_forwards=0))
