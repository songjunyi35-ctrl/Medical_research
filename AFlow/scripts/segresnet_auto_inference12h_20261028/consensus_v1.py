"""Label-free prediction consensus: hard-STAPLE, fractional-vote EM, agreement weights."""
import csv
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import threshold_scores,summary_arrays
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram


def definitions():
    out=[]
    for vote_threshold in [.2,.35,.5,.65]:
        for update_prior in [False,True]:
            for iterations in [5,20,50]:
                out.append(dict(id=f'staple_t{vote_threshold}_prior{update_prior}_i{iterations}',kind='consensus',mode='hard_em',vote_threshold=vote_threshold,update_prior=update_prior,iterations=iterations,views=list(range(8)),neural_case_forwards=8))
    for update_prior in [False,True]:
        for iterations in [5,20,50]:
            out.append(dict(id=f'fractional_em_prior{update_prior}_i{iterations}',kind='consensus',mode='fractional_em',update_prior=update_prior,iterations=iterations,views=list(range(8)),neural_case_forwards=8))
    for threshold in [.2,.35,.5,.65]:
        for representation in ['probability','logit']:
            for exponent in [.5,1.,2.,4.,8.,16.]:
                out.append(dict(id=f'agreement_t{threshold}_{representation}_w{exponent}',kind='consensus',mode='agreement_weight',vote_threshold=threshold,representation=representation,exponent=exponent,views=list(range(8)),neural_case_forwards=8))
            for k in [2,3,4,5,6]:
                out.append(dict(id=f'self_select_t{threshold}_{representation}_k{k}',kind='consensus',mode='self_select',vote_threshold=threshold,representation=representation,k=k,views=list(range(8)),neural_case_forwards=8))
    return out


def em_stream(votes,update_prior,iterations=(5,20,50)):
    q=votes.mean(0);prior=q.mean().clamp(1e-7,1-1e-7)
    for iteration in range(1,max(iterations)+1):
        positive=q.sum(dtype=torch.float64).clamp_min(1e-10)
        negative=(1-q).sum(dtype=torch.float64).clamp_min(1e-10)
        sensitivity=(votes*q[None]).flatten(1).sum(1,dtype=torch.float64)/positive
        specificity=((1-votes)*(1-q[None])).flatten(1).sum(1,dtype=torch.float64)/negative
        sensitivity=sensitivity.clamp(.01,1-1e-5).float();specificity=specificity.clamp(.01,1-1e-5).float()
        if update_prior:prior=q.mean().clamp(1e-7,1-1e-7)
        a=torch.log(sensitivity/(1-specificity));b=torch.log((1-sensitivity)/specificity)
        logodds=torch.logit(prior)+b.sum()
        z=logodds+(votes*(a-b)[:,None,None,None]).sum(0)
        q=z.sigmoid()
        if iteration in iterations:yield iteration,q


def agreement(p,threshold):
    votes=p>threshold;consensus=p.mean(0)>threshold
    tp=(votes&consensus[None]).flatten(1).sum(1).double()
    den=votes.flatten(1).sum(1).double()+consensus.sum()
    return 2*tp/den.clamp_min(1)


def prepare(cid,source):
    logits=[torch.from_numpy(np.load(TMP/'groups'/source/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
    return dict(logits=logits,p=torch.stack([z.sigmoid() for z in logits]))


def probability(cache,recipe):
    p=cache['p'];mode=recipe['mode']
    if mode in ['hard_em','fractional_em']:
        votes=(p>recipe['vote_threshold']).float() if mode=='hard_em' else p
        return next(em_stream(votes,recipe['update_prior'],iterations=[recipe['iterations']]))[1]
    scores=agreement(p,recipe['vote_threshold'])
    if mode=='agreement_weight':weights=torch.softmax(scores*recipe['exponent'],0)
    elif mode=='self_select':
        order=torch.argsort(scores,descending=True,stable=True);weights=torch.zeros_like(scores);weights[order[:recipe['k']]]=1/recipe['k']
    else:raise ValueError(mode)
    values=[p[i] for i in range(8)] if recipe['representation']=='probability' else cache['logits']
    result=torch.zeros_like(values[0])
    for i in range(8):result.add_(values[i],alpha=float(weights[i]))
    return result if recipe['representation']=='probability' else result.sigmoid()


def gate():
    torch.manual_seed(1);mask=torch.rand(7,9,11,device='cuda')>.8;votes=mask[None].expand(8,*mask.shape).float()
    for update in [False,True]:
        a=next(em_stream(votes,update,iterations=[5]))[1]
        b=next(em_stream(votes,update,iterations=[5]))[1]
        assert torch.equal(a,b) and torch.equal(a>.5,mask) and torch.isfinite(a).all()
    for value in [0.,1.]:
        a=next(em_stream(torch.full_like(votes,value),True,iterations=[5]))[1]
        assert torch.isfinite(a).all() and torch.equal(a>.5,torch.full_like(mask,bool(value)))
    return dict(at=now(),status='passed',identical_votes_recovered=True,empty_and_full_finite=True,repeat_exact=True,GT_input=False)


@torch.inference_mode()
def execute(job):
    verify_job(job);folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True);write(folder/'GATE.json',gate())
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    rows,geometry,_,labels=data();recipes=definitions();lookup={r['id']:i for i,r in enumerate(recipes)}
    thresholds=np.round(np.arange(.025,.9751,.025),6).tolist();n=len(recipes);nt=len(thresholds)
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,test_tuned=True,
          caveat='same-model views are correlated; fractional-vote EM is an engineering variant, not PSTAPLE replication; point uses mean8 probability'))
    dp=scratch/'dice.npy';hp=scratch/'hits.npy'
    dice=np.lib.format.open_memmap(dp,mode='r+' if dp.exists() else 'w+',dtype=np.float32,shape=(n,nt,270))
    hits=np.lib.format.open_memmap(hp,mode='r+' if hp.exists() else 'w+',dtype=bool,shape=(n,270))
    start=time.monotonic()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];marker=folder/f'case{slot:03}.json'
        if marker.exists():continue
        cache=prepare(cid,job['core_source']);target=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda()
        point_hit=bool(target.flatten()[int(cache['p'].mean(0).argmax())])
        def score(name,p):
            ri=lookup[name];h=histogram(p,target);dice[ri,:,slot]=threshold_scores(h,thresholds)[0];hits[ri,slot]=point_hit
        for threshold in [.2,.35,.5,.65]:
            for update in [False,True]:
                for iterations,q in em_stream((cache['p']>threshold).float(),update):score(f'staple_t{threshold}_prior{update}_i{iterations}',q)
        for update in [False,True]:
            for iterations,q in em_stream(cache['p'],update):score(f'fractional_em_prior{update}_i{iterations}',q)
        for recipe in recipes:
            if recipe['mode'] in ['agreement_weight','self_select']:score(recipe['id'],probability(cache,recipe))
        dice.flush();hits.flush();write(marker,dict(at=now(),case_id=cid))
        status('label_free_consensus',job=job['id'],completed=slot+1,total=270,configurations=n*nt,seconds=time.monotonic()-start)
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
        guard();cid=row['case_id'];cache=prepare(cid,job['core_source']);target=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda()
        for item in chosen:
            out=scratch/'histograms'/str(item['recipe_index'])/(cid+'.npy')
            if not out.exists():save_array(out,histogram(probability(cache,item['recipe']),target).astype(np.uint32))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=n*nt,recipes=n,histogram_shortlist=len(chosen),
          seconds=time.monotonic()-start,test_tuned=True,model_forwards=0,dice_sha256=sha(dp),hits_sha256=sha(hp)))
