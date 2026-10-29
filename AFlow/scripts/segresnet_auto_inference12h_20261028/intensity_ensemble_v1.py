"""Finite cached ensemble of the28 preplanned intensity groups; image-only choices."""
import csv
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard as cached_guard
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import stable_size
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.case_consistency_v1 import prepare as core_prepare,probability as core_probability
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v1 import summary_arrays

BASE=dict(id='case_agree_t0.2_w24.0_k8_logit_T4.0',kind='case_consistency',vote_threshold=.2,exponent=24.,keep_views=8,representation='logit',view_temperature=4.,views=list(range(8)),neural_case_forwards=8)
MODES=['mean_probability','mean_logit_group_probability','trim1_probability','vote_probability','mean_minus_std','mean_plus_std']


def definitions(job):
    control=dict(id='core24_brain_control',kind='group_mix',multi_group_fusion=True,groups=[],family='control',group='__core_control__',group_sources=[],base='base8soft_case24',mode='mean_probability',group_temperature=1.,alpha=0.,views=list(range(8)),neural_case_forwards=8,brain_support_margin=0)
    out=[control]
    def item(family,mode,temperature,alpha,exponent=None):
        groups=job['families'][family];value=dict(id=f'{family}_{mode}_T{temperature}_a{alpha}_w{exponent}',kind='group_mix',multi_group_fusion=True,groups=groups,group=groups[0],group_sources=groups,base='base8soft_case24',family=family,mode=mode,group_temperature=temperature,alpha=alpha,views=list(range(8)),neural_case_forwards=8*len(groups)+(8 if alpha<1 else 0),brain_support_margin=0)
        if exponent is not None:value['case_exponent']=exponent
        return value
    for family in job['families']:
        for mode in MODES:
            for temperature in [.5,1.,2.,4.]:
                for alpha in [.05,.1,.2,.35,.5,1.]:out.append(item(family,mode,temperature,alpha))
    for family in ['all','photometric','single_axis']:
        for exponent in [-8.,0.,8.,16.]:
            for mode in ['case_probability','case_logit_group_probability']:
                for temperature in [.5,1.,2.]:
                    for alpha in [.1,.35,1.]:out.append(item(family,mode,temperature,alpha,exponent))
    assert len(out)==1945
    return out


def mean(values):
    result=torch.zeros_like(values[0])
    for value in values:result.add_(value,alpha=1/len(values))
    return result


def prepare(cid,job):
    index=read(CACHE/'index.json');slot=index['case_ids'].index(cid);sl=slices(index['geometry'][cid]);images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(270,*CANVAS));image=np.array(images[slot][sl],np.float32)
    logits=[torch.from_numpy(np.load(TMP/'groups'/job['core_source']/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
    base=core_probability(core_prepare(logits),BASE)
    probs={gid:torch.from_numpy(np.load(TMP/'groups'/gid/'mean_probability'/(cid+'.npy'))).cuda() for gid in job['group_sources']}
    return dict(base=base,probs=probs,valid=torch.from_numpy(supports(image)[0]).cuda(),aggregates={})


def aggregate(cache,recipe):
    key=(recipe['family'],recipe['mode'],recipe.get('case_exponent'))
    if key in cache['aggregates']:return cache['aggregates'][key]
    values=[cache['probs'][g] for g in recipe['groups']];mode=recipe['mode'];n=len(values)
    if mode=='mean_probability':value=mean(values)
    elif mode=='mean_logit_group_probability':value=mean([torch.logit(p.clamp(1e-7,1-1e-7)) for p in values])
    elif mode.startswith('case_'):
        votes=torch.stack([p[cache['valid']]>.2 for p in values]);reference=mean(values)[cache['valid']]>.2
        tp=(votes&reference[None]).sum(1).double();den=votes.sum(1).double()+reference.sum();weights=torch.softmax((2*tp/den.clamp_min(1))*recipe['case_exponent'],0).cpu().tolist();value=torch.zeros_like(values[0])
        for p,weight in zip(values,weights):value.add_(torch.logit(p.clamp(1e-7,1-1e-7)) if mode=='case_logit_group_probability' else p,alpha=weight)
    else:
        stack=torch.stack(values);average=mean(values)
        if mode=='trim1_probability':value=(average*n-stack.min(0).values-stack.max(0).values)/(n-2) if n>2 else average
        elif mode=='vote_probability':value=(stack>.5).float().mean(0)
        elif mode=='mean_minus_std':value=average-.25*stack.std(0,unbiased=False)
        elif mode=='mean_plus_std':value=average+.25*stack.std(0,unbiased=False)
        else:raise ValueError(mode)
    if len(cache['aggregates'])>=8:cache['aggregates'].pop(next(iter(cache['aggregates'])))
    cache['aggregates'][key]=value;return value


def probability(cache,recipe):
    if recipe['alpha']==0:return cache['base']
    value=aggregate(cache,recipe)
    logit=value if recipe['mode'] in ['mean_logit_group_probability','case_logit_group_probability'] else torch.logit(value.clamp(1e-7,1-1e-7))
    p=(logit/recipe['group_temperature']).sigmoid()
    if recipe['alpha']==1:return p
    return cache['base']*(1-recipe['alpha'])+p*recipe['alpha']


def threshold_scores(hist,thresholds,total_positive):
    tail=np.cumsum(hist[:,::-1].astype(np.int64),1)[:,::-1];cuts=np.floor(np.asarray(thresholds)*65535).astype(np.int64)+1
    tp=tail[1,cuts];den=tail[:,cuts].sum(0)+total_positive
    return np.where(den>0,(2*tp).astype(np.float32)/np.maximum(den,1).astype(np.float32),1.).astype(np.float32)


@torch.inference_mode()
def gate(job):
    torch.cuda.set_per_process_memory_fraction(.07);numeric_mode();torch.cuda.reset_peak_memory_stats()
    shape=tuple(max(g['extent'][d] for g in read(CACHE/'index.json')['geometry'].values()) for d in range(3))
    generator=torch.Generator(device='cuda').manual_seed(20261029);probs={g:torch.rand(shape,device='cuda',generator=generator) for g in job['group_sources']};base=torch.rand(shape,device='cuda',generator=generator);cache=dict(base=base,probs=probs,valid=torch.ones(shape,device='cuda',dtype=torch.bool),aggregates={})
    recipes=definitions(job)
    for mode in MODES:
        r=next(r for r in recipes if r['family']=='all' and r['mode']==mode and r['alpha']==1 and r['group_temperature']==1)
        a=probability(cache,r);b=probability(cache,r);assert torch.equal(a,b) and torch.isfinite(a).all() and (a>=0).all() and (a<=1).all()
    for logit in [False,True]:
        mode='case_logit_group_probability' if logit else 'case_probability';r=next(r for r in recipes if r['family']=='all' and r['mode']==mode and r['case_exponent']==0 and r['alpha']==1 and r['group_temperature']==1)
        expected=next(x for x in recipes if x['family']=='all' and x['mode']==('mean_logit_group_probability' if logit else 'mean_probability') and x['alpha']==1 and x['group_temperature']==1)
        assert torch.equal(probability(cache,r),probability(cache,expected))
    assert torch.equal(probability(cache,recipes[0]),base)
    write(R/'INTENSITY_ENSEMBLE_GATE.json',dict(at=now(),state='passed',source_sha256=sha(Path(__file__)),recipes=len(recipes),native_max_shape=shape,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,lambda0_matches_arithmetic_exact=True,control_identity_exact=True,finite_bounded_repeat=True,patient_forward=False,network_forward=False,GT_access=False))
    print(dict(state='passed',recipes=len(recipes),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


@torch.inference_mode()
def execute(job):
    verify_job(job);assert read(R/'INTENSITY_ENSEMBLE_GATE.json')['state']=='passed';assert read(R/'INTENSITY_ENSEMBLE_GATE.json')['source_sha256']==sha(Path(__file__))
    folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True);scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    recipes=definitions(job);thresholds=np.round(np.arange(.025,.9751,.025),6).tolist();required=len(recipes)*39*270*4+len(recipes)*270+8*270*(65536*2*4+128)+8*270*11475*6+32*2**20
    assert required<1536*2**20
    assert stable_size(TMP)+stable_size(CACHE)+read(R/'MAIN_SCRATCH_BUDGET.json')['bytes']+required<300*2**30
    for gid in job['group_sources']:assert (R/'jobs'/gid/'DONE.json').exists() and len(list((TMP/'groups'/gid/'mean_probability').glob('*.npy')))==270
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,thresholds=thresholds,source_sha256=sha(Path(__file__)),configs=len(recipes)*39,full_original_GT_domain=True,model_forwards_added=0,test_tuned=True,scratch_upper_bound_bytes=required))
    rows,geometry,_,labels=data();dp=scratch/'dice.npy';hp=scratch/'hits.npy';dice=np.lib.format.open_memmap(dp,mode='r+' if dp.exists() else 'w+',dtype=np.float32,shape=(len(recipes),39,270));hits=np.lib.format.open_memmap(hp,mode='r+' if hp.exists() else 'w+',dtype=bool,shape=(len(recipes),270));totals={};start=time.monotonic()
    for slot,row in enumerate(rows):
        cached_guard();assert read(R/'RESOURCE_CURRENT.json')['token']==job['resource_token'];cid=row['case_id'];target=np.array(labels[slot][slices(geometry[cid])]>0);totals[cid]=int(target.sum())
        if (folder/f'case{slot:03}.json').exists():continue
        cache=prepare(cid,job);valid=cache['valid'];gt=torch.from_numpy(target).cuda()
        for ri,recipe in enumerate(recipes):
            p=probability(cache,recipe);h=histogram(p[valid],gt[valid]);dice[ri,:,slot]=threshold_scores(h,thresholds,totals[cid]);hits[ri,slot]=bool(gt.flatten()[int(p.argmax())])
        dice.flush();hits.flush();write(folder/f'case{slot:03}.json',dict(at=now(),case_id=cid))
        if slot%5==0:status('intensity_ensemble_scoring',job=job['id'],completed=slot+1,total=270,seconds=time.monotonic()-start)
    means,subjects=summary_arrays(np.asarray(dice),rows);ranking=[]
    for ri,recipe in enumerate(recipes):
        ti=int(means[ri].argmax());ranking.append(dict(recipe_index=ri,recipe=recipe,threshold=thresholds[ti],threshold_index=ti,scan_dice=float(means[ri,ti]),subject_dice=float(subjects[ri,ti]),point_hits=int(hits[ri].sum())))
    ranking.sort(key=lambda r:(-r['scan_dice'],-r['subject_dice'],r['recipe_index']));chosen=[r for r in ranking if r['recipe_index']!=0][:7]+[next(r for r in ranking if r['recipe_index']==0)]
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],best_by_recipe=ranking,calibration_shortlist=chosen,test_tuned=True));write(folder/'GT_TOTALS.json',totals)
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as handle:
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n');writer.writerow(['recipe','threshold','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for ti,threshold in enumerate(thresholds):writer.writerow([recipe['id'],threshold,means[ri,ti],subjects[ri,ti],int(hits[ri].sum())])
    for slot,row in enumerate(rows):
        cached_guard();cid=row['case_id'];cache=prepare(cid,job);gt=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda()
        for r in chosen:save_array(scratch/'histograms'/str(r['recipe_index'])/(cid+'.npy'),histogram(probability(cache,r['recipe'])[cache['valid']],gt[cache['valid']]).astype(np.uint32))
    write(folder/'DONE.json',dict(at=now(),state='complete',configurations=len(recipes)*39,best=ranking[0],histogram_shortlist=8,seconds=time.monotonic()-start,model_forwards_added=0,test_tuned=True,dice_sha256=sha(dp),hits_sha256=sha(hp)));print(ranking[0])

if __name__=='__main__':gate(read(R/'INTENSITY_ENSEMBLE_JOB_DRAFT.json'))
