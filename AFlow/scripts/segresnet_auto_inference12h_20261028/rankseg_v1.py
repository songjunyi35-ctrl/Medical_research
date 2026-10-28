"""Official private RankSEG0.0.8 BA/RMA decoders; no environment changes."""
import sys
import csv
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.core_fusions_v3 import prepare,probability,definitions,summary_arrays


def vendor():
    provenance=read(R/'RANKSEG_PACKAGE_PROVENANCE.json');folder=Path(provenance['private_module_path'])
    for name,digest in provenance['source_sha256'].items():assert sha(folder/name)==digest,name
    sys.path.insert(0,str(folder))
    from rankseg import RankSEG
    return RankSEG


def parameters():
    return [dict(solver=solver,post_temperature=t,smooth=smooth,pruning_prob=0.,working_precision='float64',safe_screening=False)
            for solver in ['RMA','BA'] for t in [.75,1.,1.5,2.,3.] for smooth in [0.,10.,100.,1000.]]


def decode(p,param,RankSEG):
    p=torch.sigmoid(torch.logit(p.double().clamp(1e-7,1-1e-7))/param['post_temperature'])
    extra=dict(safe_screening=False) if param['solver']=='RMA' else {}
    decoder=RankSEG(metric='dice',solver=param['solver'],output_mode='multilabel',pruning_prob=0.,smooth=param['smooth'],**extra)
    return decoder(p.reshape(1,1,*p.shape))[0,0]


@torch.inference_mode()
def execute(job):
    verify_job(job);RankSEG=vendor();folder=R/'analyses'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    scratch=TMP/'analyses'/job['id'];scratch.mkdir(parents=True,exist_ok=True)
    ranked=read(R/'analyses'/job['calibration_rank']/'RANKING.json')['best_by_recipe'];selected=ranked[:2]
    allrecipes=definitions();lookup={r['id']:r for r in allrecipes};recipes=[x['recipe'] for x in selected]
    for name in ['meanp_T1.0_subset220','meanp_T1.0_subset255']:
        if name not in {x['id'] for x in recipes}:recipes.append(lookup[name])
    params=parameters();rows,geometry,_,labels=data();n=len(recipes);m=len(params)
    write(folder/'PLAN.json',dict(at=now(),recipes=recipes,parameters=params,package_provenance=read(R/'RANKSEG_PACKAGE_PROVENANCE.json'),
          solver_claim='official BA/RMA approximations; not exact Poisson-binomial enumeration',selection='top2 test-calibrated aggregations plus old5/full8',test_tuned=True))
    dp=scratch/'dice.npy';hp=scratch/'hits.npy'
    dice=np.lib.format.open_memmap(dp,mode='r+' if dp.exists() else 'w+',dtype=np.float32,shape=(n,m,270))
    hits=np.lib.format.open_memmap(hp,mode='r+' if hp.exists() else 'w+',dtype=bool,shape=(n,270))
    start=time.monotonic()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];marker=folder/f'case{slot:03}.json'
        if marker.exists():continue
        logits=[torch.from_numpy(np.load(TMP/'groups'/job['core_source']/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
        cache=prepare(logits);target=torch.from_numpy(np.array(labels[slot][slices(geometry[cid])]>0)).cuda();positive=int(target.sum())
        for ri,recipe in enumerate(recipes):
            p=probability(cache,recipe);hits[ri,slot]=bool(target.flatten()[int(p.argmax())])
            for pi,param in enumerate(params):
                mask=decode(p,param,RankSEG);tp=int((mask&target).sum());den=int(mask.sum())+positive
                dice[ri,pi,slot]=float(np.float32(2*tp)/np.float32(den)) if den else 1.
                if slot==0 and pi in [0,len(params)-1]:
                    repeated=decode(p,param,RankSEG);assert torch.equal(mask,repeated),'official solver repeat mask mismatch'
        dice.flush();hits.flush();write(marker,dict(at=now(),case_id=cid))
        status('official_rankseg_scoring',job=job['id'],completed=slot+1,total=270,configurations=n*m,seconds=time.monotonic()-start)
    means,subjects=summary_arrays(np.asarray(dice),rows);ranking=[]
    with (folder/'ALL_CONFIGURATIONS.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['recipe','solver','post_temperature','smooth','scan_dice','subject_dice','point_hits'])
        for ri,recipe in enumerate(recipes):
            for pi,param in enumerate(params):
                item=dict(recipe_index=ri,recipe=recipe,parameters=param,parameter_index=pi,scan_dice=float(means[ri,pi]),subject_dice=float(subjects[ri,pi]),point_hits=int(hits[ri].sum()))
                ranking.append(item);w.writerow([recipe['id'],param['solver'],param['post_temperature'],param['smooth'],means[ri,pi],subjects[ri,pi],int(hits[ri].sum())])
    ranking.sort(key=lambda d:(-d['scan_dice'],-d['subject_dice'],len(d['recipe']['views']),d['recipe_index'],d['parameter_index']))
    write(folder/'RANKING.json',dict(at=now(),best=ranking[0],all_configurations=ranking,total_configurations=n*m,test_tuned=True))
    write(folder/'DONE.json',dict(at=now(),state='complete',best=ranking[0],configurations=n*m,seconds=time.monotonic()-start,test_tuned=True,
          model_forwards=0,dice_sha256=sha(dp),hits_sha256=sha(hp)))
