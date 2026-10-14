"""Posthoc Monte Carlo precision of historical near-zero bootstrap endpoints."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'archive/reviews/2026-09-07_ucsd56_bootstrap_precision'
RAW=ROOT/'archive/research_records/2026-09-06_ucsd_brats2024test_frozen_panel/RAW_UCSD_BRATS2024TEST56.json'


def estimate(values,seed,count):
    values=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(seed)
    pieces=[]
    for start in range(0,count,10000):
        draw=rng.integers(0,len(values),(min(10000,count-start),len(values)))
        pieces.append(values[draw].mean(1))
    return np.quantile(np.concatenate(pieces),[.025,.975]).tolist()


def main():
    if (OUT/'RESULT.json').exists():raise FileExistsError('immutable audit output exists')
    body=RAW.read_bytes();digest=hashlib.sha256(body).hexdigest()
    assert digest=='74466d1cdc56d864d48e5d8604e00d004ecdb636882366546b4738a1b794201c'
    data=json.loads(body);a=data['full972_kd168']['identity']['per_case'];b=data['gcsam184']['identity']['per_case']
    delta=[];group=defaultdict(list)
    for left,right in zip(a,b,strict=True):
        assert (left['case_id'],left['subject_id'])==(right['case_id'],right['subject_id'])
        d=right['locator_dsc']-left['locator_dsc'];delta.append(d);group[left['subject_id']].append(d)
    subject=[statistics.fmean(group[s]) for s in sorted(group)]
    result={}
    for name,values,offset in [('scan',delta,0),('subject',subject,100)]:
        estimates=[{'seed':seed+offset,'ci':estimate(values,seed+offset,10000)} for seed in range(2026090801,2026090821)]
        result[name]={'n':len(values),'point_delta':statistics.fmean(values),'replicates_10000':estimates,'upper_bound_min':min(x['ci'][1] for x in estimates),'upper_bound_max':max(x['ci'][1] for x in estimates),'replicates_cross_zero':sum(x['ci'][0]<=0<=x['ci'][1] for x in estimates),'million_seed':2026090901+offset,'million_ci':estimate(values,2026090901+offset,1000000)}
    report={'schema':'aflow.ucsd56.bootstrap_mc_precision','created_utc':datetime.now(timezone.utc).isoformat(),'status':'posthoc_numerical_sensitivity_not_replacement','result':result,'original_comparison':data['comparisons']['gcsam_identity_minus_full_identity'],'raw_sha256':digest,'plan_sha256':hashlib.sha256((OUT/'PLAN.md').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'access':{'image_arrays':False,'checkpoint':False,'new_forward':0,'current_development180':False},'forbidden_use':'do not replace original preregistered intervals/gates or select seed/checkpoint/loss'}
    (OUT/'RESULT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k!='replicates_10000'} for name,row in result.items()},indent=2))


if __name__=='__main__':main()
