"""Fit-image-only online parity and timing for frozen primary fold winners."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from sammed3d.automatic_fixed_workflow import restored_atom,online_logits,fuse_atoms
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_automatic_workflow_search'


@torch.inference_mode()
def main():
    if (RECORD/'ONLINE_PREACCESS.json').exists():raise FileExistsError('online benchmark already started')
    analysis=json.loads((RECORD/'ANALYSIS.json').read_text())
    verified=json.loads((RECORD/'SCORE_VERIFICATION.json').read_text())
    assert verified['status']=='passed' and verified['candidate_case_scores_verified']==46080
    assert verified['raw_sha256']==analysis['raw_sha256']
    pre=json.loads((RECORD/'DEV_PREACCESS.json').read_text())
    for name,digest in pre['files'].items():p.checked(p.ROOT/name,digest)
    policies={};weights={}
    for report in pre['models']:
        seed=str(report['seed']);p.checked(report['checkpoint'],report['sha256'])
        policies[seed]={'full8':list(range(8))};weights[seed]={}
        for r in analysis['selection_receipts']:
            if str(r['seed'])==seed and r['controller']=='exhaustive':
                name=f"fold{r['fold']}";policies[seed][name]=r['selection']['views']
                weights[seed][name]=sum(row['subject_id'] in r['assessment_subjects'] for row in p.rows(p.DEV))/180
        assert len(policies[seed])==4 and abs(sum(weights[seed].values())-1)<1e-12
    p.checked(p.FIT,p.FIT_SHA)
    fit=p.rows(p.FIT)[:24]
    p.checked(p.CACHE/'index.json',pre['index_sha256']);p.checked(p.CACHE/'images.f32',pre['images_sha256'])
    expected=sum(16+53*sum(map(len,values.values())) for values in policies.values())
    p.write_new(RECORD/'ONLINE_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_sha256':p.sha(RECORD/'ANALYSIS.json'),'verification_sha256':p.sha(RECORD/'SCORE_VERIFICATION.json'),
        'benchmark_sha256':p.sha(Path(__file__)),'models':pre['models'],'policies':policies,'assessment_scan_weights':weights,
        'fit_cases':[r['case_id'] for r in fit],'expected_case_forwards':expected,
        'measurements_per_policy':48,'warmup_per_policy':3,'parity_cases_per_model':2,
        'forwards_at_write':0,'labels_opened':False,
        'scope':'H2D through mask-point D2H; excludes file reads/preprocessing/model loading'})
    idx=json.loads((p.CACHE/'index.json').read_text())
    lookup={case:i for i,case in enumerate(idx['case_ids'])}
    images=np.memmap(p.CACHE/'images.f32',dtype=np.float32,mode='r',shape=(len(lookup),128,128,128))
    tensors=[torch.from_numpy(np.array(images[lookup[r['case_id']]],copy=True))[None,None] for r in fit]
    torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    rows=[];parity=[];calls=[0];started=time.monotonic();peaks={}
    def hook(model,args,output):calls[0]+=len(args[0])
    def deadline():
        if time.monotonic()-started>900:raise TimeoutError('15-minute online benchmark limit')
    try:
        for report in pre['models']:
            seed=str(report['seed']);model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model_state_dict'],strict=True);handle=model.register_forward_hook(hook)
            torch.cuda.reset_peak_memory_stats(device)
            for ci,cpu in enumerate(tensors[:2]):
                deadline();image=cpu.to(device);atoms=[restored_atom(model,image,k) for k in range(8)]
                for name,views in policies[seed].items():
                    n=calls[0];actual=online_logits(model,image,views);reference=fuse_atoms(atoms,views)
                    assert torch.isfinite(actual).all() and torch.equal(actual,reference)
                    assert calls[0]-n==len(views)
                    parity.append({'seed':seed,'case':fit[ci]['case_id'],'workflow':name,'views':views,'max_abs_error':0})
                del atoms,actual,reference,image
            def measure(cpu,name):
                deadline();n=calls[0];torch.cuda.synchronize();t=time.perf_counter()
                image=cpu.to(device);logits=online_logits(model,image,policies[seed][name])
                mask=(logits>0).cpu();point=int(logits.flatten().argmax().cpu());torch.cuda.synchronize()
                duration=time.perf_counter()-t
                assert mask.shape==(1,1,128,128,128) and 0<=point<128**3
                assert calls[0]-n==len(policies[seed][name])
                return duration
            for name in policies[seed]:
                for cpu in tensors[:3]:measure(cpu,name)
            names=list(policies[seed])
            for repeat in range(2):
                for ci,cpu in enumerate(tensors):
                    for name in (names if (ci+repeat)%2==0 else names[::-1]):
                        rows.append({'seed':seed,'case_id':fit[ci]['case_id'],'repeat':repeat,'workflow':name,
                                     'seconds':measure(cpu,name),'case_forwards':len(policies[seed][name])})
            peaks[seed]=torch.cuda.max_memory_allocated(device)
            handle.remove();del model,ck;torch.cuda.empty_cache()
        assert calls[0]==expected and len(rows)==384 and len(parity)==16
        summaries={}
        for seed in policies:
            lookup_time={(r['case_id'],r['repeat'],r['workflow']):r['seconds'] for r in rows if r['seed']==seed}
            ratio=[];selected_times=[]
            for case in fit:
                for repeat in range(2):
                    weighted=sum(w*lookup_time[(case['case_id'],repeat,name)] for name,w in weights[seed].items())
                    selected_times.append(weighted)
                    ratio.append(lookup_time[(case['case_id'],repeat,'full8')]/weighted)
            summaries[seed]={'per_policy_mean_seconds':{name:statistics.fmean(r['seconds'] for r in rows if r['seed']==seed and r['workflow']==name) for name in policies[seed]},
                'assessment_scan_weighted_selected_mean_seconds':statistics.fmean(selected_times),
                'paired_median_speedup':statistics.median(ratio),
                'speedup_at_least1_5':statistics.median(ratio)>=1.5}
        raw={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'parity':parity,
             'summary':summaries,'peak_bytes':peaks,'case_forwards':calls[0],'seconds':time.monotonic()-started,
             'hardware':torch.cuda.get_device_name(device),'online_preaccess_sha256':p.sha(RECORD/'ONLINE_PREACCESS.json'),
             'all_gates_passed':all(v['speedup_at_least1_5'] for v in summaries.values()),
             'labels_opened':False,'test_access':False,
             'interpretation':'weighted timing across fixed fold policies; no label-free case-adaptive router or single final workflow claimed'}
        p.write_new(RECORD/'ONLINE_BENCHMARK.json',raw)
        print(json.dumps({k:v for k,v in raw.items() if k not in ('rows','parity')},indent=2),flush=True)
    except BaseException as error:
        p.write_new(RECORD/'ONLINE_FAILURE.json',{'error':repr(error),'case_forwards':calls[0],'partial_rows':rows,'labels_opened':False})
        raise


if __name__=='__main__':main()
