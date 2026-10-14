"""Evaluate fixed checkpoints only after both seeds and paired engineering checks pass."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts import structured_background_protocol as p


def bootstrap(rows, seed):
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['subject_id']].append(r['dice'])
    ids=sorted(grouped)
    sums=np.asarray([sum(grouped[s]) for s in ids]); counts=np.asarray([len(grouped[s]) for s in ids]); means=sums/counts
    draw=np.random.default_rng(seed).integers(0,len(ids),size=(10000,len(ids)))
    return {'subject_95ci':np.quantile(means[draw].mean(1),[.025,.975]).tolist(),
            'scan_cluster_95ci':np.quantile(sums[draw].sum(1)/counts[draw].sum(1),[.025,.975]).tolist()}


def summarize(rows,seed):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['subject_id']].append(row['dice'])
    assert len(rows)==180 and len(grouped)==78
    return {'scan_dice':statistics.fmean(r['dice'] for r in rows),'subject_dice':statistics.fmean(statistics.fmean(v) for v in grouped.values()), 'point_hits':sum(r['point_hit'] for r in rows),'scans':180,'subjects':78, **bootstrap(rows,seed),'per_case':rows}


def compare(control,candidate,seed):
    pairs=[]
    for a,b in zip(control['per_case'],candidate['per_case'],strict=True):
        assert (a['case_id'],a['subject_id'])==(b['case_id'],b['subject_id'])
        pairs.append({'case_id':a['case_id'],'subject_id':a['subject_id'],'dice':b['dice']-a['dice']})
    interval=bootstrap(pairs,seed)
    return {'scan_delta':candidate['scan_dice']-control['scan_dice'], 'subject_delta':candidate['subject_dice']-control['subject_dice'], 'point_delta':candidate['point_hits']-control['point_hits'], 'improved':sum(r['dice']>0 for r in pairs), 'worse':sum(r['dice']<0 for r in pairs), 'tied':sum(r['dice']==0 for r in pairs), **interval}


@torch.inference_mode()
def main():
    p.check_freeze()
    output=p.RECORD/'RAW_DEVELOPMENT.json'
    pre_path=p.RECORD/'DEV_PREACCESS.json'
    if output.exists() or pre_path.exists():
        raise FileExistsError('development already started')
    receipt=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'images.f32',receipt['source_images_sha256']);p.checked(p.CACHE/'labels.u8',receipt['source_labels_sha256'])
    reports={};pairing={};hashes={}
    for seed in p.SEEDS:
        for arm in p.ARMS:
            key=f'seed{seed}_{arm}'
            report_path=p.WORK/key/'TRAIN_DONE.json'
            data=json.loads(report_path.read_text())
            assert data['status']=='complete' and data['all_gates_passed'] and data['fixed_epochs']==168
            assert data['fit_sha256']==p.FIT_SHA and not data['access']['development_metrics'] and not data['teacher_checkpoints']
            assert len(data['history'])==168 and all(r['steps']==121 for r in data['history'])
            p.checked(Path(data['checkpoint']['path']),data['checkpoint']['sha256'])
            reports[key]=data;hashes[key]={'train_report_sha256':p.sha(report_path),'checkpoint_sha256':data['checkpoint']['sha256']}
        a,b=(reports[f'seed{seed}_{arm}'] for arm in p.ARMS)
        pairing[str(seed)]={'initial_exact':a['initial_state_sha256']==b['initial_state_sha256'],'sampling_weights_exact':a['sampling_weight_sha256']==b['sampling_weight_sha256'], 'epoch_sampler_exact':all(x['sampler_sha256']==y['sampler_sha256'] for x,y in zip(a['history'],b['history'])),'epoch_augmentation_exact':all(x['augmentation_rng_sha256']==y['augmentation_rng_sha256'] and x['first_batch']==y['first_batch'] for x,y in zip(a['history'],b['history']))}
        assert all(pairing[str(seed)].values())
    _,dev=p.check_roles()
    device=torch.device('cuda:0')
    assert torch.cuda.is_available()
    torch.cuda.set_device(device);torch.set_num_threads(1)
    p.write_new(pre_path,{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'four_fixed_models_frozen_before_development','reports':hashes,'pairing':pairing,'dev_sha256':p.DEV_SHA,'evaluator_sha256':p.sha(Path(__file__)),'reused_development':True,'model_forwards_at_write':0})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,shuffle=False,num_workers=0)
    results={};start=time.monotonic();forwards=0
    for ordinal,(key,report) in enumerate(reports.items()):
        model=AutomaticPromptLocator(center_aware=False).to(device).eval()
        payload=torch.load(report['checkpoint']['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model_state_dict'],strict=True)
        rows=[]
        for batch in loader:
            image=batch['image'].to(device);truth=batch['label'].to(device)>0.5
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(image).segmentation_logits.float()
            assert torch.isfinite(logits).all()
            for i,case in enumerate(batch['case_id']):
                pred=logits[i,0]>0; target=truth[i,0]; inter=int((pred&target).sum());npred=int(pred.sum());ntarget=int(target.sum()); assert ntarget>0
                point=int(logits[i,0].flatten().argmax())
                rows.append({'case_id':case,'subject_id':batch['subject_id'][i],'dice':2*inter/(npred+ntarget),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':ntarget})
            forwards+=len(image)
        results[key]=summarize(rows,2026090730+1000+ordinal)
        del model,payload;torch.cuda.empty_cache()
    comparisons={};gates={}
    for i,seed in enumerate(p.SEEDS):
        c=compare(results[f'seed{seed}_control'],results[f'seed{seed}_candidate'],2026090730+i*100)
        comparisons[str(seed)]=c
        gates[str(seed)]={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':c['point_delta']>=-1,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}
    average=statistics.fmean(c['scan_delta'] for c in comparisons.values())
    supported=all(all(g.values()) for g in gates.values()) and average>=.003
    assert forwards==720
    raw={'schema':'aflow.structured_background_aux.reused_development','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':comparisons,'gates':gates,'average_scan_delta':average,'average_scan_gain_gate':average>=.003,'all_gates_passed':supported,'pairing':pairing,'forwards':forwards,'seconds':time.monotonic()-start,'dev_preaccess_sha256':p.sha(pre_path),'evaluator_sha256':p.sha(Path(__file__)),'access':{'reused_development180':True,'test270':False,'fixed_validation108':False,'outer':False,'confirmation':False,'external':False},'decision':'supported_reused_development_only' if supported else 'close_current_recipe_no_weight_epoch_seed_search'}
    p.write_new(output,raw)
    print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2))


if __name__=='__main__':
    main()
