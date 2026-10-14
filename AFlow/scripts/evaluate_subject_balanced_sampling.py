"""One developer readout for two fixed sampler candidates; reuse fixed control counts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts import subject_balanced_protocol as p


@torch.inference_mode()
def main():
    p.check_freeze()
    out=p.RECORD/'RAW_DEVELOPMENT.json';pre=p.RECORD/'DEV_PREACCESS.json'
    if out.exists() or pre.exists():raise FileExistsError('one readout already started')
    admission=json.loads((p.RECORD/'TRAIN_PREACCESS.json').read_text())
    source_raw_path=p.SOURCE_RECORD/'RAW_DEVELOPMENT.json';p.checked(source_raw_path,admission['source_raw_development_sha256'])
    source_raw=json.loads(source_raw_path.read_text());reports={};pairing={}
    for seed in p.SEEDS:
        path=p.WORK/f'seed{seed}_subject_balanced/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['all_gates_passed'] and d['fixed_epochs']==168 and d['seed']==seed and d['arm']=='subject_balanced'
        assert d['fit_sha256']==p.FIT_SHA and not d['access']['development_metrics'] and not d['teacher_checkpoints']
        p.checked(Path(d['checkpoint']['path']),d['checkpoint']['sha256'])
        old=admission['source_controls'][str(seed)];p.checked(Path(old['train_report_path']),old['train_report_sha256']);control=json.loads(Path(old['train_report_path']).read_text())
        checks={'same_initial':d['initial_state_sha256']==control['initial_state_sha256'],'same_168_epochs':len(d['history'])==len(control['history'])==168,'same_121_steps':all(x['steps']==121 for x in d['history']),'same_augmentation_rng':all(x['augmentation_rng_sha256']==y['augmentation_rng_sha256'] for x,y in zip(d['history'],control['history'])),'sampling_weights_differ':d['sampling_weight_sha256']!=control['sampling_weight_sha256'],'sampler_changed':any(x['sampler_sha256']!=y['sampler_sha256'] for x,y in zip(d['history'],control['history']))}
        assert all(checks.values());pairing[str(seed)]=checks;reports[str(seed)]=d
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'index.json',p.INDEX_SHA);p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    _,dev=p.check_roles()
    device=torch.device('cuda:0');assert torch.cuda.is_available();torch.cuda.set_device(device);torch.set_num_threads(1)
    p.write_new(pre,{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'two_sampler_candidates_frozen_before_dev','source_raw_sha256':p.sha(source_raw_path),'source_controls':admission['source_controls'],'candidate_checkpoints':{s:d['checkpoint'] for s,d in reports.items()},'pairing':pairing,'dev_sha256':p.DEV_SHA,'evaluator_sha256':p.sha(Path(__file__)),'new_forwards_at_write':0})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,shuffle=False,num_workers=0)
    result={};comparisons={};gates={};forwards=0;start=time.monotonic()
    for i,seed in enumerate(p.SEEDS):
        control=source_raw['results'][f'seed{seed}_control']
        model=AutomaticPromptLocator(center_aware=False).to(device).eval()
        checkpoint=torch.load(reports[str(seed)]['checkpoint']['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True);rows=[]
        for batch in loader:
            image=batch['image'].to(device);truth=batch['label'].to(device)>0.5
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
            assert torch.isfinite(logits).all()
            for j,case in enumerate(batch['case_id']):
                pred=logits[j,0]>0;target=truth[j,0];intersection=int((pred&target).sum());n_pred=int(pred.sum());n_target=int(target.sum());assert n_target>0
                point=int(logits[j,0].flatten().argmax())
                rows.append({'case_id':case,'subject_id':batch['subject_id'][j],'dice':2*intersection/(n_pred+n_target),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':intersection,'predicted_voxels':n_pred,'target_voxels':n_target})
            forwards+=len(image)
        candidate=summarize(rows,2026091830+i)
        assert all(a['target_voxels']==b['target_voxels'] for a,b in zip(candidate['per_case'],control['per_case'],strict=True))
        c=compare(control,candidate,2026090830+i*100)
        result[str(seed)]={'reused_control':control,'candidate':candidate};comparisons[str(seed)]=c
        gates[str(seed)]={'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':c['point_delta']>=-1,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}
        del model,checkpoint;torch.cuda.empty_cache()
    average=statistics.fmean(c['scan_delta'] for c in comparisons.values());supported=all(all(g.values()) for g in gates.values()) and average>=.003
    assert forwards==360
    raw={'schema':'aflow.subject_balanced_sampling.reused_development','created_utc':datetime.now(timezone.utc).isoformat(),'results':result,'comparisons':comparisons,'gates':gates,'average_scan_delta':average,'all_gates_passed':supported,'pairing':pairing,'new_forwards':forwards,'seconds':time.monotonic()-start,'dev_preaccess_sha256':p.sha(pre),'source_raw_sha256':p.sha(source_raw_path),'access':{'reused_development180':True,'new_control_inference':False,'test270':False,'fixed_validation108':False,'outer':False,'confirmation':False,'external':False},'decision':'supported_reused_development_only' if supported else 'close_patient_equal_sampling_no_weight_mixture_epoch_seed_search'}
    p.write_new(out,raw)
    print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2))


if __name__=='__main__':main()
