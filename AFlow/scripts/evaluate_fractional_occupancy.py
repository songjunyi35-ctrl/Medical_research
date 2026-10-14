"""Two fixed target-representation candidates scored against unchanged hard GT."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset,write_json_atomic
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts import fractional_occupancy_protocol as p


def pairing(candidate,control):
    a,b=candidate['history'],control['history']
    return {'same_initial':candidate['initial_state_sha256']==control['initial_state_sha256'],
            'same_weights':candidate['sampling_weight_sha256']==control['sampling_weight_sha256'],
            'same168':len(a)==len(b)==168,
            'same121steps':all(x['steps']==y['steps']==121 for x,y in zip(a,b,strict=True)),
            'same_sampler':all(x['sampler_sha256']==y['sampler_sha256'] for x,y in zip(a,b,strict=True)),
            'same_augmentation_rng':all(x['augmentation_rng_sha256']==y['augmentation_rng_sha256'] for x,y in zip(a,b,strict=True)),
            'same_first_batch_images':all(x['first_batch']['image_sha256']==y['first_batch']['image_sha256'] for x,y in zip(a,b,strict=True)),
            'target_labels_actually_changed':any(x['first_batch']['labels_sha256']!=y['first_batch']['labels_sha256'] for x,y in zip(a,b,strict=True)),
            'export_exact':candidate['export_max_abs_error']==0}


def performance_gates(c):
    return {'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_safety':c['point_delta']>=-1,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}


@torch.inference_mode()
def main():
    p.check_freeze();out=p.RECORD/'RAW_DEVELOPMENT.json';pre=p.RECORD/'DEV_PREACCESS.json'
    code_freeze=json.loads((p.RECORD/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json').read_text())
    for name,digest in code_freeze['files'].items():p.checked(p.ROOT/name,digest)
    if out.exists() or pre.exists():raise FileExistsError('one development readout only')
    assert json.loads((p.RECORD/'TRAINING_DONE.json').read_text())['status']=='both_fixed168_candidates_complete'
    assert json.loads((p.RECORD/'TRAINING_REPLAY_FINAL.json').read_text())['status']=='passed_full168'
    frozen=json.loads((p.RECORD/'IMPLEMENTATION_FREEZE.json').read_text())
    source_path=p.source.RECORD/'RAW_DEVELOPMENT.json';p.checked(source_path,'80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6');source=json.loads(source_path.read_text())
    reports={};report_refs={};checks={}
    for seed in p.SEEDS:
        key=str(seed);path=p.WORK/f'seed{seed}_fractional_occupancy/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['all_gates_passed'] and d['fixed_epochs']==168 and d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints']
        assert d['seed']==seed and d['arm']=='fractional_occupancy' and not d['access']['development_metrics']
        p.checked(d['checkpoint']['path'],d['checkpoint']['sha256'])
        old=frozen['controls'][key];p.checked(old['report_path'],old['report_sha256']);control=json.loads(Path(old['report_path']).read_text())
        checks[key]=pairing(d,control);assert all(checks[key].values())
        target=path.parent/'TRAINING_TARGET_RECEIPT.json';t=json.loads(target.read_text());assert t['train_done_exists'] and t['auxiliary_coefficient']==0 and not t['development_access']
        p.checked(p.RECORD/'CACHE_RECEIPT.json',t['cache_receipt_sha256'])
        reports[key]=d;report_refs[key]={'report_path':str(path),'report_sha256':p.sha(path),'checkpoint':d['checkpoint'],'target_receipt_path':str(target),'target_receipt_sha256':p.sha(target)}
    aux=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'images.f32',aux['source_images_sha256']);p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    _,dev=p.check_roles();device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    files=['scripts/evaluate_fractional_occupancy.py','scripts/finalize_fractional_occupancy.py','scripts/evaluate_structured_background_aux.py','scripts/finalize_structured_background_aux.py','tests/test_fractional_occupancy_evaluation.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py']
    p.write_new(pre,{'status':'both_fixed168_before_same_hard_gt_development','created_utc':datetime.now(timezone.utc).isoformat(),'files':{name:p.sha(p.ROOT/name) for name in files},'reports':report_refs,'controls':frozen['controls'],'pairing':checks,'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'evaluation_labels':'unchanged_original_hard128','source_raw_sha256':p.sha(source_path),'cache_index_sha256':p.INDEX_SHA,'hard_labels_sha256':aux['source_labels_sha256'],'soft_training_cache_receipt_sha256':p.sha(p.RECORD/'CACHE_RECEIPT.json'),'bootstrap':{'samples':10000,'seeds':[2026091081,2026091082],'unit':'patient_cluster'},'new_forwards_at_write':0,'expected_new_forwards':360,'fixed_test_seed_if_eligible':2026090711,'test_access':False})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,shuffle=False,num_workers=0)
    results={};comparisons={};gates={};forwards=0;start=time.monotonic()
    for i,seed in enumerate(p.SEEDS):
        key=str(seed);model=AutomaticPromptLocator(center_aware=False).to(device).eval()
        ck=torch.load(reports[key]['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
        rows=[]
        for batch in loader:
            image=batch['image'].to(device);truth=batch['label'].to(device)>.5
            write_json_atomic(p.RECORD/'DEV_ACTIVE_BATCH.json',{'seed':seed,'case_ids':list(batch['case_id']),'completed_forwards_before_batch':forwards,'this_batch_forwards':len(image)})
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
            assert torch.isfinite(logits).all()
            for j,case in enumerate(batch['case_id']):
                pred=logits[j,0]>0;target=truth[j,0];inter=int((pred&target).sum());npred=int(pred.sum());ntarget=int(target.sum());assert ntarget>0
                point=int(logits[j,0].flatten().argmax())
                rows.append({'case_id':case,'subject_id':batch['subject_id'][j],'dice':2*inter/(npred+ntarget),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':ntarget})
            forwards+=len(image)
            write_json_atomic(p.RECORD/'DEV_PROGRESS.json',{'seed':seed,'completed_forwards':forwards,'current_rows':rows,'completed_results':results})
        candidate=summarize(rows,2026091081+i);control=source['results'][f'seed{seed}_control']
        assert all(a['target_voxels']==b['target_voxels'] for a,b in zip(rows,control['per_case'],strict=True))
        c=compare(control,candidate,2026091081+i)
        results[key]={'reused_control':control,'candidate':candidate};comparisons[key]=c;gates[key]=performance_gates(c)
        del model,ck;torch.cuda.empty_cache()
    avg=statistics.fmean(c['scan_delta'] for c in comparisons.values());passed=all(all(g.values()) for g in gates.values()) and avg>=.003
    assert forwards==360
    raw={'schema':'aflow.fractional_occupancy.reused_development','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':comparisons,'gates':gates,'average_scan_delta':avg,'all_gates_passed':passed,'pairing':checks,'new_forwards':forwards,'seconds':time.monotonic()-start,'dev_preaccess_sha256':p.sha(pre),'access':{'reused_development180':True,'new_control_inference':False,'test270':False,'confirmation':False,'fixed_validation108':False,'outer':False,'external':False},'decision':'eligible_only_for_preregistered_fixed_seed11_test_panel' if passed else 'close_exact_occupancy_target_with_original_loss_no_kernel_threshold_seed_search'}
    p.write_new(out,raw);print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        path=p.RECORD/'DEVELOPMENT_FAILURE.json'
        if not path.exists():p.write_new(path,{'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'preaccess_exists':(p.RECORD/'DEV_PREACCESS.json').exists(),'progress_exists':(p.RECORD/'DEV_PROGRESS.json').exists(),'partial_results_must_not_select_models':True,'test_access':False})
        raise
