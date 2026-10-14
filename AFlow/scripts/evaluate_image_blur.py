"""One unified exploratory development readout, two fixed168 checkpoints."""
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
from scripts import image_blur_protocol as p


def performance_gates(native_delta, own_delta, subject_arm):
    gates={'subject_gain':native_delta['subject_delta']>=.003,
           'scan_positive':native_delta['scan_delta']>0,
           'point_safety':native_delta['point_delta']>=-1,
           'subject_ci_safety':native_delta['subject_95ci'][0]>=-.003,
           'scan_cluster_ci_safety':native_delta['scan_cluster_95ci'][0]>=-.003}
    if subject_arm:
        gates.update(own_subject_positive=own_delta['subject_delta']>0,
                     own_scan_positive=own_delta['scan_delta']>0,
                     own_point_nonnegative=own_delta['point_delta']>=0)
    return gates


def select_arm(results, comparisons, gates):
    arm_gates={}
    for arm in p.ARMS:
        average=statistics.fmean(comparisons[f'{seed}_{arm}']['versus_native']['scan_delta'] for seed in p.SEEDS)
        arm_gates[arm]={'all_seed_gates':all(all(gates[f'{seed}_{arm}'].values()) for seed in p.SEEDS),'average_scan_delta':average,'average_scan_gain_gate':average>=.003}
        arm_gates[arm]['eligible']=arm_gates[arm]['all_seed_gates'] and arm_gates[arm]['average_scan_gain_gate']
    eligible=[arm for arm in p.ARMS if arm_gates[arm]['eligible']]
    # max() preserves p.ARMS order on exact tie: native first.
    chosen=max(eligible,key=lambda arm:statistics.fmean(results[f'{seed}_{arm}']['subject_dice'] for seed in p.SEEDS)) if eligible else None
    return arm_gates,chosen


def training_pairing(candidate, control):
    a,b=candidate['history'],control['history']
    return {'same_initial':candidate['initial_state_sha256']==control['initial_state_sha256'],
            'same_weights':candidate['sampling_weight_sha256']==control['sampling_weight_sha256'],
            'same168':len(a)==len(b)==168,
            'same121steps':all(x['steps']==y['steps']==121 for x,y in zip(a,b,strict=True)),
            'same_sampler':all(x['sampler_sha256']==y['sampler_sha256'] for x,y in zip(a,b,strict=True)),
            'same_original_augmentation_rng':all(x['augmentation_rng_sha256']==y['augmentation_rng_sha256'] for x,y in zip(a,b,strict=True)),
            'same_first_batch_labels':all(x['first_batch']['labels_sha256']==y['first_batch']['labels_sha256'] for x,y in zip(a,b,strict=True)),
            'export_exact':candidate['export_max_abs_error']==0}


@torch.inference_mode()
def main():
    p.check_freeze()
    development_freeze=json.loads((p.RECORD/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json').read_text())
    for name,digest in development_freeze['files'].items():p.checked(p.ROOT/name,digest)
    out=p.RECORD/'RAW_DEVELOPMENT.json';pre=p.RECORD/'DEV_PREACCESS.json'
    if out.exists() or pre.exists():raise FileExistsError('readout already started')
    assert json.loads((p.RECORD/'TRAINING_DONE.json').read_text())['status']=='both_complete'
    frozen=json.loads((p.RECORD/'IMPLEMENTATION_FREEZE.json').read_text())
    sources={}
    for key,value in frozen['source_raws'].items():
        p.checked(Path(value['path']),value['sha256']);sources[key]=json.loads(Path(value['path']).read_text())
    reports={};receipts={};pairing={};blur={}
    for seed in p.SEEDS:
        for arm in p.ARMS:
            key=f'{seed}_{arm}';path=p.WORK/f'seed{key}/TRAIN_DONE.json';d=json.loads(path.read_text())
            assert d['status']=='complete' and d['all_gates_passed'] and d['fixed_epochs']==168 and d['seed']==seed and d['arm']==arm
            assert d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints'] and not d['access']['development_metrics']
            p.checked(Path(d['checkpoint']['path']),d['checkpoint']['sha256'])
            old=frozen['controls'][key];p.checked(Path(old['report_path']),old['report_sha256'])
            control=json.loads(Path(old['report_path']).read_text())
            pairing[key]=training_pairing(d,control);assert all(pairing[key].values())
            ep=path.parent/'BLUR_FINAL.json';e=json.loads(ep.read_text())
            assert e['steps']==168*121 and e['samples']==168*968 and 0<e['selected']<e['samples']
            assert .5<=e['min_observed_sigma']<=e['max_observed_sigma']<=1.000001 and e['kernel_size']==7 and e['padding']=='reflect'
            assert e['probability']==.2 and e['sigma_range']==[.5,1.]
            reports[key]=d;blur[key]=e
            receipts[key]={'train_report':{'path':str(path),'sha256':p.sha(path)},'checkpoint':d['checkpoint'],'blur_report':{'path':str(ep),'sha256':p.sha(ep)}}
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    _,dev=p.check_roles()
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    code=['scripts/evaluate_image_blur.py','scripts/evaluate_structured_background_aux.py','sammed3d/t1ce_wt_capg3d.py','scripts/train_t1ce_wt_capg3d.py','tests/test_image_blur_evaluation.py','scripts/finalize_image_blur.py','scripts/finalize_structured_background_aux.py','scripts/audit_image_blur_training.py','scripts/audit_subject_sampler_replay.py']
    p.write_new(pre,{'status':'both_fixed_before_unified_development','created_utc':datetime.now(timezone.utc).isoformat(),'checkpoints_reports':receipts,'source_raws':frozen['source_raws'],'dev_sha256':p.DEV_SHA,'cache_index_sha256':p.INDEX_SHA,'pairing':pairing,'blur':blur,'files':{name:p.sha(p.ROOT/name) for name in code},'bootstrap':{'samples':10000,'seed_by_training_seed':{'2026090711':2026091061,'2026090712':2026091062},'unit':'patient_cluster','selection_correction':False},'new_model_forwards_at_write':0,'new_test_access':False,'both_before_selection':True})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,shuffle=False,num_workers=0)
    results={};comparisons={};gates={};forwards=0;start=time.monotonic()
    for i,seed in enumerate(p.SEEDS):
        native=sources['native']['results'][f'seed{seed}_control']
        subject=sources['subject']['results'][str(seed)]['candidate']
        for arm in p.ARMS:
            key=f'{seed}_{arm}';model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(reports[key]['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
            rows=[]
            for batch in loader:
                image=batch['image'].to(device);truth=batch['label'].to(device)>.5
                write_json_atomic(p.RECORD/'DEV_ACTIVE_BATCH.json',{'key':key,'case_ids':list(batch['case_id']),'completed_forwards_before_batch':forwards,'this_batch_forwards':len(image),'created_utc':datetime.now(timezone.utc).isoformat()})
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
                assert torch.isfinite(logits).all()
                for j,case in enumerate(batch['case_id']):
                    pred=logits[j,0]>0;target=truth[j,0];inter=int((pred&target).sum());npred=int(pred.sum());ntarget=int(target.sum());assert ntarget>0
                    point=int(logits[j,0].flatten().argmax())
                    rows.append({'case_id':case,'subject_id':batch['subject_id'][j],'dice':2*inter/(npred+ntarget),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':ntarget})
                forwards+=len(image)
                write_json_atomic(p.RECORD/'DEV_PROGRESS.json',{'key':key,'completed_forwards':forwards,'current_model_rows':rows,'completed_model_results':results,'updated_utc':datetime.now(timezone.utc).isoformat()})
            summary=summarize(rows,2026091061+i)
            assert all(a['case_id']==b['case_id'] and a['target_voxels']==b['target_voxels'] for a,b in zip(rows,native['per_case'],strict=True))
            cn=compare(native,summary,2026091061+i);co=compare(subject if arm=='subject_blur' else native,summary,2026091061+i)
            results[key]=summary;comparisons[key]={'versus_native':cn,'versus_own_sampling_control':co}
            gates[key]=performance_gates(cn,co,arm=='subject_blur')
            del model,ck;torch.cuda.empty_cache()
    assert forwards==360
    arm_gates,chosen=select_arm(results,comparisons,gates)
    raw={'schema':'aflow.image_blur.reused_development','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':comparisons,'gates':gates,'arm_gates':arm_gates,'selected_arm':chosen,'pairing':pairing,'new_forwards':forwards,'seconds':time.monotonic()-start,'dev_preaccess_sha256':p.sha(pre),'access':{'fit':True,'reused_development180':True,'test270':False,'fixed_validation108':False,'outer':False,'confirmation':False,'external':False},'decision':'eligible_for_preregistered_fixed_seed11_test_panel' if chosen else 'close_fixed_image_blur_recipe_no_sigma_probability_seed_epoch_search'}
    p.write_new(out,raw)
    print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        failure=p.RECORD/'DEVELOPMENT_FAILURE.json'
        if not failure.exists():p.write_new(failure,{'status':'failed','created_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error),'traceback':traceback.format_exc(),'preaccess_exists':(p.RECORD/'DEV_PREACCESS.json').exists(),'active_batch_exists':(p.RECORD/'DEV_ACTIVE_BATCH.json').exists(),'progress_exists':(p.RECORD/'DEV_PROGRESS.json').exists(),'partial_results_must_not_select_models':True,'test_access':False})
        raise
