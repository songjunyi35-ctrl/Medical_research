"""Fixed seed11 old-hard versus occupancy target, one four-output test270 panel."""
from datetime import datetime,timezone
import json
import time
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.packed_prediction_masks import PackedPredictionWriter
from scripts import fractional_occupancy_protocol as p
from scripts import weak_elastic_test270_helpers as h
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import flip8_logits


@torch.inference_mode()
def main():
    p.check_freeze();out=p.RECORD/'RAW_TEST270.json';started=p.RECORD/'TEST_STARTED.json';pre=p.RECORD/'TEST_PREACCESS.json'
    if out.exists() or started.exists():raise FileExistsError('single test already started')
    receipt=json.loads(pre.read_text());assert receipt['status']=='two_development_seeds_passed_fixed_seed11_before_single_test270' and receipt['one_run_limit'] and receipt['fixed_test_seed']==2026090711
    assert not (p.ROOT/'archive/research_records/2026-09-10_weak_elastic_full864/TEST_PREACCESS.json').exists()
    for name,digest in receipt['files'].items():p.checked(p.ROOT/name,digest)
    for path,key in [(h.TEST,'manifest_sha256'),(h.CACHE/'index.json','cache_index_sha256'),(h.CACHE/'images.f32','cache_images_sha256'),(h.CACHE/'labels.u8','cache_labels_sha256')]:p.checked(path,receipt['test'][key])
    p.checked(h.LEADER,h.LEADER_SHA);old=h.leader_rows(json.loads(h.LEADER.read_text()))
    arms=('control','candidate')
    for arm in arms:
        for item in receipt['reports'][arm].values():p.checked(item['path'],item['sha256'])
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats(device)
    p.write_new(started,{'created_utc':datetime.now(timezone.utc).isoformat(),'preaccess_sha256':p.sha(pre),'new_forwards':0,'fixed_seed':2026090711})
    ids=[r['case_id'] for r in p.rows(h.TEST)]
    loader=DataLoader(TestBrainCropDataset(h.CACHE,ids),batch_size=8,shuffle=False,num_workers=0)
    results={};forwards=0;start=time.monotonic();partial={};writer=None
    try:
        writer=PackedPredictionWriter(p.RECORD/'test_prediction_masks',[f'{arm}_{mode}' for arm in arms for mode in ('identity','flip8')],ids)
        for arm in arms:
            model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(receipt['reports'][arm]['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
            collected={'identity':[],'flip8':[]};partial={'arm':arm,'rows':collected}
            for batch in loader:
                image=batch['image'].to(device);labels=batch['label'].to(device)
                identity,flip8=flip8_logits(model,image);forwards+=8*len(image)
                assert torch.isfinite(identity).all() and torch.isfinite(flip8).all()
                for mode,values in [('identity',identity),('flip8',flip8)]:
                    collected[mode].extend(h.rows_from_logits(values,labels,batch));writer.append(f'{arm}_{mode}',(values[:,0]>0).cpu().numpy(),batch['case_id'])
            for mode in ('identity','flip8'):results[f'{arm}_{mode}']=h.summarize(collected[mode])
            del model,ck,identity,flip8,image,labels;torch.cuda.empty_cache()
        assert forwards==4320
        contrasts={};pairs={}
        for arm in arms:
            name=f'{arm}_flip8_vs_identity';contrasts[name]=h.compare(results[f'{arm}_identity'],results[f'{arm}_flip8']);pairs[name]=[f'{arm}_identity',f'{arm}_flip8']
            for mode in ('identity','flip8'):
                name=f'{arm}_{mode}_vs_old_overall';contrasts[name]=h.compare(old,results[f'{arm}_{mode}']);pairs[name]=['old_overall',f'{arm}_{mode}']
        for mode in ('identity','flip8'):
            name=f'candidate_{mode}_vs_matched_control';contrasts[name]=h.compare(results[f'control_{mode}'],results[f'candidate_{mode}']);pairs[name]=[f'control_{mode}',f'candidate_{mode}']
        roles={'candidate_identity':'candidate_identity','candidate_flip8':'candidate_flip8','control_flip8':'control_flip8'}
        gates=h.gates(results,contrasts,roles=roles);overall=all(v for k,v in gates.items() if k!='candidate_identity_1x_scan_lead')
        mask_index=writer.finalize();p.write_new(p.RECORD/'TEST_MASK_INDEX.json',mask_index)
        raw={'schema':'aflow.fractional_occupancy.fixed_seed11.test270','status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':contrasts,'comparison_pairs':pairs,'gates':gates,'gate_role_outputs':roles,'candidate_overall_scientific_gate':overall,'test_preaccess_sha256':p.sha(pre),'old_leader_sha256':h.LEADER_SHA,'new_forwards':forwards,'seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),'prediction_masks':{'index_path':str(p.RECORD/'TEST_MASK_INDEX.json'),'index_sha256':p.sha(p.RECORD/'TEST_MASK_INDEX.json'),'bytes':sum(v['bytes'] for v in mask_index['files'].values()),'mask_outputs':list(mask_index['files'])},'access':{'fit684':True,'reused_development180_for_prior_selection':True,'former_development180_as_training':False,'reused_test270':True,'old_confirmation':False,'fixed_validation108':False,'outer':False,'external':False},'fixed_seed':2026090711,'test_tuned':False,'historical_test_results_known':True,'test_tuning_scope':'No parameters selected from this new panel; inherited baseline pipeline and historical test results were already known. Seed11 fixed before candidate development scores.','further_test_tuning_or_repeat_allowed':False,'all_four_outputs_must_be_leaderboarded':True}
        p.write_new(out,raw);print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2),flush=True)
    except BaseException as error:
        p.write_new(p.RECORD/'TEST_FAILURE.json',{'status':'failed','created_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error),'completed_results':results,'partial_current_model':partial,'completed_forward_counter':forwards,'partial_forward_exception_count_may_be_lower_bound':True,'prediction_mask_cases_written':writer.counts if writer else {},'test_preaccess_sha256':p.sha(pre),'must_not_retune_or_repeat':True});raise
    finally:
        if writer:writer.close()


if __name__=='__main__':main()
