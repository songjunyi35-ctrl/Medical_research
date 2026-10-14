"""A single frozen internal benchmark only after eligible full864 pair completes."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
from scripts import weak_elastic_full864_protocol as p
from scripts import weak_elastic_test270_helpers as h
from scripts.evaluate_weak_elastic import training_pairing
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import REFLECTION_AXES


def main():
    output=p.RECORD/'TEST_PREACCESS.json'
    if output.exists():raise FileExistsError('test already admitted')
    p.check_freeze()
    assert json.loads((p.RECORD/'TRAINING_DONE.json').read_text())['status']=='both_full864_fixed168_complete'
    deadline=datetime(2026,9,10,11,28,19,tzinfo=timezone.utc).timestamp()
    if deadline-time.time()<600:raise RuntimeError('less than10min test/archive reserve')
    mask_bytes=4*270*128**3//8
    if shutil.disk_usage(p.RECORD).free<mask_bytes+1024**3:raise RuntimeError('prediction mask storage reserve')
    reports={};hashes={}
    for arm in p.ARMS:
        path=p.WORK/f'seed2026091031_{arm}/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['all_gates_passed'] and d['fixed_epochs']==168 and d['seed']==2026091031 and d['arm']==arm
        assert d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints'] and not d['access']['development_metrics']
        p.checked(d['checkpoint']['path'],d['checkpoint']['sha256'])
        rolepath=path.parent/'FULL864_ROLE_RECEIPT.json';role=json.loads(rolepath.read_text())
        assert role['development180_consumed_as_training'] and not role['validation_or_test_metrics_accessed']
        if arm=='full_elastic':assert role['deformation']['steps']==168*121 and role['deformation']['max_lipschitz_bound']<=.250001
        reports[arm]=d;hashes[arm]={'train_report':{'path':str(path),'sha256':p.sha(path)},'checkpoint':d['checkpoint'],'role_receipt':{'path':str(rolepath),'sha256':p.sha(rolepath)}}
    pairing=training_pairing(reports['full_elastic'],reports['full_control']);assert all(pairing.values())
    p.checked(h.TEST,h.TEST_SHA);p.checked(h.CACHE/'index.json',h.INDEX_SHA);p.checked(h.CACHE/'images.f32',h.IMAGES_SHA);p.checked(h.CACHE/'labels.u8',h.LABELS_SHA);p.checked(h.LEADER,h.LEADER_SHA)
    test=p.rows(h.TEST);assert len(test)==270 and len({'-'.join(r['case_id'].split('-')[:3]) for r in test})==119
    fit,_=p.check_roles();assert not {r['subject_id'] for r in fit}&{'-'.join(r['case_id'].split('-')[:3]) for r in test}
    old=h.leader_rows(json.loads(h.LEADER.read_text()))
    files=['scripts/freeze_weak_elastic_test270.py','scripts/evaluate_weak_elastic_test270.py','scripts/weak_elastic_test270_helpers.py','scripts/evaluate_weak_elastic.py','scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py','scripts/evaluate_t1ce_wt_plain_locator_test270.py','scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','tests/test_weak_elastic_test270_helpers.py']
    files+=['sammed3d/packed_prediction_masks.py','tests/test_packed_prediction_masks.py','scripts/verify_saved_test270_panel.py','tests/test_saved_test270_panel_verification.py','archive/research_records/2026-09-10_weak_elastic_full864/TEST_OUTPUT_ADDENDUM.md']
    files+=['scripts/archive_verified_test270_panel.py','tests/test_test270_panel_archival_rendering.py','scripts/verify_aflow_brats_leaderboard.py']
    p.write_new(output,{'status':'full864_pair_frozen_before_single_reused_test270','created_utc':datetime.now(timezone.utc).isoformat(),'reports':hashes,'pairing':pairing,'files':{name:p.sha(p.ROOT/name) for name in files},'training_preaccess_sha256':p.sha(p.RECORD/'TRAIN_PREACCESS.json'),'implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'test':{'manifest_path':str(h.TEST),'manifest_sha256':h.TEST_SHA,'cache_path':str(h.CACHE),'cache_index_sha256':h.INDEX_SHA,'cache_images_sha256':h.IMAGES_SHA,'cache_labels_sha256':h.LABELS_SHA,'scans':270,'subjects':119},'old_leader':{'path':str(h.LEADER),'sha256':h.LEADER_SHA,'scan_dice':h.OVERALL_SCAN,'subject_dice':h.OVERALL_SUBJECT,'point_hits':265},'inference':{'batch_size':8,'model_dtype':'bfloat16_autocast','logit_fusion':'ordered_float32_mean_inverse_reflections','reflection_axes':[list(a) for a in REFLECTION_AXES],'threshold':0.,'identity_first_atom_reused':True,'expected_forwards':4320,'postprocessing':False,'sam':False},'bootstrap':{'samples':10000,'seed':h.BOOTSTRAP_SEED,'unit':'patient_cluster','multiple_comparison_adjustment':False},'outputs':str(p.RECORD/'RAW_TEST270.json'),'prediction_masks':{'directory':str(p.RECORD/'test_prediction_masks'),'index':str(p.RECORD/'TEST_MASK_INDEX.json'),'bytes':mask_bytes,'bitorder':'little','flatten_order':'C'},'new_forwards_at_write':0,'access':{'test270_new_forwards':False,'old_confirmation':False,'fixed_validation108':False,'outer':False,'external':False},'source_limitation':'mirror_training_source_official_byte_parity_unproven','historical_test_results_known':True,'evidence':'reused_internal_test_nonblind_not_external; inherited_historical_test_informed_pipeline; no_new_panel_retuning','one_run_limit':True})
    print('single full864 test270 panel frozen',flush=True)


if __name__=='__main__':main()
