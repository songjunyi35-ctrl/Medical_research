"""Single seed11 panel admission only after both soft-target development seeds pass."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
from scripts import fractional_occupancy_protocol as p
from scripts import weak_elastic_test270_helpers as h
from scripts.evaluate_fractional_occupancy import pairing
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import REFLECTION_AXES


def main():
    output=p.RECORD/'TEST_PREACCESS.json'
    if output.exists():raise FileExistsError('single test already admitted')
    p.check_freeze()
    result=json.loads((p.RECORD/'RESULT.json').read_text());verification=json.loads((p.RECORD/'VERIFICATION_RECEIPT.json').read_text())
    assert verification['status']=='passed';p.checked(p.RECORD/'RESULT.json',verification['result_sha256'])
    assert result['status']=='completed_reused_development_supported' and result['all_gates_passed'] and result['fixed_test_seed_if_eligible']==2026090711
    assert all(all(g.values()) for g in result['gates'].values())
    assert not (p.ROOT/'archive/research_records/2026-09-10_weak_elastic_full864/TEST_PREACCESS.json').exists()
    if datetime(2026,9,10,11,28,19,tzinfo=timezone.utc).timestamp()-time.time()<600:raise RuntimeError('test/archive time reserve')
    if shutil.disk_usage(p.RECORD).free<4*270*128**3//8+(1<<30):raise RuntimeError('mask/archive disk reserve')
    paths={'control':p.source.WORK/'seed2026090711_control/TRAIN_DONE.json','candidate':p.WORK/'seed2026090711_fractional_occupancy/TRAIN_DONE.json'}
    reports={};bodies={}
    for arm,path in paths.items():
        d=json.loads(path.read_text());assert d['all_gates_passed'] and d['fixed_epochs']==168 and d['seed']==2026090711 and d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints']
        p.checked(d['checkpoint']['path'],d['checkpoint']['sha256']);bodies[arm]=d
        reports[arm]={'train_report':{'path':str(path),'sha256':p.sha(path)},'checkpoint':d['checkpoint']}
    checks=pairing(bodies['candidate'],bodies['control']);assert all(checks.values())
    p.checked(h.TEST,h.TEST_SHA);p.checked(h.CACHE/'index.json',h.INDEX_SHA);p.checked(h.CACHE/'images.f32',h.IMAGES_SHA);p.checked(h.CACHE/'labels.u8',h.LABELS_SHA);p.checked(h.LEADER,h.LEADER_SHA)
    ids=p.rows(h.TEST);assert len(ids)==270 and len({'-'.join(r['case_id'].split('-')[:3]) for r in ids})==119
    fit,dev=p.check_roles();assert not {r['subject_id'] for r in fit+dev}&{'-'.join(r['case_id'].split('-')[:3]) for r in ids}
    h.leader_rows(json.loads(h.LEADER.read_text()))
    files=['scripts/freeze_fractional_occupancy_test270.py','scripts/evaluate_fractional_occupancy_test270.py','scripts/weak_elastic_test270_helpers.py','scripts/evaluate_fractional_occupancy.py','scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py','scripts/evaluate_t1ce_wt_plain_locator_test270.py','scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','sammed3d/packed_prediction_masks.py','scripts/verify_saved_test270_panel.py','scripts/archive_verified_test270_panel.py','scripts/verify_aflow_brats_leaderboard.py','tests/test_weak_elastic_test270_helpers.py','tests/test_packed_prediction_masks.py','tests/test_saved_test270_panel_verification.py','tests/test_test270_panel_archival_rendering.py','archive/research_records/2026-09-10_fractional_occupancy/TEST_PLAN.md','archive/research_records/2026-09-10_fractional_occupancy/TEST_OUTPUT_ADDENDUM.md']
    p.write_new(output,{'status':'two_development_seeds_passed_fixed_seed11_before_single_test270','created_utc':datetime.now(timezone.utc).isoformat(),'files':{name:p.sha(p.ROOT/name) for name in files},'reports':reports,'pairing':checks,'source_development_result_sha256':p.sha(p.RECORD/'RESULT.json'),'source_development_verification_sha256':p.sha(p.RECORD/'VERIFICATION_RECEIPT.json'),'fixed_test_seed':2026090711,'test_seed_selected_from_scores':False,'fit_scans':684,'fit_subjects':313,'former_development180_consumed_as_training':False,'test':{'manifest_path':str(h.TEST),'manifest_sha256':h.TEST_SHA,'cache_path':str(h.CACHE),'cache_index_sha256':h.INDEX_SHA,'cache_images_sha256':h.IMAGES_SHA,'cache_labels_sha256':h.LABELS_SHA,'scans':270,'subjects':119},'old_leader':{'path':str(h.LEADER),'sha256':h.LEADER_SHA,'scan_dice':h.OVERALL_SCAN,'subject_dice':h.OVERALL_SUBJECT,'point_hits':265},'inference':{'batch_size':8,'model_dtype':'bfloat16_autocast','logit_fusion':'ordered_float32_mean_inverse_reflections','reflection_axes':[list(a) for a in REFLECTION_AXES],'threshold':0.,'identity_first_atom_reused':True,'expected_forwards':4320,'postprocessing':False,'sam':False},'bootstrap':{'samples':10000,'seed':h.BOOTSTRAP_SEED,'unit':'patient_cluster','multiple_comparison_adjustment':False},'outputs':str(p.RECORD/'RAW_TEST270.json'),'prediction_masks':{'directory':str(p.RECORD/'test_prediction_masks'),'index':str(p.RECORD/'TEST_MASK_INDEX.json'),'bytes':4*270*128**3//8,'bitorder':'little','flatten_order':'C'},'new_forwards_at_write':0,'new_test_access':False,'historical_test_results_known':True,'evidence':'reused_internal_test_nonblind_not_external; inherited_historical_test_informed_pipeline; no_new_panel_retuning','one_run_limit':True})
    print('one fixed seed11 test270 panel admitted',flush=True)


if __name__=='__main__':main()
