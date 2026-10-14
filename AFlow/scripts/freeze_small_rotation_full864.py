"""Conditional admission with a complete-trajectory wall-clock budget gate."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
from scripts import small_rotation_full864_protocol as p
from scripts.smoke_small_rotation_full864 import eligible_source


def main():
    if (p.RECORD/'TRAIN_PREACCESS.json').exists():raise FileExistsError('already admitted')
    p.source.check_freeze();p.check_roles();result=eligible_source()
    smoke=json.loads((p.RECORD/'SMOKE_RECEIPT.json').read_text());assert smoke['status']=='passed' and all(smoke['gates'].values())
    assert smoke['selected_arm']==result['selected_arm']
    for name,digest in smoke['files'].items():p.checked(p.ROOT/name,digest)
    times=[]
    for seed in p.source.SEEDS:
        report=json.loads((p.source.WORK/f'seed{seed}_{result["selected_arm"]}/TRAIN_DONE.json').read_text())
        times.append(report['elapsed_seconds'])
    # 10% margin over slowest whole168 source run, plus 20 min for evaluation/archive.
    estimate=max(times)*1.1+1200
    deadline=datetime(2026,9,10,21,39,55,tzinfo=timezone.utc).timestamp()
    if time.time()+estimate>deadline:raise RuntimeError(f'full168 budget admission failed: need {estimate:.1f}s, remaining {deadline-time.time():.1f}s')
    cache=json.loads((p.source.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    p.write_new(p.RECORD/'AUX_CACHE_RECEIPT.json',{'schema':'aflow.full864.zero_auxiliary_placeholder','auxiliary_weight':0,'rc_file':None,'source_images_sha256':cache['source_images_sha256'],'source_labels_sha256':cache['source_labels_sha256'],'source_fit_auxiliary_receipt_sha256':p.sha(p.source.RECORD/'AUX_CACHE_RECEIPT.json'),'no_rc_cache_read_or_created':True})
    files=[p.RECORD/n for n in ['IDEA.md','PREREG.md','FIT864.jsonl','MANIFEST_RECEIPT.json','SMOKE_RECEIPT.json']]
    files += [p.source.RECORD/n for n in ['RESULT.json','VERIFICATION_RECEIPT.json','RAW_DEVELOPMENT.json','DEV_PREACCESS.json','TRAINING_REPLAY_FINAL.json']]
    files += [p.ROOT/n for n in ['scripts/small_rotation_full864_protocol.py','scripts/train_small_rotation_full864.py','scripts/smoke_small_rotation_full864.py','scripts/freeze_small_rotation_full864.py','scripts/run_small_rotation_full864.py','scripts/prepare_small_rotation_full864_manifest.py','tests/test_small_rotation_full864.py','scripts/train_structured_background_aux.py','sammed3d/structured_background_aux.py','sammed3d/small_rotation.py','sammed3d/subject_balanced_sampling.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_gcsam_inner.py','scripts/build_t1ce_wt_native_volume_artifact.py']]
    files += [p.source.RECORD/n for n in ['DEV_MASK_VERIFICATION.json','DEV_MASK_INDEX.json','DEV_MASK_PREACCESS.json']]
    files += [p.ROOT/n for n in ['scripts/require_development_mask_verification.py','scripts/verify_resumed_development_masks.py','sammed3d/prediction_recording.py']]
    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'session_end_utc':'2026-09-10T21:39:55+00:00','compute_deadline_utc':'2026-09-10T21:24:55+00:00','files':{str(path.relative_to(p.ROOT)):p.sha(path) for path in files},'selected_arm':result['selected_arm'],'source_full168_seconds':times,'budget_estimated_seconds_with_margin_and_archive':estimate,'deadline_utc':'2026-09-10T21:39:55+00:00','training_supervisor_deadline_utc':'2026-09-10T21:24:55+00:00'})
    p.write_new(p.RECORD/'TRAIN_PREACCESS.json',{'status':'frozen_before_formal_training','created_utc':datetime.now(timezone.utc).isoformat(),'session_end_utc':'2026-09-10T21:39:55+00:00','compute_deadline_utc':'2026-09-10T21:24:55+00:00','implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'aux_cache_receipt_sha256':p.sha(p.RECORD/'AUX_CACHE_RECEIPT.json'),'smoke_sha256':p.sha(p.RECORD/'SMOKE_RECEIPT.json'),'fit864_sha256':p.FIT_SHA,'seed':2026091071,'selected_sampling_policy':result['selected_arm'],'development180_consumed_as_training':True,'teacher_checkpoints':[],'new_test_access':False})
    print('full864 complete-trajectory pair admitted',flush=True)


if __name__=='__main__':main()
