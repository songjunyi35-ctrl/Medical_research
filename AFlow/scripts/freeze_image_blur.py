"""Hash-bound training admission, fixed historical controls and pre-access design."""
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts import image_blur_protocol as p


def main():
    if (p.RECORD/'TRAIN_PREACCESS.json').exists():raise FileExistsError('already admitted')
    p.source.check_freeze();p.check_roles()
    smoke=json.loads((p.RECORD/'SMOKE_RECEIPT.json').read_text())
    assert smoke['status']=='passed' and all(smoke['gates'].values())
    for name,digest in smoke['files'].items():p.checked(p.ROOT/name,digest)
    controls={}
    for seed in p.SEEDS:
        for arm in p.ARMS:
            path=p.control_path(seed,arm);report=json.loads(path.read_text())
            assert report['status']=='complete' and report['fixed_epochs']==168 and report['all_gates_passed']
            assert report['fit_sha256']==p.FIT_SHA and not report['teacher_checkpoints']
            p.checked(Path(report['checkpoint']['path']),report['checkpoint']['sha256'])
            controls[f'{seed}_{arm}']={'report_path':str(path),'report_sha256':p.sha(path),'checkpoint':report['checkpoint']}
    source_raws={
        'native':p.source.RECORD/'RAW_DEVELOPMENT.json',
        'subject':p.ROOT/'archive/research_records/2026-09-07_subject_balanced_sampling/RAW_DEVELOPMENT.json'}
    expected={'native':'80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6','subject':'96785e393fd1033684014fda091f2de1545b42a0fce6fe226f79a8a10630017d'}
    for key,path in source_raws.items():p.checked(path,expected[key])
    with (p.RECORD/'AUX_CACHE_RECEIPT.json').open('xb') as f:f.write((p.source.RECORD/'AUX_CACHE_RECEIPT.json').read_bytes())
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    files=[p.RECORD/name for name in ['IDEA.md','PREREG.md','SESSION.md','SMOKE_RECEIPT.json']]
    files += [p.ROOT/name for name in ['sammed3d/image_blur.py','scripts/image_blur_protocol.py','scripts/train_image_blur.py','scripts/freeze_image_blur.py','scripts/smoke_image_blur.py','tests/test_image_blur.py','scripts/train_structured_background_aux.py','scripts/train_subject_balanced_sampling.py','sammed3d/subject_balanced_sampling.py','sammed3d/structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','scripts/train_t1ce_wt_gcsam_inner.py']]
    files += [p.ROOT/name for name in ['scripts/run_image_blur_training.py','scripts/build_t1ce_wt_native_volume_artifact.py','sammed3d/t1ce_wt_capg3d.py','scripts/structured_background_protocol.py','scripts/subject_balanced_protocol.py']]
    files += [p.FIT,p.DEV,p.CACHE/'index.json',p.NATIVE,p.ROLES/'AUTHORIZATION.md',p.ROLES/'SPLIT.json']
    files += [p.ROOT/n for n in ['scripts/require_development_mask_verification.py','scripts/verify_resumed_development_masks.py','scripts/evaluate_resumed_with_masks.py','sammed3d/prediction_recording.py','tests/test_prediction_recording.py']]
    parent=p.ROOT/'archive/research_records/2026-09-10_small_rotation'
    if (parent/'RESULT.json').exists():
        files += [parent/n for n in ['RESULT.json','VERIFICATION_RECEIPT.json','DEV_MASK_VERIFICATION.json','DEV_MASK_INDEX.json','DEV_MASK_PREACCESS.json']]
    else:
        assert (parent/'TRAINING_FAILURE.json').exists()
        files += [parent/'TRAINING_FAILURE.json']
    p.write_new(p.RECORD/'IMPLEMENTATION_FREEZE.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(path.relative_to(p.ROOT)):p.sha(path) for path in files},'controls':controls,'source_raws':{k:{'path':str(v),'sha256':expected[k]} for k,v in source_raws.items()}})
    p.write_new(p.RECORD/'TRAIN_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_formal_training','implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'aux_cache_receipt_sha256':p.sha(p.RECORD/'AUX_CACHE_RECEIPT.json'),'smoke_sha256':p.sha(p.RECORD/'SMOKE_RECEIPT.json'),'new_development_access':False,'controls_previously_revealed':True})
    print('two image-blur candidates admitted',flush=True)


if __name__=='__main__':main()
