"""Run the unchanged frozen evaluator with a transparent mask-only observer."""
import argparse
from datetime import datetime,timezone
import importlib
import json
from pathlib import Path
import shutil
import traceback
from sammed3d.prediction_recording import recording_locator
from sammed3d.packed_prediction_masks import PackedPredictionWriter


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['small_rotation','image_blur'],required=True);args=ap.parse_args()
    evaluator=importlib.import_module('scripts.evaluate_'+args.study);p=evaluator.p;p.check_freeze()
    rec=p.RECORD;prepath=rec/'DEV_MASK_PREACCESS.json'
    if prepath.exists() or (rec/'DEV_PREACCESS.json').exists() or (rec/'RAW_DEVELOPMENT.json').exists():raise FileExistsError('development readout already attempted')
    done=json.loads((rec/'TRAINING_DONE.json').read_text())
    assert done['status']==('all_four_complete' if args.study=='small_rotation' else 'both_complete')
    assert json.loads((rec/'TRAINING_REPLAY_FINAL.json').read_text())['status']=='passed_full168'
    smoke_path=p.ROOT/'archive/research_records/2026-09-10_resumed_eight_hour/PREDICTION_RECORDING_SMOKE.json';smoke=json.loads(smoke_path.read_text())
    assert smoke['status']=='passed' and smoke['exact_output'] and smoke['exact_saved_threshold_mask']
    for n,d in smoke['files'].items():p.checked(p.ROOT/n,d)
    _,dev=p.check_roles();ids=[r['case_id'] for r in dev];keys=[f'{s}_{a}' for s in p.SEEDS for a in p.ARMS]
    expected=180*len(keys);mask_bytes=expected*128**3//8
    assert len(ids)==180 and len(keys)==(4 if args.study=='small_rotation' else 2)
    assert shutil.disk_usage(rec).free>mask_bytes+1024**3
    checkpoints={}
    for key in keys:
        path=p.WORK/f'seed{key}/TRAIN_DONE.json';r=json.loads(path.read_text())
        assert r['all_gates_passed'] and r['fixed_epochs']==168 and r['fit_sha256']==p.FIT_SHA
        p.checked(r['checkpoint']['path'],r['checkpoint']['sha256'])
        checkpoints[key]={'checkpoint':r['checkpoint'],'train_report_path':str(path),'train_report_sha256':p.sha(path)}
    anatomy=p.ROOT/'archive/research_records/2026-09-10_development_error_anatomy/CLASS_CACHE.json';class_cache=json.loads(anatomy.read_text())
    assert class_cache['status']=='passed' and class_cache['case_ids']==ids;p.checked(Path(class_cache['path']),class_cache['sha256'])
    files=['scripts/evaluate_resumed_with_masks.py','scripts/verify_resumed_development_masks.py','sammed3d/prediction_recording.py','tests/test_prediction_recording.py','sammed3d/packed_prediction_masks.py','tests/test_packed_prediction_masks.py','scripts/evaluate_'+args.study+'.py']
    files+=['sammed3d/development_mask_contract.py','tests/test_development_mask_contract.py']
    p.write_new(prepath,{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.ROOT/n) for n in files},'addendum_sha256':p.sha(rec/'DEV_MASK_ADDENDUM.md'),'original_implementation_freeze_sha256':p.sha(rec/'IMPLEMENTATION_FREEZE.json'),'original_development_freeze_sha256':p.sha(rec/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json'),'smoke_sha256':p.sha(smoke_path),'checkpoints':checkpoints,'case_ids':ids,'keys':keys,'expected_forwards':expected,'extra_forwards':0,'mask_bytes':mask_bytes,'class_cache_receipt_path':str(anatomy),'class_cache_receipt_sha256':p.sha(anatomy),'class_cache_sha256':class_cache['sha256'],'original_output_returned_unchanged':True,'new_candidate_development_forwards_at_write':0,'test_access':False})
    writer=PackedPredictionWriter(rec/'dev_prediction_masks',keys,ids);original=evaluator.AutomaticPromptLocator
    recorded,counters=recording_locator(original,writer,ids,keys);evaluator.AutomaticPromptLocator=recorded
    try:
        evaluator.main()
        raw=json.loads((rec/'RAW_DEVELOPMENT.json').read_text())
        assert raw['new_forwards']==counters['forwards']==expected
        assert counters['instances']==len(keys) and all(n==180 for n in counters['rows_by_key'].values())
        assert set(raw['results'])==set(keys)
        index=writer.finalize();index.update(associated_raw_sha256=p.sha(rec/'RAW_DEVELOPMENT.json'),observer_preaccess_sha256=p.sha(prepath),observed_forwards=counters['forwards'])
        p.write_new(rec/'DEV_MASK_INDEX.json',index)
        p.write_new(rec/'DEV_MASK_ARCHIVAL_DONE.json',{'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'raw_sha256':p.sha(rec/'RAW_DEVELOPMENT.json'),'index_sha256':p.sha(rec/'DEV_MASK_INDEX.json'),'preaccess_sha256':p.sha(prepath),'counters':counters,'extra_model_forwards':0,'bytes':sum(x['bytes'] for x in index['files'].values())})
        print('unchanged evaluator complete; actual candidate masks archived',flush=True)
    except BaseException as error:
        p.write_new(rec/'DEV_MASK_ARCHIVAL_FAILURE.json',{'status':'failed','created_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error),'traceback':traceback.format_exc(),'observer_counters':counters,'source_preaccess_exists':(rec/'DEV_PREACCESS.json').exists(),'source_raw_exists':(rec/'RAW_DEVELOPMENT.json').exists(),'no_repeat_for_missing_masks':True,'test_access':False});raise
    finally:
        evaluator.AutomaticPromptLocator=original;writer.close()


if __name__=='__main__':main()
