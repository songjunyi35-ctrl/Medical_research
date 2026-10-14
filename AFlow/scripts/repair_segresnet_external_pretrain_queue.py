"""Rebuild external cache with unchanged numerics, then take over the idle GPU queue."""
import sys,os,json,time,fcntl,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_two_ideas_common import *
from scripts import run_segresnet_two_ideas as pipeline


def check():
    verify();a=json.loads((R/'ENGINEERING_AMENDMENT_002.json').read_text())
    assert sha(R/'FREEZE.json')==a['original_freeze_sha256']
    for n,h in a['new_files'].items():assert sha(ROOT/n)==h
    for n,h in a['evidence_sha256'].items():assert sha(R/n)==h


def status(phase,**extra):
    write(R/'EXTERNAL_REPAIR_STATUS.json',{'utc':now(),'phase':phase,'worker':1104883,**extra})


def run():
    repair_lock=(R/'external_repair.lock').open('a');fcntl.flock(repair_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
    if (R/'EXTERNAL_REPAIR_START.json').exists():raise RuntimeError('repair already attempted')
    assert not (OUT/'pretrain').exists() and not (OUT/'finetune_pretrained').exists()
    assert not (R/'PRETRAIN_CACHE_RECEIPT.json').exists()
    old=PRETRAIN_CACHE.with_name(PRETRAIN_CACHE.name+'_failed_external_id')
    assert PRETRAIN_CACHE.exists() and not old.exists()
    PRETRAIN_CACHE.rename(old)
    write(R/'EXTERNAL_REPAIR_START.json',{'utc':now(),'pid':os.getpid(),'worker':1104883,
        'amendment_sha256':sha(R/'ENGINEERING_AMENDMENT_002.json'),'preserved_failed_cache':str(old),
        'external_source_pixels_access_prior_attempt':True,'pretrain_model_access_prior_attempt':False,
        'new_test_access':False,'no_data_or_training_recipe_change':True})
    status('rebuilding_cache')
    pipeline.stage([pipeline.PY,'scripts/build_segresnet_two_ideas_pretrain_v2.py'],'build_pretrain_v2')
    write(R/'EXTERNAL_PREPARATION_STATUS.json',{'utc':now(),'status':'passed','engineering_recovery':2,
          'cache_receipt_sha256':sha(R/'PRETRAIN_CACHE_RECEIPT.json'),'new_test_access':False})
    status('waiting_for_existing_semantic_pipeline_to_exit')
    # The old parent captured its failed preparation result. It will finish B, skip A, and exit.
    # Do not kill B or its supervisor. Acquire the same exclusive lock only after it exits.
    lock=(R/'pipeline.lock').open('a');deadline=time.monotonic()+12*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('timed out waiting for first pipeline; no GPU task started')
            status('waiting_for_existing_semantic_pipeline_to_exit');time.sleep(20)
    check()
    assert not (OUT/'pretrain').exists() and not (OUT/'finetune_pretrained').exists()
    prior=json.loads((R/'PIPELINE_STATUS.json').read_text())
    write(R/'EXTERNAL_QUEUE_HANDOFF.json',{'utc':now(),'prior_pipeline_status':prior,'exclusive_queue_lock_acquired':True,
          'reason':'old supervisor retained failed source-preparation result; corrected cache is now verified',
          'cache_receipt_sha256':sha(R/'PRETRAIN_CACHE_RECEIPT.json')})
    rec=RECORDS['finetune_pretrained']
    if (rec/'PIPELINE_FAILURE.json').exists():(rec/'PIPELINE_FAILURE.json').rename(rec/'PIPELINE_FAILURE_before_id_repair.json')
    status('pretraining');pipeline.train('pretrain')
    status('finetuning');pipeline.train('finetune_pretrained')
    status('finalizing');pipeline.stage([pipeline.PY,'scripts/finalize_segresnet_two_ideas.py','finetune_pretrained'],'finalize_finetune_pretrained')
    outcomes={'semantic_regions':'complete_test270' if (RECORDS['semantic_regions']/'DONE.json').exists() else 'see_first_pipeline_failure',
              'finetune_pretrained':'complete_test270'}
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete' if outcomes['semantic_regions']=='complete_test270' else 'ended_with_failure',
          'outcomes':outcomes,'worker':1104883,'running':False})
    body='2026-09-21两候选流水线已退出：'+str(outcomes)+'。公开数据编号兼容工程修复已归档，无新增第三组或test定向训练。[最终状态](archive/research_records/2026-09-21_segresnet_two_experiments/PIPELINE_STATUS.json)。'
    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:pipeline.update_block(ROOT/filename,'TWO_IDEAS_PIPELINE',body)
    status('complete')


def launch():
    check()
    if (R/'EXTERNAL_REPAIR_LAUNCH.json').exists():raise RuntimeError('already launched')
    with (R/'external_repair.log').open('x') as f:
        p=subprocess.Popen([pipeline.PY,'scripts/repair_segresnet_external_pretrain_queue.py','run'],cwd=ROOT,env=pipeline.ENV,
              stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(R/'EXTERNAL_REPAIR_LAUNCH.json',{'utc':now(),'pid':p.pid,'worker':1104883});print(json.dumps({'launched':True,'pid':p.pid}))


if __name__=='__main__':
    try:{'launch':launch,'run':run}[sys.argv[1]]()
    except BaseException as e:
        failure={'utc':now(),'phase':'failed','error':repr(e),'traceback':traceback.format_exc()}
        write(R/'EXTERNAL_REPAIR_FAILURE.json',failure);write(R/'EXTERNAL_REPAIR_STATUS.json',failure);raise
