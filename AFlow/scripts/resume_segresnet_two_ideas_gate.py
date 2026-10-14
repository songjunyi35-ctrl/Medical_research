"""Resume only the pre-training class-cache gate after documented field-name fix."""
import sys,os,json,time,threading,subprocess,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_two_ideas_common import *
from scripts import run_segresnet_two_ideas as p


def check():
    verify();amend=json.loads((R/'ENGINEERING_AMENDMENT_001.json').read_text())
    assert sha(R/'FREEZE.json')==amend['original_freeze_sha256']
    for name,digest in amend['new_files'].items():assert sha(ROOT/name)==digest
    for name,digest in amend['passed_gate_sha256'].items():assert sha(R/name)==digest


def run():
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check()
    if (R/'RECOVERY_START.json').exists():raise RuntimeError('recovery already started')
    assert not OUT.exists() and not list(R.glob('TRAIN_PREACCESS_*.json'))
    assert (R/'CLASS_CACHE_RAW.jsonl').stat().st_size==0
    assert not (R/'CLASS_CACHE_RECEIPT.json').exists()
    # Preserve the failed pre-access receipt and sparse zero-filled output, never delete historical data.
    for name in ['CLASS_CACHE_PREACCESS.json','CLASS_CACHE_RAW.jsonl']:
        (R/name).rename(R/(name+'.attempt1'))
    CLASS_CACHE.rename(CLASS_CACHE.with_name(CLASS_CACHE.name+'_failed_field_name'))
    write(R/'RECOVERY_START.json',{'utc':now(),'pid':os.getpid(),'worker':1104883,
          'amendment_sha256':sha(R/'ENGINEERING_AMENDMENT_001.json'),'training_started_before_recovery':False,
          'source_labels_read_before_failure':False,'no_historical_files_deleted':True})
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'preparing_training','stage':'corrected_class_cache','worker':1104883,'running':True})
    p.stage([p.PY,'scripts/build_segresnet_two_ideas_classes_v2.py'],'build_classes_v2')
    p.stage([p.PY,'scripts/prepare_segresnet_two_ideas.py','gate'],'execution_gate')
    external={'status':'running'};thread=threading.Thread(target=p.prepare_external,args=(external,),daemon=False);thread.start()
    outcomes={}
    try:
        p.train('semantic_regions');p.stage([p.PY,'scripts/finalize_segresnet_two_ideas.py','semantic_regions'],'finalize_semantic_regions')
        outcomes['semantic_regions']='complete_test270'
    except BaseException as e:
        outcomes['semantic_regions']='failed';write(RECORDS['semantic_regions']/'PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
    while thread.is_alive():
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'waiting_external_cache','outcomes':outcomes,'worker':1104883});thread.join(20)
    if external['status']=='passed':
        try:
            p.train('pretrain');p.train('finetune_pretrained')
            p.stage([p.PY,'scripts/finalize_segresnet_two_ideas.py','finetune_pretrained'],'finalize_finetune_pretrained')
            outcomes['finetune_pretrained']='complete_test270'
        except BaseException as e:
            outcomes['finetune_pretrained']='failed';write(RECORDS['finetune_pretrained']/'PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
    else:
        outcomes['finetune_pretrained']='source_preparation_failed'
        write(RECORDS['finetune_pretrained']/'PIPELINE_FAILURE.json',{'utc':now(),**external,'training_started':False,'test_access':False})
    done=all(x=='complete_test270' for x in outcomes.values())
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete' if done else 'ended_with_failure','outcomes':outcomes,'worker':1104883,'running':False})
    body='2026-09-21两候选流水线已退出：'+str(outcomes)+'。无自动新增组或测试调优。[最终状态](archive/research_records/2026-09-21_segresnet_two_experiments/PIPELINE_STATUS.json)。'
    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:p.update_block(ROOT/filename,'TWO_IDEAS_PIPELINE',body)


def launch():
    check()
    if (R/'RECOVERY_LAUNCH.json').exists():raise RuntimeError('already launched recovery')
    with (R/'recovery_pipeline.log').open('x') as f:
        proc=subprocess.Popen([p.PY,'scripts/resume_segresnet_two_ideas_gate.py','run'],cwd=ROOT,env=p.ENV,stdout=f,
                              stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(R/'RECOVERY_LAUNCH.json',{'utc':now(),'pid':proc.pid,'worker':1104883});print(json.dumps({'launched':True,'pid':proc.pid}))


if __name__=='__main__':
    try:{'launch':launch,'run':run}[sys.argv[1]]()
    except BaseException as e:
        write(R/'RECOVERY_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'failed','error':repr(e),'running':False});raise
