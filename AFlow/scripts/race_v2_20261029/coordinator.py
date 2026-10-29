"""New finite pipeline: prepare -> 108 diagnosis -> frozen80 -> train -> export.

No historical coordinator is started. At most two checkpoint-only recoveries;
failed source/data/numerical gates never cause patient training to be repeated.
"""
import subprocess
import traceback
import fcntl
import numpy as np
from scripts.race_v2_20261029.common import *
from scripts.race_v2_stage_20261029.governance import guard

def status(phase,**fields):write(R/'PIPELINE_STATUS.json',dict(at=now(),phase=phase,training_test_pixels=False,**fields))

def wait_for(path,phase):
    while not path.exists():
        guard(1154259);deadline();disk()
        if list(R.glob('PREPARE_FAILURE_*.json')): raise RuntimeError('preparation failure requires audit')
        status(phase,prepare=read(R/'PREPARE_HEARTBEAT.json') if (R/'PREPARE_HEARTBEAT.json').exists() else {})
        time.sleep(20)

def call(mode,resume=False):
    environment=env()
    if resume:environment['AFLOW_RESUME']='1'
    result=subprocess.run([str(PY),'-m','scripts.race_v2_20261029.launch',mode],cwd=ROOT,env=environment)
    return result.returncode

def audit_payload(split):
    current=read(CACHE/split/'index.json')
    historical=read(ROOT/f'archive/research_records/2026-10-28_race_seg/DATA_{split}.json')
    assert current['case_ids']==historical['case_ids'] and current['canvas']==historical['canvas']
    assert current['digests']==historical['digests']
    for cid in current['case_ids']:
        for key in ('origin','extent','canvas_offset','reflection','reference_enabled'):
            assert current['geometry'][cid][key]==historical['geometry'][cid][key],(cid,key)
    per_case={'images.f32':int(np.prod(CANVAS))*4,'labels.u8':int(np.prod(CANVAS)),
              'classes.u8':int(np.prod(CANVAS)),'canonical_images.f32':128**3*4,'canonical_labels.u8':128**3}
    sizes={}
    for name,digest in current['digests'].items():
        path=CACHE/split/name;sizes[name]=path.stat().st_size
        assert sizes[name]==current['count']*per_case[name]
        assert sha(path)==digest
    write(R/f'PAYLOAD_AUDIT_{split}.json',dict(at=now(),status='passed',actual_payload_sha_checked=True,
          sizes=sizes,digests=current['digests'],geometry_reflection_exact=True,historical_case_order_exact=True,
          worker='1154259',test_pixels=False))

def freeze():
    if (R/'EXECUTION_FREEZE.json').exists():
        verify_sources()
        return
    smoke=read(R/'SMOKE_DDP.json');assert smoke['status']=='passed' and smoke['bf16_deployment_exact']
    assert read(R/'CPU_CHECKS.json')['status']=='passed'
    checks=read(R/'FREEZE_CHECKS.json');assert checks['status']=='passed'
    for path,digest in checks['source_sha256'].items():assert sha(ROOT/path)==digest
    assert read(R/'DIAGNOSIS108.json')['status']=='complete'
    for path,digest in smoke['numerical_source_sha256'].items():assert sha(ROOT/path)==digest
    assert numerical_functions()==smoke['numerical_function_ast_sha256']
    assert smoke['smoke_source_freeze_sha256']==sha(R/'SMOKE_SOURCE_FREEZE_V3.json')
    allocation=guard(1154259,training=True);deadline(required=True)
    speed=max(rank['seconds_per_update'] for rank in smoke['ranks'])
    # Full optimizer updates are measured; validation60s and 50% CPU/I/O slack
    # are conservative relative to the prior native DDP run. Keep45min extra.
    per_epoch=(122*speed+60)*1.5
    available=allocation['deadline_unix']-time.time()-2700
    epochs=min(80,int(available/per_epoch))
    if epochs<1:raise RuntimeError('not enough remaining new-stage budget')
    write(R/'BUDGET_DECISION.json',dict(at=now(),status='passed',new_epochs=epochs,new_updates=epochs*122,
          seconds_per_update=speed,per_epoch_estimated_seconds=per_epoch,
          remaining_training_seconds=available,estimated_training_seconds=epochs*per_epoch,
          training_deadline_unix=allocation['deadline_unix'],effective_batch=8,microbatch_per_gpu=2,
          world_size=2,updates_per_epoch=122,examples_per_epoch=972,epochs_chosen_before_patient_updates=True))
    files=sorted((ROOT/'scripts/race_v2_20261029').glob('*.py'))
    dependencies=['sammed3d/race_seg.py','sammed3d/t1ce_wt_capg3d.py',
                  'scripts/race_seg_20261028/data.py','scripts/race_seg_20261028/prepare.py',
                  'scripts/segresnet_pretrained_native_common.py','scripts/train_t1ce_wt_capg3d.py',
                  'scripts/train_t1ce_wt_plain_long_schedule.py','scripts/nnunet_resenc_20261013/common.py',
                  'scripts/nnunet_local_full_20261022/native_builder_v2.py',
                  'scripts/race_v2_inference_a_20261029/statistics.py',
                  'scripts/race_v2_stage_20261029/governance.py']
    paths=[str(p.relative_to(ROOT)) for p in files]+dependencies
    write(R/'EXECUTION_FREEZE.json',dict(at=now(),source_sha256={p:sha(ROOT/p) for p in paths},
          recipe_sha256=sha(R/'RECIPE.json'),idea_sha256=sha(R/'IDEA.md'),budget_sha256=sha(R/'BUDGET_DECISION.json'),
          source_checkpoint_sha256=SOURCE_SHA,numerical_function_ast_sha256=numerical_functions(),
          smoke_gate_sha256=sha(R/'SMOKE_DDP.json'),smoke_source_freeze_sha256=sha(R/'SMOKE_SOURCE_FREEZE_V3.json'),
          bf16_deployment_exact=smoke['bf16_deployment_exact'],numerical_source_sha256=smoke['numerical_source_sha256'],
          patient_training_updates_yet=0))

def live_training():
    path=R/'TRAIN_LAUNCH.json'
    if not path.exists():return False
    pid=read(path)['pid'];proc=Path(f'/proc/{pid}')
    try:
        return proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z' and b'torch.distributed.run' in proc.joinpath('cmdline').read_bytes()
    except (FileNotFoundError,ProcessLookupError):return False

def recovery_count():
    numbers=sorted(int(p.stem.rsplit('_',1)[1]) for p in R.glob('ENGINEERING_RECOVERY_*.json'))
    assert numbers==list(range(1,len(numbers)+1))
    return len(numbers)

def record_recovery(code):
    count=recovery_count()
    if count>=2 or not (TMP/'run/latest.pth').exists():
        raise RuntimeError('training exit not recoverable within finite completed-checkpoint budget')
    verify_sources();guard(1154259,training=True)
    with (R/f'ENGINEERING_RECOVERY_{count+1}.json').open('x') as handle:
        json.dump(dict(at=now(),exitcode=code,cumulative_engineering_recoveries=count+1,
              last_complete=read(R/'STATUS.json') if (R/'STATUS.json').exists() else None,
              optimizer_checkpoint_sha256=sha(TMP/'run/latest.pth'),
              recipe_unchanged=True,source_freeze_unchanged=True),handle)

def main():
    R.mkdir(parents=True,exist_ok=True)
    lock=(R/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('waiting_for_validation_payload_and_final_synthetic_gate')
    wait_for(R/'DATA_validation.json','waiting_for_validation_payload')
    audit_payload('validation')
    wait_for(R/'SMOKE_DDP.json','waiting_for_synthetic_gate')
    assert read(R/'SMOKE_DDP.json').get('bf16_deployment_exact') is True
    if not (R/'DIAGNOSIS108.json').exists():
        status('diagnosing108_normal_base_mismatch')
        if call('diagnose'):raise RuntimeError('diagnosis failed; no training started')
    wait_for(R/'DATA_READY.json','waiting_for_train972_payload')
    audit_payload('train')
    freeze();status('all_gates_passed_starting_new_optimizer_finetuning',budget=read(R/'BUDGET_DECISION.json'))
    while live_training() and not (R/'DONE.json').exists():
        guard(1154259,training=True)
        status('attached_to_existing_training',training=read(R/'STATUS.json') if (R/'STATUS.json').exists() else None)
        time.sleep(20)
    if not (R/'DONE.json').exists():
        resume=(TMP/'run/latest.pth').exists()
        if resume:record_recovery('coordinator_attach_after_training_exit')
        elif (R/'TRAIN_PREACCESS.json').exists():
            raise RuntimeError('prior patient training access but no complete latest checkpoint; no automatic repeat')
        while True:
            code=call('train',resume=resume)
            if code==0 and (R/'DONE.json').exists():break
            record_recovery(code);resume=True
    status('training_complete_exporting_fixed_weight')
    subprocess.run([str(PY),'-m','scripts.race_v2_20261029.finalize'],cwd=ROOT,env=env(),check=True)
    status('complete',result=read(R/'RESULT.json'))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        traceback.print_exc();status('stopped_needs_audit',error=repr(error))
        write(R/f'COORDINATOR_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error)));raise
