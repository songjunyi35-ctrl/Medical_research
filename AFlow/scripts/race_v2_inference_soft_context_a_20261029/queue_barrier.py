"""Serialize the immutable normalization transfer after local-dependency output."""
import argparse,fcntl,os,time
import torch
from scripts.race_v2_inference_soft_context_a_20261029.common import *
from scripts.followup_test270_common import numeric_mode

LOCAL=ROOT/'archive/research_records/2026-10-29_v2_inference_innovation_a/final_snapshot'
NORM=ROOT/'archive/research_records/2026-10-29_race_v2_inference_normalization_a'
TRAIN=ROOT/'archive/research_records/2026-10-29_race_v2_training'
FREEZE=R/'SOURCE_QUEUE_BARRIER_FREEZE.json'

def freeze():
    assert not FREEZE.exists();old=BASE/'FINAL270_NORMALIZATION_ENTRY_FREEZE_V2.json'
    sources=dict(read(old)['source_sha256'])
    for p in [Path(__file__),ROOT/'scripts/race_v2_inference_soft_context_a_20261029/common.py']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    write(FREEZE,dict(at=now(),source_sha256=sources,original_numeric_entry_freeze_path=str(old),
        original_numeric_entry_freeze_sha256=sha(old),predecessor_result_path=str(LOCAL/'RESULT.json'),
        original_norm_sources_preserved=True,same_process_delegate=True,best0_direct_skip=True,new_patient_access=False))

def main():
    guard();f=read(FREEZE)
    for p,digest in f['source_sha256'].items():assert sha(ROOT/p)==digest,p
    assert sha(f['original_numeric_entry_freeze_path'])==f['original_numeric_entry_freeze_sha256']
    numeric_mode()
    write(R/'QUEUE_BARRIER_RECEIPT_norm_transfer.json',dict(at=now(),pid=os.getpid(),same_process_delegate=True,
        predecessor=str(LOCAL/'RESULT.json'),source_queue_barrier_freeze_sha256=sha(FREEZE),
        backend_mode=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
        new_model_patient_forwards_yet=0))
    while True:
        guard();done=TRAIN/'DONE.json'
        if done.exists() and read(done)['state']=='complete':
            status=read(done)
            if status['best_new_epoch']==0 and (STAGE/'FINAL_HANDOFF_NO_NEW_MODEL.json').exists():break
            if status['best_new_epoch']>0 and (LOCAL/'RESULT.json').exists():
                assert read(LOCAL/'RESULT.json')['state']=='complete'
                # RESULT may precede process exit; wait until its GPU lock closes.
                lock=LOCK.open('a');started=time.monotonic()
                while True:
                    guard();assert time.monotonic()-started<600,'predecessor result exists but GPU lock not released within600seconds'
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:time.sleep(1)
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close();break
        time.sleep(10)
    from scripts.race_v2_inference_normalization_a_20261029 import numeric_entry
    write(R/'QUEUE_BARRIER_PASSED_norm_transfer.json',dict(at=now(),pid=os.getpid(),
        predecessor_result_sha256=sha(LOCAL/'RESULT.json') if (LOCAL/'RESULT.json').exists() else None,
        zero_epoch_direct_skip=read(TRAIN/'DONE.json')['best_new_epoch']==0,actual_new_patient_forwards=0))
    numeric_entry.main('transfer',True,False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:main()
        except BaseException as error:
            import traceback
            write(R/f'QUEUE_BARRIER_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
