"""Bounded one-shot evaluation; never restarts neural access after failure."""
import fcntl
import signal
import subprocess
from scripts.race_inference_test270_20261029.common import *


def main():
    lock=(R/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();verify('EXECUTION_FREEZE_V2.json');processes=[]
    while not (R/'CACHE_READY.json').exists():
        guard()
        if (R/'CACHE_FAILURE_V3.json').exists():raise RuntimeError('current cache builder failed')
        write(R/'PIPELINE_STATUS.json',dict(at=now(),state='preparing_cache',training=False,
              patient_model_forward=False,cache_heartbeat=read(R/'CACHE_HEARTBEAT.json') if (R/'CACHE_HEARTBEAT.json').exists() else None))
        time.sleep(10)
    assert read(R/'BATCH_DECISION.json')['status']=='passed'
    assert read(R/'CACHE_READY.json')['all_four_payloads_exact']
    verify();manifests()
    assert sha(SOURCE_RECIPE)==read(R/'SELECTED_RECIPE.json')['source_recipe_sha256']
    for field in ('aggregation','decoder','decoder_kind','postprocessing','point_rule'):
        assert read(SOURCE_RECIPE)[field]==read(R/'SELECTED_RECIPE.json')[field]
    exclusive(R/'MODEL_PREACCESS_V2.json',dict(at=now(),checkpoint_epoch=164,checkpoint_sha256=CHECKPOINT_SHA,
          recipe_sha256=sha(R/'SELECTED_RECIPE.json'),execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),
          test_scans=270,test_subjects=119,views=list(range(8)),maximum_case_forwards=2160,
          original_identity_reused=True,current_test_parameter_search=False,
          rank_slots={str(rank):list(range(rank,270,2)) for rank in (0,1)},no_automatic_neural_retry=True))
    try:
        for rank in (0,1):
            e=env();e['CUDA_VISIBLE_DEVICES']=str(rank)
            with (TMP/f'evaluate_v2_rank{rank}.log').open('ab') as log:
                processes.append(subprocess.Popen([str(PY),'-m','scripts.race_inference_test270_20261029.evaluate_v2','--rank',str(rank)],
                     cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        write(R/'INFERENCE_PIDS.json',dict(at=now(),worker=WORKER,rank_pids={str(i):p.pid for i,p in enumerate(processes)}))
        while any(p.poll() is None for p in processes):
            guard()
            write(R/'PIPELINE_STATUS.json',dict(at=now(),state='fixed_inference_test270',training=False,
                  test_access=True,rank_heartbeat={str(rank):read(R/f'HEARTBEAT_{rank}.json') for rank in (0,1) if (R/f'HEARTBEAT_{rank}.json').exists()}))
            time.sleep(10)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('inference failed; complete raw outputs may be decoded without another forward')
        subprocess.run([str(PY),'-m','scripts.race_inference_test270_20261029.publish_v2'],cwd=ROOT,env=env(),check=True)
        for name in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py'):
            subprocess.run([str(PY),str(ROOT/'scripts'/name)],cwd=ROOT,env=env(),check=True)
    except BaseException:
        for process in processes:
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(R/'COORDINATOR_FAILURE_V2.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),no_automatic_neural_retry=True));raise
