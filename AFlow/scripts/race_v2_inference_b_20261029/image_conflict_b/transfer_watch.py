"""One fixed-formula transfer after the existing final snapshot/partner queue."""
import subprocess
import sys
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.image_conflict_b.transfer import I,S,verify_transfer

def main():
    verify_transfer();exclusive(S/'WATCH_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,
        physical_gpu=1,stage='wait_final_snapshot_then_one_fixed_formula',new_model_forwards=0))
    job_path=STAGE/'SNAPSHOT108_QUEUE/race_v2_final_best108.json';legacy_folder=R/'final_snapshot'
    while not job_path.exists():
        guard()
        if (STAGE/'FINAL_HANDOFF_NO_NEW_MODEL.json').exists():
            handoff_path=ROOT/'archive/research_records/2026-10-29_race_v2_training/HANDOFF.json'
            handoff=read(handoff_path);done=read(handoff_path.with_name('DONE.json'))
            assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']
            assert handoff['actual_new_epochs']==0
            write(S/'NO_NEW_MODEL.json',dict(at=now(),state='skipped_no_new_selected_model',
                handoff_sha256=sha(handoff_path),root_no_new_marker_sha256=sha(STAGE/'FINAL_HANDOFF_NO_NEW_MODEL.json'),
                new_model_forwards=0,new_parameter_search=False));return
        time.sleep(30)
    job=read(job_path);assert job['registered_by']=='root' and job['approval_scope']=='final_completed_training_snapshot108_only'
    while True:
        guard()
        complete=legacy_folder/'WATCH_STATUS.json'
        cov_complete=R/'innovation_b/final_snapshot/WATCH_STATUS.json'
        if complete.exists() and read(complete).get('state')=='complete' and (legacy_folder/'RESULT.json').exists() and cov_complete.exists() and read(cov_complete).get('state')=='complete':break
        time.sleep(10)
    previous=read(legacy_folder/'RESULT.json');handoff=read(job['handoff_path'])
    assert previous['frozen_partner']['candidate_checkpoint_sha256']==handoff['checkpoint_sha256']
    candidate_ready=ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/snapshots/race_v2_final_best108/ATOMS_READY.json'
    assert candidate_ready.exists()
    if not (S/'RESULT.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.image_conflict_b.transfer',
            '--handoff',job['handoff_path'],'--registration',str(job_path),'--candidate-ready',str(candidate_ready)],check=True)
    result=read(S/'RESULT.json');assert result['state']=='complete'
    assert result['partner_model_id']==previous['global_partner']
    write(S/'WATCH_STATUS.json',dict(at=now(),state='complete',result_sha256=sha(S/'RESULT.json'),
        configuration_count=1,new_model_forwards=0,new_parameter_search=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(S/f'WATCH_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_model_forwards=0));raise
