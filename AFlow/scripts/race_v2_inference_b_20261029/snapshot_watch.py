"""Quiet final-only root-registered108 dependency queue; no network calls."""
import subprocess
import sys
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.snapshot_partner import S,verify_entry,choose_partner,final_handoff,NoNewBest

def main():
    verify_entry();frozen=read(R/'SNAPSHOT_WATCH_FREEZE.json')
    for source,digest in frozen['source_sha256'].items():assert sha(ROOT/source)==digest,source
    exclusive(S/'WATCH_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=1,
        behavior='root-registered full-budget final only; no network access; quiet dependency wait'))
    job_path=STAGE/'SNAPSHOT108_QUEUE/race_v2_final_best108.json'
    while not job_path.exists():
        guard()
        if (STAGE/'FINAL_HANDOFF_NO_NEW_MODEL.json').exists():
            try:final_handoff(ROOT/'archive/research_records/2026-10-29_race_v2_training/HANDOFF.json')
            except NoNewBest:return
            raise RuntimeError('root no-new-model marker disagrees with actual final handoff')
        time.sleep(30)
    job=read(job_path)
    assert job['registered_by']=='root' and job['approval_scope']=='final_completed_training_snapshot108_only'
    assert job['job_id']=='race_v2_final_best108'
    assert sha(job['handoff_path'])==job['handoff_sha256'] and sha(job['training_done_path'])==job['training_done_sha256']
    handoff,_=final_handoff(job['handoff_path']);assert handoff['run_id']==job['run_id']
    if not (S/'PARTNER_FREEZE.json').exists():choose_partner(job['handoff_path'])
    ready=ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/snapshots/race_v2_final_best108/ATOMS_READY.json'
    while not ready.exists():
        guard()
        if (ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/SNAPSHOT108_FAILURE.json').exists():
            raise RuntimeError('A final snapshot108 queue failed; no B fallback network')
        time.sleep(10)
    if not (S/'RESULT.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.snapshot_analyze',
            '--cache',str(TMP/'shared_validation/native'),'--candidate-ready',str(ready),
            '--handoff',job['handoff_path']],check=True)
    write(S/'WATCH_STATUS.json',dict(at=now(),state='complete',result_sha256=sha(S/'RESULT.json'),new_model_forwards=0))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(S/'WATCH_FAILURE.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
