"""Hardware-only two-GPU profile; independent candidates have isolated CUDA state."""
import concurrent.futures
import subprocess
from scripts.race_inference_test270_20261029.common import *


def rank_profile(rank):
    e=env();e['CUDA_VISIBLE_DEVICES']=str(rank)
    for batch in (2,4,8):
        with (TMP/f'profile_rank{rank}_batch{batch}.log').open('ab') as log:
            subprocess.run([str(PY),'-m','scripts.race_inference_test270_20261029.profile','--rank',str(rank),'--batch',str(batch)],
                cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)


def main():
    TMP.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(rank_profile,(0,1)))
    safe={}
    for batch in (2,4,8):
        results=[read(R/f'PROFILE_rank{rank}_batch{batch}.json') for rank in (0,1)]
        if all(x['status']=='passed' for x in results):safe[batch]=min(x['scans_per_second'] for x in results)
    assert safe,'no common safe hardware batch'
    fastest=max(safe.values());chosen=min(b for b,speed in safe.items() if speed>=.97*fastest)
    write(R/'BATCH_DECISION.json',dict(at=now(),status='passed',batch=chosen,tail=135%chosen or chosen,
          safe_scans_per_second=safe,criterion='minimum batch within97percent of best shared synthetic throughput',
          memory_ceiling_gib=38,selection_from_patient_scores=False))


if __name__=='__main__':main()
