"""New-stage dependency queue; exact atoms are never regenerated automatically."""
import argparse
from scripts.race_v2_inference_b_20261029.common import *

def main(cache):
    from scripts.race_v2_inference_b_20261029.forward import main as forward
    from scripts.race_v2_inference_b_20261029.analyze import main as analyze
    OUT.mkdir(parents=True,exist_ok=True);verify()
    exclusive(R/'QUEUE_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=1))
    ready=Path(cache).parent/'CACHE_READY.json'
    while not ready.exists():
        guard()
        if (ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/CACHE_FAILURE.json').exists():
            raise RuntimeError('A shared validation builder failed; no independent duplicate rebuild')
        write(R/'HEARTBEAT.json',dict(at=now(),stage='waiting_shared108_cache',patient_forwards=0,
            actual_dependency=str(ready),no_artificial_gpu_load=True))
        time.sleep(10)
    forward(cache)
    while not (A/'ATOMS_READY.json').exists():
        guard()
        if (ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/QUEUE_FAILURE.json').exists():
            raise RuntimeError('A RACE atoms queue failed; completed B atoms retained')
        write(R/'HEARTBEAT.json',dict(at=now(),stage='waiting_RACE_atoms',w32_forwards=864,
            actual_dependency=str(A/'ATOMS_READY.json'),no_artificial_gpu_load=True))
        time.sleep(10)
    analyze(cache,str(A/'ATOMS_READY.json'))
    write(R/'PIPELINE_STATUS.json',dict(at=now(),state='complete_validation',test_access=False,
        result_sha256=sha(R/'RESULT.json'),raw_sha256=sha(R/'RAW_VALIDATION.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default=str(TMP/'shared_validation/native'));a=p.parse_args()
    try:main(a.cache)
    except BaseException as error:
        import traceback
        write(R/f'QUEUE_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),
            no_automatic_neural_retry=True,completed_atoms_retained=True));raise
