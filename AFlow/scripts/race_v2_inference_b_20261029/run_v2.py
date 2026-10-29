"""Process-separated queue: release CUDA and GPU flock before decode stage."""
import subprocess
import sys
from scripts.race_v2_inference_b_20261029.common import *

def main():
    verify()
    freeze=read(R/'QUEUE_V2_FREEZE.json')
    for path,digest in freeze['source_sha256'].items():assert sha(ROOT/path)==digest,path
    cache=TMP/'shared_validation/native'
    while not (cache.parent/'CACHE_READY.json').exists():
        guard();time.sleep(10)
    if not (OUT/'ATOMS_READY.json').exists():
        assert not (R/'MODEL_PREACCESS.json').exists(),'incomplete neural access requires root investigation'
        subprocess.run([sys.executable,'-m','scripts.race_v2_inference_b_20261029.forward','--cache',str(cache)],check=True)
    assert read(OUT/'ATOMS_READY.json')['case_forwards']==864
    while not (A/'ATOMS_READY.json').exists():
        guard();time.sleep(10)
    subprocess.run([sys.executable,'-m','scripts.race_v2_inference_b_20261029.analyze',
                    '--cache',str(cache),'--race-ready',str(A/'ATOMS_READY.json')],check=True)
    write(R/'PIPELINE_STATUS.json',dict(at=now(),state='complete_validation',test_access=False,
          result_sha256=sha(R/'RESULT.json'),raw_sha256=sha(R/'RAW_VALIDATION.json')))

if __name__=='__main__':main()
