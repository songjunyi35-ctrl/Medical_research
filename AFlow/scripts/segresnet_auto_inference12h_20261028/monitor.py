"""Read-only utilization audit, independent of GPU job hashing and CPU analysis."""
import subprocess
from scripts.segresnet_auto_inference12h_20261028.common import *


def main():
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
        append_log('UTILIZATION_SAMPLES.jsonl',dict(at=now(),unix=time.time(),gpu=result.stdout.strip(),
              exit_code=result.returncode,stage=read(R/'PIPELINE_STATUS.json').get('state')))
        time.sleep(20)


if __name__=='__main__':main()
