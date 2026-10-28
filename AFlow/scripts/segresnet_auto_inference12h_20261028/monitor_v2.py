"""Read-only, current-resource two-GPU utilization sampling."""
import subprocess
from scripts.segresnet_auto_inference12h_20261028.common import *

def main():
    resource=read(R/'RESOURCE_CURRENT.json');token=resource['token']
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        if read(R/'RESOURCE_CURRENT.json')['token']!=token:return
        result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
        append_log('UTILIZATION_V2.jsonl',dict(at=now(),unix=time.time(),worker=resource['worker'],resource_token=token,gpu=result.stdout.strip(),exit_code=result.returncode))
        time.sleep(20)
if __name__=='__main__':main()
