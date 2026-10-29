"""One assigned-worker queue holding both GPU locks for the full torchrun."""
import argparse
import subprocess
from scripts.race_v2_20261029.common import *
from scripts.race_v2_stage_20261029.governance import gpu_queue_lock,guard

def main(mode):
    TMP.mkdir(parents=True,exist_ok=True)
    guard(1154259,training=mode=='train')
    environment=env();environment['CUDA_VISIBLE_DEVICES']='0,1'
    command=[str(PY),'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
             '-m','scripts.race_v2_20261029.runner',mode]
    with gpu_queue_lock(1154259,[0,1],'training'), (TMP/f'{mode}.log').open('a') as handle:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT)
        write(R/f'{mode.upper()}_LAUNCH.json',dict(at=now(),pid=process.pid,launcher_pid=os.getpid(),worker='1154259',
              gpus=[0,1],command=command,lock_paths=[str(TMP/f'locks/worker1154259_gpu{gpu}.lock') for gpu in (0,1)]))
        code=process.wait()
    write(R/f'{mode.upper()}_PROCESS_EXIT.json',dict(at=now(),returncode=code))
    raise SystemExit(code)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','diagnose','train'));args=parser.parse_args();main(args.mode)
