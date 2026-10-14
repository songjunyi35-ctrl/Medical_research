"""Two GPU waves, four fixed runs; never starts development or test evaluation."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
import traceback
from scripts import small_rotation_protocol as p
from scripts.train_t1ce_wt_capg3d import write_json_atomic


def main():
    p.check_freeze()
    p.write_new(p.RECORD/'TRAINING_STARTED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'runner_sha256':p.sha(__file__),'deadline_utc':'2026-09-10T21:24:55+00:00'})
    logs=p.WORK/'logs';logs.mkdir(parents=True,exist_ok=False)
    children=[]
    deadline=datetime(2026,9,10,21,24,55,tzinfo=timezone.utc).timestamp()
    try:
        for seed in p.SEEDS:
            children=[]
            for device,arm in enumerate(p.ARMS):
                with (logs/f'{seed}_{arm}.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','scripts.train_small_rotation','--seed',str(seed),'--arm',arm,'--device',f'cuda:{device}'],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
                children.append((arm,child))
            print(f'seed {seed} started '+str({a:c.pid for a,c in children}),flush=True)
            while any(c.poll() is None for _,c in children):
                for arm,child in children:
                    if child.poll() not in (None,0):raise RuntimeError(f'{seed}/{arm} exit={child.returncode}')
                    progress=p.WORK/f'seed{seed}_{arm}/progress.json'
                    if progress.exists():
                        h=json.loads(progress.read_text())['history']
                        if len(h)>=2:
                            per_epoch=(h[-1]['elapsed_seconds']-h[0]['elapsed_seconds'])/(len(h)-1)
                            remaining=(168-len(h))*per_epoch
                            # Budget checks use timing only, never losses/development metrics.
                            if time.time()+remaining+300>deadline:raise TimeoutError('full168 cannot fit session deadline')
                write_json_atomic(p.RECORD/'TRAINING_PROGRESS.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'processes':{a:{'pid':c.pid,'returncode':c.poll()} for a,c in children},'development_started':False})
                if time.time()>deadline-300:raise TimeoutError('session archival reserve')
                time.sleep(20)
            assert all(c.returncode==0 for _,c in children)
            print(f'seed {seed} both fixed168 complete',flush=True)
        p.write_new(p.RECORD/'TRAINING_DONE.json',{'status':'all_four_complete','finished_utc':datetime.now(timezone.utc).isoformat(),'development_accessed':False})
        print('all four training runs complete; development requires separate admission',flush=True)
    except BaseException as error:
        for _,child in children:
            if child.poll() is None:child.terminate()
        for _,child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        p.write_new(p.RECORD/'TRAINING_FAILURE.json',{'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'development_accessed':False,'created_utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__':main()
