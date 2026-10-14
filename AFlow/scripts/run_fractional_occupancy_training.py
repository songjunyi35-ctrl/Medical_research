"""Two fixed candidates in parallel; no automated development/test selection."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
import traceback
from scripts import fractional_occupancy_protocol as p
from scripts.train_t1ce_wt_capg3d import write_json_atomic


def main():
    p.check_freeze();p.write_new(p.RECORD/'TRAINING_STARTED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'runner_sha256':p.sha(__file__)})
    logs=p.WORK/'logs';logs.mkdir(parents=True,exist_ok=False);children=[]
    deadline=datetime(2026,9,10,11,28,19,tzinfo=timezone.utc).timestamp()
    try:
        for gpu,seed in enumerate(p.SEEDS):
            with (logs/f'seed{seed}.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','-m','scripts.train_fractional_occupancy','--seed',str(seed),'--arm','fractional_occupancy','--device',f'cuda:{gpu}'],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((seed,child))
        print('fractional target candidates started '+str({str(s):c.pid for s,c in children}),flush=True)
        while any(c.poll() is None for _,c in children):
            for seed,c in children:
                if c.poll() not in (None,0):raise RuntimeError(f'seed{seed} exit={c.returncode}')
                path=p.WORK/f'seed{seed}_fractional_occupancy/progress.json'
                if path.exists():
                    history=json.loads(path.read_text())['history']
                    if len(history)>=2:
                        per_epoch=(history[-1]['elapsed_seconds']-history[0]['elapsed_seconds'])/(len(history)-1)
                        if time.time()+(168-len(history))*per_epoch+600>deadline:raise TimeoutError('full168 plus archival reserve cannot fit deadline')
            if time.time()>deadline-600:raise TimeoutError('session archival reserve')
            write_json_atomic(p.RECORD/'TRAINING_PROGRESS.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'processes':{str(s):{'pid':c.pid,'returncode':c.poll()} for s,c in children},'new_development_access':False})
            time.sleep(20)
        assert all(c.returncode==0 for _,c in children)
        p.write_new(p.RECORD/'TRAINING_DONE.json',{'status':'both_fixed168_candidates_complete','finished_utc':datetime.now(timezone.utc).isoformat(),'new_development_access':False});print('two fixed168 candidates done',flush=True)
    except BaseException as error:
        for _,c in children:
            if c.poll() is None:c.terminate()
        for _,c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        p.write_new(p.RECORD/'TRAINING_FAILURE.json',{'status':'failed','created_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error),'traceback':traceback.format_exc(),'new_development_access':False});raise


if __name__=='__main__':main()
