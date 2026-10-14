"""Only the two fixed subject-blur seeds, concurrently on separate GPUs."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
import traceback
from scripts import image_blur_protocol as p


def main():
    p.check_freeze()
    now=datetime.now(timezone.utc)
    assert now<datetime(2026,9,10,19,0,tzinfo=timezone.utc),'too late to admit complete paired168'
    deadline=datetime(2026,9,10,21,24,55,tzinfo=timezone.utc).timestamp()
    p.write_new(p.RECORD/'TRAINING_STARTED.json',{'created_utc':now.isoformat(),'pid':os.getpid(),'deadline_timestamp':deadline,'runner_sha256':p.sha(__file__)})
    logs=p.WORK/'logs';logs.mkdir(parents=True,exist_ok=False);children=[]
    try:
        for gpu,seed in enumerate(p.SEEDS):
            with (logs/f'{seed}_subject_blur.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','-m','scripts.train_image_blur','--seed',str(seed),'--arm','subject_blur','--device',f'cuda:{gpu}'],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((seed,child))
        print('fixed blur seeds started '+str({s:c.pid for s,c in children}),flush=True)
        while any(c.poll() is None for _,c in children):
            for seed,child in children:
                if child.poll() not in (None,0):raise RuntimeError(f'{seed} exit {child.returncode}')
                file=p.WORK/f'seed{seed}_subject_blur/progress.json'
                if file.exists():
                    h=json.loads(file.read_text())['history']
                    if len(h)>=2:
                        seconds=(h[-1]['elapsed_seconds']-h[0]['elapsed_seconds'])/(len(h)-1)
                        if time.time()+(168-len(h))*seconds+300>deadline:raise TimeoutError('complete168 and evaluation no longer fit')
            if time.time()>deadline-300:raise TimeoutError('archival reserve')
            time.sleep(20)
        p.write_new(p.RECORD/'TRAINING_DONE.json',{'status':'both_complete','finished_utc':datetime.now(timezone.utc).isoformat(),'development_accessed':False})
    except BaseException as error:
        for _,c in children:
            if c.poll() is None:c.terminate()
        for _,c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        p.write_new(p.RECORD/'TRAINING_FAILURE.json',{'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'development_accessed':False});raise


if __name__=='__main__':main()
