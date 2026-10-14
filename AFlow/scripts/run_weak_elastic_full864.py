"""Run only the admitted full864 pair; test needs its separate fixed pre-access."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
import traceback
from scripts import weak_elastic_full864_protocol as p
from scripts.train_t1ce_wt_capg3d import write_json_atomic


def main():
    p.check_freeze();p.write_new(p.RECORD/'TRAINING_STARTED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'runner_sha256':p.sha(__file__)})
    logs=p.WORK/'logs';logs.mkdir(parents=True,exist_ok=False);children=[]
    deadline=datetime(2026,9,10,11,28,19,tzinfo=timezone.utc).timestamp()
    try:
        for gpu,arm in enumerate(p.ARMS):
            with (logs/f'{arm}.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u','-m','scripts.train_weak_elastic_full864','--seed','2026091031','--arm',arm,'--device',f'cuda:{gpu}'],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((arm,child))
        print('full864 pair started '+str({a:c.pid for a,c in children}),flush=True)
        while any(c.poll() is None for _,c in children):
            for arm,c in children:
                if c.poll() not in (None,0):raise RuntimeError(f'{arm} exit {c.returncode}')
                path=p.WORK/f'seed2026091031_{arm}/progress.json'
                if path.exists():
                    h=json.loads(path.read_text())['history']
                    if len(h)>=2:
                        rate=(h[-1]['elapsed_seconds']-h[0]['elapsed_seconds'])/(len(h)-1)
                        if time.time()+(168-len(h))*rate+300>deadline:raise TimeoutError('full168 cannot fit deadline')
            if time.time()>deadline-300:raise TimeoutError('session archival reserve')
            write_json_atomic(p.RECORD/'TRAINING_PROGRESS.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'processes':{a:{'pid':c.pid,'returncode':c.poll()} for a,c in children},'old_development180_now_training':True,'test_started':False})
            time.sleep(20)
        assert all(c.returncode==0 for _,c in children)
        p.write_new(p.RECORD/'TRAINING_DONE.json',{'status':'both_full864_fixed168_complete','finished_utc':datetime.now(timezone.utc).isoformat(),'new_test_access':False})
        print('full864 pair done; test requires separate admission',flush=True)
    except BaseException as error:
        for _,c in children:
            if c.poll() is None:c.terminate()
        for _,c in children:
            try:c.wait(timeout=30)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        p.write_new(p.RECORD/'TRAINING_FAILURE.json',{'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'new_test_access':False,'created_utc':datetime.now(timezone.utc).isoformat()});raise


if __name__=='__main__':main()
