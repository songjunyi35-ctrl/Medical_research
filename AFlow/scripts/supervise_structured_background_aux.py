"""Continue the fixed two-seed workflow, with live PID checks and fail-closed gates."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts import structured_background_protocol as p


def now():return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--first-control-pid',type=int,required=True)
    ap.add_argument('--first-candidate-pid',type=int,required=True)
    args=ap.parse_args()
    p.check_freeze()
    started=p.RECORD/'CHAIN_STARTED.json'
    if started.exists():raise FileExistsError('chain already started')
    expected={'control':args.first_control_pid,'candidate':args.first_candidate_pid}
    identities={}
    for arm,pid in expected.items():
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'scripts.train_structured_background_aux' in cmd and f'--arm {arm}' in cmd and '--seed 2026090711' in cmd
        identities[arm]={'pid':pid,'cmdline':cmd,'start_ticks':Path(f'/proc/{pid}/stat').read_text().split()[21]}
    p.write_new(started,{'created_utc':now(),'supervisor_pid':os.getpid(),'first_pair':identities,'second_seed':2026090712,'workflow':['wait_first_pair_success','run_second_pair','verify_four_and_evaluate_dev','finalize_saved_counts'],'implementation_freeze_sha256':p.sha(p.RECORD/'IMPLEMENTATION_FREEZE.json'),'supervisor_sha256':p.sha(Path(__file__)),'finalizer_sha256':p.sha(p.ROOT/'scripts/finalize_structured_background_aux.py')})
    progress=p.RECORD/'CHAIN_PROGRESS.json'
    children=[]
    def save(stage,**extra):
        data={'updated_utc':now(),'stage':stage,**extra}
        tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(progress)
    def complete(seed,arm):
        f=p.WORK/f'seed{seed}_{arm}'/'TRAIN_DONE.json'
        if not f.exists():return False
        d=json.loads(f.read_text())
        if d['status']!='complete' or not d['all_gates_passed']:raise RuntimeError(f'failed {seed}/{arm}')
        p.checked(Path(d['checkpoint']['path']),d['checkpoint']['sha256'])
        return True
    def live(identity):
        try:
            stat=Path(f'/proc/{identity["pid"]}/stat').read_text().split()
            return stat[2]!='Z' and stat[21]==identity['start_ticks']
        except FileNotFoundError:return False
    begin=time.monotonic()
    logroot=p.WORK/'chain_logs';logroot.mkdir(exist_ok=False)
    try:
        while True:
            done={arm:complete(2026090711,arm) for arm in p.ARMS}
            if all(done.values()):break
            for arm,is_done in done.items():
                if not is_done and not live(identities[arm]):raise RuntimeError(f'first {arm} exited without successful TRAIN_DONE')
            if time.monotonic()-begin>6*3600:raise TimeoutError('first pair time bound')
            save('first_pair_running',done=done)
            time.sleep(20)
        p.check_freeze()
        for i,arm in enumerate(p.ARMS):
            if (p.WORK/f'seed2026090712_{arm}').exists():raise FileExistsError('second pair already exists; refusing duplicate')
            logfile=(logroot/f'seed2026090712_{arm}.log').open('x')
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.train_structured_background_aux','--seed','2026090712','--arm',arm,'--device',f'cuda:{i}'],cwd=p.ROOT,stdout=logfile,stderr=subprocess.STDOUT)
            logfile.close();children.append((arm,child))
        save('second_pair_running',pids={arm:c.pid for arm,c in children})
        second_start=time.monotonic()
        while any(c.poll() is None for _,c in children):
            for arm,c in children:
                if c.poll() not in (None,0):raise RuntimeError(f'second {arm} exit {c.returncode}')
            if time.monotonic()-second_start>6*3600+120:raise TimeoutError('second pair time bound')
            save('second_pair_running',pids={arm:c.pid for arm,c in children},return_codes={arm:c.poll() for arm,c in children})
            time.sleep(20)
        assert all(c.returncode==0 and complete(2026090712,arm) for arm,c in children)
        for stage,module in [('development_evaluation','scripts.evaluate_structured_background_aux'),('result_archival','scripts.finalize_structured_background_aux')]:
            save(stage)
            with (logroot/(stage+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-m',module],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
        save('completed',result=str(p.RECORD/'RESULT.md'))
        p.write_new(p.RECORD/'CHAIN_DONE.json',{'status':'completed','finished_utc':now(),'result_sha256':p.sha(p.RECORD/'RESULT.json'),'verification_sha256':p.sha(p.RECORD/'VERIFICATION_RECEIPT.json'),'canonical_document_update_pending':True})
        print('all fixed training/evaluation/archive steps completed; canonical handoff/catalog update pending',flush=True)
    except BaseException as e:
        for _,child in children:
            if child.poll() is None:child.terminate()
        save('failed',error=repr(e))
        p.write_new(p.RECORD/'CHAIN_FAILURE.json',{'status':'failed','finished_utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'development_started':(p.RECORD/'DEV_PREACCESS.json').exists(),'raw_development_exists':(p.RECORD/'RAW_DEVELOPMENT.json').exists()})
        raise


if __name__=='__main__':main()
