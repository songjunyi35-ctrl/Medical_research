"""Execute the predeclared sampler workflow after the preceding study is archived."""
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
import time
import traceback
from scripts import subject_balanced_protocol as p


def main():
    if (p.RECORD/'CHAIN_STARTED.json').exists():raise FileExistsError('sampler workflow already started')
    closure=json.loads((p.RECORD/'SOURCE_CANONICAL_CLOSED.json').read_text())
    assert closure['status']=='source_results_archived_and_canonical_docs_updated'
    p.checked(p.SOURCE_RECORD/'RESULT.json',closure['source_result_sha256'])
    p.source.check_freeze();p.check_roles()
    design=json.loads((p.RECORD/'DESIGN_FREEZE.json').read_text())
    p.checked(p.RECORD/'IDEA.md',design['idea_sha256']);p.checked(p.RECORD/'PREREG.md',design['prereg_sha256'])
    p.write_new(p.RECORD/'CHAIN_STARTED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'supervisor_pid':os.getpid(),'source_closure_sha256':p.sha(p.RECORD/'SOURCE_CANONICAL_CLOSED.json'),'design_sha256':p.sha(p.RECORD/'DESIGN_FREEZE.json'),'conditional_on_source_performance':False,'runner_sha256':p.sha(__file__)})
    p.WORK.mkdir(parents=True,exist_ok=True);logroot=p.WORK/'chain_logs';logroot.mkdir(exist_ok=False)
    children=[]
    def stage(name,announce=True,**details):
        value={'updated_utc':datetime.now(timezone.utc).isoformat(),'stage':name,**details}
        path=p.RECORD/'CHAIN_PROGRESS.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)
        if announce:print(name,flush=True)
    def run(name,module):
        stage(name)
        with (logroot/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-u','-m',module],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    try:
        run('synthetic_smoke','scripts.smoke_subject_balanced_sampling')
        run('training_admission','scripts.freeze_subject_balanced_sampling')
        for i,seed in enumerate(p.SEEDS):
            with (logroot/f'seed{seed}.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','scripts.train_subject_balanced_sampling','--seed',str(seed),'--arm','subject_balanced','--device',f'cuda:{i}'],cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((seed,process))
        stage('two_sampler_candidates_running',pids={str(s):c.pid for s,c in children})
        start=time.monotonic()
        while any(c.poll() is None for _,c in children):
            for seed,c in children:
                if c.poll() not in (None,0):raise RuntimeError(f'candidate {seed} exit {c.returncode}')
            if time.monotonic()-start>6*3600+120:raise TimeoutError('formal candidate time bound')
            stage('two_sampler_candidates_running',announce=False,pids={str(s):c.pid for s,c in children},return_codes={str(s):c.poll() for s,c in children})
            time.sleep(20)
        assert all(c.returncode==0 for _,c in children)
        run('development_evaluation','scripts.evaluate_subject_balanced_sampling')
        run('result_archival','scripts.finalize_subject_balanced_sampling')
        stage('completed')
        p.write_new(p.RECORD/'CHAIN_DONE.json',{'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':p.sha(p.RECORD/'RESULT.json'),'verification_sha256':p.sha(p.RECORD/'VERIFICATION_RECEIPT.json'),'canonical_document_update_pending':True})
    except BaseException as e:
        for _,child in children:
            if child.poll() is None:child.terminate()
        stage('failed',error=repr(e))
        p.write_new(p.RECORD/'CHAIN_FAILURE.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'dev_started':(p.RECORD/'DEV_PREACCESS.json').exists(),'raw_exists':(p.RECORD/'RAW_DEVELOPMENT.json').exists()})
        raise


if __name__=='__main__':main()
