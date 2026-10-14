"""Cache preparation overlaps patient candidate; each model access is once-only."""
import sys,subprocess,time,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.followup_test270_common import *
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}


def spawn(label,script,args=(),gpu='0'):
    handle=(S/(label+'.log')).open('a')
    p=subprocess.Popen([PY,'scripts/'+script,*args],cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':gpu},stdout=handle,stderr=subprocess.STDOUT)
    handle.close();return p


def wait(p,label):
    while p.poll() is None:
        write(S/'HEARTBEAT.json',{'utc':now(),'stage':label,'pid':p.pid,'alive':True});time.sleep(10)
    if p.returncode:raise RuntimeError(label+f' exit={p.returncode}; inspect log, no automatic repeat')


def main():
    lock=(S/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    write_new(S/'PIPELINE_START.json',{'utc':now(),'worker':1104883,'pid':os.getpid(),'freeze_sha256':sha(S/'FREEZE.json')})
    write(S/'PIPELINE_STATUS.json',{'utc':now(),'status':'synthetic_gpu_checks','running':True})
    wait(spawn('smoke','prepare_followup_test270.py',['smoke']),'smoke')
    wait(spawn('canonical_audit','prepare_followup_test270.py',['audit']),'canonical_audit')
    build=spawn('build_native_cache','build_followup_test270_cache.py')
    patient=spawn('evaluate_patient','evaluate_followup_test270.py',['patient'],gpu='1')
    write(S/'PIPELINE_STATUS.json',{'utc':now(),'status':'native_cache_build_and_patient_evaluation','running':True,'build_pid':build.pid,'patient_pid':patient.pid})
    outcomes={}
    try:
        wait(patient,'patient_evaluation');wait(spawn('publish_patient','publish_followup_test270.py',['patient']),'publish_patient');outcomes['patient']='complete_test270'
    except BaseException as e:
        outcomes['patient']='failed';write(S/'PATIENT_STAGE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
    write(S/'PIPELINE_STATUS.json',{'utc':now(),'status':'waiting_native_cache','running':True,'outcomes':outcomes})
    wait(build,'native_cache')
    wait(spawn('evaluate_native','evaluate_followup_test270.py',['native']),'native_evaluation')
    wait(spawn('publish_native','publish_followup_test270.py',['native']),'publish_native');outcomes['native']='complete_test270'
    write(S/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete' if outcomes['patient']=='complete_test270' else 'partial_failure',
          'running':False,'outcomes':outcomes,'test_tuning_used':False,'new_outer_confirmation_access':False})
    write(S/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','outcomes':outcomes})


def launch():
    verify();assert not (S/'LAUNCH.json').exists()
    with (S/'pipeline.log').open('x') as f:
        p=subprocess.Popen([PY,'scripts/run_followup_test270.py','run'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    write(S/'LAUNCH.json',{'utc':now(),'pid':p.pid,'worker':1104883});print(json.dumps({'launched':True,'pid':p.pid}))


if __name__=='__main__':
    try:{'launch':launch,'run':main}[sys.argv[1]]()
    except BaseException as e:
        write(S/'FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
        write(S/'PIPELINE_STATUS.json',{'utc':now(),'status':'failed','running':False,'error':repr(e)});raise
