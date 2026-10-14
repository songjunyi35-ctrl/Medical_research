import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.patient_equal_common import *
from scripts.finalize_patient_equal import documents,checks
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}


def stage(name,command):
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':name,'running':True,'worker':1105022,'test_access':False})
    with (R/(name+'.log')).open('a') as f:
        proc=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            progress=json.loads((OUT/'STATUS.json').read_text()) if (OUT/'STATUS.json').exists() else None
            write(R/'HEARTBEAT.json',{'utc':now(),'alive':True,'stage':name,'pid':proc.pid,'progress':progress,'test_access':False});time.sleep(20)
    return proc.returncode


def require(name,cmd):
    if stage(name,cmd):raise RuntimeError(name+' failed; see stage log')


def pipeline():
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    assert not (R/'PIPELINE_START.json').exists()
    write(R/'PIPELINE_START.json',{'utc':now(),'pid':os.getpid(),'worker':1105022,'freeze_sha256':sha(R/'FREEZE.json')})
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'preparing','running':True,'worker':1105022,'test_access':False})
    (R/'RESULT.md').write_text('# 患者等权损失进度\n\n已启动执行检查，正式200轮结果尚未产生。[七问](IDEA.md) · [阶段](PIPELINE_STATUS.json) · [心跳](HEARTBEAT.json)。\n')
    documents('执行检查中');checks()
    require('data_audit',[PY,'scripts/prepare_patient_equal.py','audit'])
    require('gpu_smoke',[PY,'scripts/train_patient_equal.py','smoke'])
    documents('训练中')
    for attempt in range(3):
        code=stage('train_attempt'+str(attempt+1),[PY,'scripts/train_patient_equal.py','train'])
        done=(OUT/'STATUS.json').exists() and json.loads((OUT/'STATUS.json').read_text())['status']=='completed'
        if code==0 and done:break
        if attempt==2 or not (OUT/'latest.pth').exists():raise RuntimeError('training failure or resume limit')
        write(R/f'RESUME_{attempt+1}.json',{'utc':now(),'returncode':code,'latest_sha256':sha(OUT/'latest.pth'),'same_recipe':True})
    require('finalize_validation',[PY,'scripts/finalize_patient_equal.py'])
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete_validation_only','running':False,'worker':1105022,'test_access':False})
    write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','test_access':False})


def launch():
    verify();assert not (R/'LAUNCH.json').exists()
    with (R/'pipeline.log').open('x') as f:
        p=subprocess.Popen([PY,'scripts/run_patient_equal.py','pipeline'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    write(R/'LAUNCH.json',{'utc':now(),'pid':p.pid,'worker':1105022});print(json.dumps({'launched':True,'pid':p.pid}))


if __name__=='__main__':
    try:{'launch':launch,'pipeline':pipeline}[sys.argv[1]]()
    except BaseException as e:
        failure={'utc':now(),'status':'failed','running':False,'error':repr(e),'traceback':traceback.format_exc(),'test_access':False}
        write(R/'FAILURE.json',failure);write(R/'PIPELINE_STATUS.json',failure);write(R/'HEARTBEAT.json',failure);raise
