"""Durable new combination pipeline; never starts a test evaluation."""
import sys,os,json,time,subprocess,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_pretrained_native_common import *
from scripts.finalize_pretrained_native import documents,checks
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
     'OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN','AFLOW_BUILD_PROCESSES':'24'}


def stage(name,cmd):
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':name,'running':True,'worker':1104883,'test_access':False})
    with (R/(name+'.log')).open('a') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
        last_epoch=-1
        while proc.poll() is None:
            progress=json.loads((OUT/'STATUS.json').read_text()) if (OUT/'STATUS.json').exists() else None
            write(R/'HEARTBEAT.json',{'utc':now(),'stage':name,'pid':proc.pid,'alive':True,'progress':progress,'test_access':False})
            if name.startswith('train') and progress and progress['last_complete_epoch']!=last_epoch:
                last_epoch=progress['last_complete_epoch'];write(R/'STATUS.json',{'utc':now(),'status':'training','progress':progress,'test_access':False})
                if last_epoch==1 or last_epoch%25==0:documents('训练中',progress)
            time.sleep(20)
    return proc.returncode


def required(name,cmd):
    code=stage(name,cmd)
    if code:raise RuntimeError(f'{name} exited {code}; see corresponding log')


def pipeline():
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    if (R/'PIPELINE_START.json').exists():raise RuntimeError('already attempted; no automatic duplicate cache/model access')
    assert torch.cuda.is_available() and torch.cuda.device_count()==2
    write(R/'PIPELINE_START.json',{'utc':now(),'pid':os.getpid(),'worker':1104883,'freeze_sha256':sha(R/'FREEZE.json'),'test_authorized':False})
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'initializing','running':True,'worker':1104883,'test_access':False})
    (R/'RESULT.md').write_text('# 公开预训练＋原分辨率组合进度\n\n已授权并启动新流水线，先执行合成与数据门；尚无本候选最终精度。[七问](IDEA.md) · [实时阶段](PIPELINE_STATUS.json) · [心跳](HEARTBEAT.json)。\n')
    documents('准备与执行门检查');checks()
    required('canonical_audit',[PY,'scripts/prepare_pretrained_native.py','audit'])
    required('gpu_smoke',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_pretrained_native.py','smoke'])
    required('build_cache',[PY,'scripts/build_segresnet_pretrained_native_cache.py'])
    required('fixed_references',[PY,'scripts/evaluate_pretrained_native_references.py'])
    required('execution_gate',[PY,'scripts/prepare_pretrained_native.py','gate'])
    for attempt in range(3):
        code=stage('train_attempt'+str(attempt+1),[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_pretrained_native.py','train'])
        status=json.loads((OUT/'STATUS.json').read_text()) if (OUT/'STATUS.json').exists() else {}
        if code==0 and status.get('status')=='completed':break
        if status.get('reason')=='wall_budget':raise RuntimeError('frozen wall budget reached; do not extend')
        if attempt==2 or not (OUT/'latest.pth').exists():raise RuntimeError('training failed; automatic resume limit or no committed epoch')
        write(R/f'ENGINEERING_RESUME_{attempt+1}.json',{'utc':now(),'returncode':code,'status':status,
              'latest_sha256':sha(OUT/'latest.pth'),'same_recipe':True})
    required('finalize_validation',[PY,'scripts/finalize_pretrained_native.py'])
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete_validation_only','running':False,'worker':1104883,'test_access':False})
    write(R/'HEARTBEAT.json',{'utc':now(),'alive':False,'status':'completed','test_access':False})


def launch():
    verify();assert not (R/'LAUNCH.json').exists()
    with (R/'pipeline.log').open('x') as f:
        proc=subprocess.Popen([PY,'scripts/run_pretrained_native.py','pipeline'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,start_new_session=True)
    write(R/'LAUNCH.json',{'utc':now(),'pid':proc.pid,'worker':1104883});print(json.dumps({'launched':True,'pid':proc.pid}))


if __name__=='__main__':
    try:{'launch':launch,'pipeline':pipeline}[sys.argv[1]]()
    except BaseException as e:
        failure={'utc':now(),'status':'failed','running':False,'error':repr(e),'traceback':traceback.format_exc(),
                 'new_test_access':False,'training_preaccess_exists':(R/'TRAIN_PREACCESS.json').exists()}
        write(R/'FAILURE.json',failure);write(R/'PIPELINE_STATUS.json',failure);write(R/'HEARTBEAT.json',failure)
        if (R/'RESULT.md').exists():
            with (R/'RESULT.md').open('a') as f:f.write('\n执行失败：'+repr(e)+'。未完成部分不能称训练完成。\n')
        raise
