"""One durable worker pipeline: build, audit, synthetic smoke, then single training."""
import sys,os,subprocess,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_plain972_no_n4_common import ROOT,R,sha,write,now,verify
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}

def main():
    verify()
    if (R/'PIPELINE_START.json').exists():raise RuntimeError('pipeline already attempted; no automatic duplicate')
    write(R/'PIPELINE_START.json',{'utc':now(),'pid':os.getpid(),'worker':'1100631','no_test_access':True,'freeze_sha256':sha(R/'FREEZE.json')})
    steps=[('building_cache',[PY,'scripts/build_segresnet_plain972_no_n4_cache.py']),('synthetic_gpu_smoke',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_plain972_no_n4.py','smoke'])]
    for stage,cmd in steps:
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':stage,'pid':os.getpid(),'test_access':False})
        with (R/(stage+'.log')).open('x') as f:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    smoke=json.loads((R/'GPU_SMOKE.json').read_text());assert smoke['status']=='passed'
    write(R/'EXECUTION_GATE.json',{'utc':now(),'status':'passed','files':{str(p.relative_to(ROOT)):sha(p) for p in [R/'FREEZE.json',R/'DATA_AUDIT.json',R/'GPU_SMOKE.json',R/'UNIT_TESTS.json']}})
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'training','pid':os.getpid(),'test_access':False})
    subprocess.run([PY,'scripts/watch_segresnet_plain972_no_n4.py'],cwd=ROOT,env=ENV,check=True)
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'completed','test_access':False})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(R/'PIPELINE_FAILURE.json',{'utc':now(),'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'test_access':False})
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'failed','error':repr(e),'test_access':False});raise
