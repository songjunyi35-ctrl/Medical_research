"""One durable worker pipeline: native cache -> baseline reference row -> memory probe/smoke -> supervised training."""
import sys,os,subprocess,json,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_native_res_common import ROOT,R,sha,write,now,verify
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}


def stage(name,cmd):
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':name,'pid':os.getpid(),'test_access':False})
    with (R/(name+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)


def cache_ready():
    """Skip the 30-minute rebuild when the audited caches from a previous attempt are still intact."""
    if not (R/'DATA_AUDIT.json').exists():return False
    d=json.loads((R/'DATA_AUDIT.json').read_text())
    try:return d['status']=='passed' and all(sha(Path(c['path'])/'index.json')==c['index_sha256'] for c in d['caches'])
    except OSError:return False


def main():
    verify()
    attempts=sorted(R.glob('PIPELINE_START_attempt*.json'))
    if (R/'PIPELINE_START.json').exists():
        if not (R/'PIPELINE_FAILURE.json').exists():raise RuntimeError('pipeline already running or completed; no automatic duplicate')
        n=len(attempts)+1   # archive the failed attempt's receipts under an attempt suffix, keep everything
        for name in ['PIPELINE_START.json','PIPELINE_FAILURE.json','PIPELINE_STATUS.json','GPU_SMOKE.json','BATCH_DECISION.json','EXECUTION_GATE.json']:
            if (R/name).exists():os.replace(R/name,R/name.replace('.json',f'_attempt{n}.json'))
    write(R/'PIPELINE_START.json',{'utc':now(),'pid':os.getpid(),'worker':'1100631','attempt':len(attempts)+1,'no_test_access':True,'freeze_sha256':sha(R/'FREEZE.json')})
    if cache_ready():write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'building_cache_skipped_intact','test_access':False})
    else:stage('building_cache',[PY,'scripts/build_segresnet_native_res_cache.py'])
    if not (R/'BASELINE_E193_NATIVE_VALIDATION.json').exists():stage('baseline_native_validation',[PY,'scripts/evaluate_plain972_e193_native_validation.py'])
    stage('synthetic_gpu_smoke',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_native_res.py','smoke'])
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    write(R/'EXECUTION_GATE.json',{'utc':now(),'status':'passed','files':{str(p.relative_to(ROOT)):sha(p) for p in [R/'FREEZE.json',R/'DATA_AUDIT.json',R/'GPU_SMOKE.json',R/'BATCH_DECISION.json',R/'UNIT_TESTS.json',R/'BASELINE_E193_NATIVE_VALIDATION.json']}})
    stage('training',[PY,'scripts/watch_segresnet_native_res.py'])
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'completed','test_access':False})


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(R/'PIPELINE_FAILURE.json',{'utc':now(),'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'test_access':False})
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'failed','error':repr(e),'test_access':False});raise
