"""Exclusive inference-only continuation; refuses repeated test access."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *

if __name__=='__main__':
    if read(R/'RECOVERY_GPU_CHECK.json')['status']!='passed':raise RuntimeError('recovery GPU gate failed')
    for name in ['TEST_PIXEL_PREACCESS.json','MODEL_PREACCESS.json','FINALIZATION_RECOVERY_LAUNCH.json']:
        if (R/name).exists():raise RuntimeError('prior access/launch exists: '+name)
    verify_source_freeze('FINALIZATION_RECOVERY_FREEZE.json')
    launch_env=env();launch_env['AFLOW_EVAL_CPU_COUNT']='31'
    with (R/'finalize_v3.log').open('xb') as log:
        p=subprocess.Popen([str(PY),'-m','scripts.nnunet_resenc_20261013.finalize_v3'],cwd=ROOT,env=launch_env,
          stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    d={'at':now(),'pid':p.pid,'worker':1133854,'training':False,'test_model_inference_started':False,
       'checkpoint_sha256':read(R/'SELECTED_CHECKPOINT_CPU_AUDIT.json')['checkpoint_sha256'],
       'freeze_sha256':sha(R/'FINALIZATION_RECOVERY_FREEZE.json')}
    write(R/'FINALIZATION_RECOVERY_LAUNCH.json',d);print(json.dumps(d),flush=True)
