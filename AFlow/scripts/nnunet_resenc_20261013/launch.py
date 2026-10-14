"""Start the exclusive coordinator detached, preserving a reviewable launch receipt."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *

if __name__=='__main__':
    if (R/'COORDINATOR_STARTED.json').exists() or (R/'LAUNCH.json').exists():raise RuntimeError('coordinator already launched')
    with (R/'pipeline.log').open('xb') as log:
        p=subprocess.Popen([str(PY),'-m','scripts.nnunet_resenc_20261013.pipeline'],cwd=ROOT,env=env(),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'at':now(),'worker':1132660,'pid':p.pid,'script_sha256':sha(ROOT/'scripts/nnunet_resenc_20261013/pipeline.py'),
       'phase':'wait for fixed data gates; no implicit duplicate data construction'}
    write(R/'LAUNCH.json',receipt);print(json.dumps(receipt),flush=True)
