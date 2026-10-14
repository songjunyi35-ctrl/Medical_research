"""Create an isolated venv with the existing CUDA torch available read-only."""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *

def main():
    disk();TMP.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(BASE_PY),'-m','venv','--without-pip',str(ENV)],check=True)
    site=ENV/'lib/python3.11/site-packages'
    (site/'aflow_base.pth').write_text(str(ROOT.parent/'sammed3d-venv/lib/python3.11/site-packages')+'\n')
    constraints=TMP/'constraints.txt'
    constraints.write_text('torch==2.7.1+cu126\nnumpy==2.4.6\ntorchvision==0.22.1\n')
    subprocess.run([str(PY),'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
      '-c',str(constraints),'nnunetv2==2.6.2','dynamic-network-architectures==0.4.4',
      'batchgenerators==0.25.5','batchgeneratorsv2==0.3.6','acvl-utils==0.2.6','blosc2==4.14.1','timm==1.0.22'],check=True)
    check=subprocess.run([str(PY),'-c','import torch,numpy; from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer; print(torch.__version__,numpy.__version__)'],env=env(),capture_output=True,text=True)
    write(R/'ENVIRONMENT_RECOVERY.json',{'at':now(),'import_returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr,
      'venv':str(ENV),'shared_base_environment_modified':False,
      'pip_freeze':subprocess.check_output([str(PY),'-m','pip','freeze'],text=True).splitlines()})
    if check.returncode:raise RuntimeError(check.stderr)

if __name__=='__main__': main()
