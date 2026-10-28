"""Worker entry point: wait for cache/source gates, then execute reviewed research."""
import os
import time
from pathlib import Path
from scripts.segresnet_auto_inference12h_20261028.common import R,TMP,PY,read


def main():
    TMP.mkdir(parents=True,exist_ok=True)
    while time.time()<read(R/'AUTHORIZATION.json')['deadline_unix']:
        start=TMP/'RESOURCE_START.json'
        if start.exists():
            control=read(start);module=control['module']
            assert module.startswith('scripts.segresnet_auto_inference12h_20261028.')
            os.environ['PYTHONDONTWRITEBYTECODE']='1'
            os.execv(str(PY),[str(PY),'-m',module])
        time.sleep(3)


if __name__=='__main__':main()
