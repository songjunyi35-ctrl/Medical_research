"""CPU-only decoding queue; no neural retry and no270 access."""
import subprocess,time
from scripts.race_v2_inference_a_20261029.common import *
def main():
    while not (R/'ATOMS_READY.json').exists():
        guard()
        if (R/'QUEUE_FAILURE.json').exists() or (R/'CACHE_FAILURE.json').exists():raise RuntimeError('prepare/forward failed; no neural retry')
        time.sleep(10)
    verify();expected=read(R/'SUPERVISION_FREEZE.json')['source_sha256']
    assert expected==sha(Path(__file__))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with (TMP/'analyze.log').open('xb') as log:
        p=subprocess.Popen([str(ROOT.parent/'sammed3d-venv/bin/python'),'-u','-m','scripts.race_v2_inference_a_20261029.analyze'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(R/'ANALYSIS_PID.json',dict(at=now(),worker=1154121,pid=p.pid,device='CPU',workers=8))
    while p.poll() is None:
        guard();time.sleep(10)
    assert p.returncode==0,'analysis failed; no implicit candidate expansion'
if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write(R/'SUPERVISION_FAILURE.json',dict(at=now(),error=repr(e),traceback=traceback.format_exc()));raise
