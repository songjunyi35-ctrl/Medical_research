"""Process-separated GPU1 queue; never retries neural inference automatically."""
import subprocess,sys
from scripts.race_v2_inference_b_20261029.common import *
E=R/'edge_input_b'
def main():
    exclusive(E/'QUEUE_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=1,maximum_new_case_forwards=1728,training_untouched=True))
    for stage in ('forward','analyze'):
        subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.edge_input_b.execute',stage],check=True)
    write(E/'EDGE_GPU_RELEASE.json',dict(at=now(),state='complete',GPU_released=True,new_case_forwards=1728,result_sha256=sha(E/'RESULT.json')))
if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(E/f'QUEUE_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),automatic_neural_retry=False));raise
