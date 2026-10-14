"""Engineering amendment002: original supervision with committed NCCL cleanup runner."""
import sys,os,time,subprocess,signal,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *

ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/aflow_matplotlib_cache'}
def heartbeat(which,phase,**extra):
 resource=disk();write(R/f'HEARTBEAT_{which}.json',{'utc':now(),'supervisor_pid':os.getpid(),'worker':1104883 if which=='dual' else 1105022,'phase':phase,'deadline':P['deadline_utc'],'seconds_remaining':max(0,timestamp(P['deadline_utc'])-time.time()),'resources':resource,**extra})
def wait_for(which,path):
 while not Path(path).exists():
  heartbeat(which,'waiting_for_'+Path(path).name)
  if time.time()>timestamp(P['new_training_stop_utc']) or (R/'STOP').exists():raise RuntimeError('deadline reached while waiting')
  time.sleep(15)
def kill_group(proc):
 if proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def run(which,arm,mode):
 command=[PY]
 if arm=='A':command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
 command+=['scripts/train_t1ce_10h_v2.py',mode,arm]
 for attempt in range(1,4 if mode=='train' else 2):
  if mode=='train' and time.time()>timestamp(P['new_training_stop_utc']):return
  logpath=R/arm/f'{mode}_v2_attempt{attempt}.log'
  if logpath.exists():raise RuntimeError('refuse duplicate attempt log')
  with logpath.open('w') as log:
   proc=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(R/arm/f'LAUNCH_{mode}_v2_{attempt}.json',{'utc':now(),'pid':proc.pid,'command':command,'attempt':attempt,'worker':which})
   changed=time.monotonic();last_size=-1;last_record=0
   try:
    while proc.poll() is None:
     progress=read(OUT/arm/'STATUS.json') if (OUT/arm/'STATUS.json').exists() else None
     heartbeat(which,mode+'_'+arm,child_pid=proc.pid,attempt=attempt,progress=progress)
     size=logpath.stat().st_size
     if size!=last_size:last_size=size;changed=time.monotonic()
     if time.monotonic()-changed>600:raise RuntimeError('no progress for600s')
     if time.time()>=timestamp(P['training_stop_utc']) or (R/'STOP').exists():
      kill_group(proc);write(R/arm/'DEADLINE_STOP.json',{'utc':now(),'reason':'global training deadline or safety stop','progress':progress});return
     if time.monotonic()-last_record>=1800:
      event(R/'RESEARCH_MONITOR.jsonl',worker=which,arm=arm,mode=mode,progress=progress,resources=disk());last_record=time.monotonic()
     time.sleep(30)
   except BaseException:
    kill_group(proc);raise
  if proc.returncode==0:return
  event(R/arm/'RECOVERY.jsonl',attempt=attempt,returncode=proc.returncode,latest_exists=(OUT/arm/'latest.pth').exists())
  if mode!='train' or attempt==3 or not (OUT/arm/'latest.pth').exists():raise RuntimeError(f'{arm} {mode} failed exit{proc.returncode}')

def main(which):
 verify();lock=(R/f'owner_{which}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(R/f'SUPERVISOR_{which}.json',{'utc':now(),'pid':os.getpid(),'state':'running'})
 if which=='dual':
  wait_for(which,R/'DATA_READY.json')
  if not (R/'REFERENCE_READY.json').exists():
   with (R/'references.log').open('a') as log:
    proc=subprocess.Popen([PY,'scripts/evaluate_t1ce_10h.py','references'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while proc.poll() is None:
     heartbeat(which,'reference_panel',child_pid=proc.pid)
     if time.time()>=timestamp(P['new_training_stop_utc']):kill_group(proc);raise RuntimeError('reference deadline')
     time.sleep(30)
    if proc.returncode:raise RuntimeError('reference panel failed')
  run(which,'A','smoke');run(which,'A','train')
 else:
  wait_for(which,R/'B/READY.json');wait_for(which,R/'REFERENCE_READY.json')
  run(which,'B','smoke');run(which,'B','train')
  if read(R/'DIAGNOSTIC.json')['enable_C']:
   run(which,'C','smoke');run(which,'C','train')
  else:write(R/'C/STOP_RESULT.json',{'utc':now(),'state':'not_started','reason':'initial FNR-FDR below0.05','test_access':False,'completed_updates':0})
 write(R/f'TRAINING_DONE_{which}.json',{'utc':now(),'state':'training_phase_finished','no_new_training':True})
 while time.time()<timestamp(P['deadline_utc']):
  heartbeat(which,'training_finished_monitoring_until_deadline')
  time.sleep(30)
 write(R/f'SUPERVISOR_{which}.json',{'utc':now(),'state':'complete','pid':os.getpid()})
 heartbeat(which,'complete')

if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:
  write(R/f'SUPERVISOR_FAILURE_{sys.argv[1]}_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
