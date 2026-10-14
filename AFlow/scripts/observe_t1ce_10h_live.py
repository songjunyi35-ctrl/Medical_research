"""Bounded live-handle observation; never launches or restarts a research job."""
import argparse,concurrent.futures,datetime,fcntl,json,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h'
HOST='fdbd:dccd:cde2:1001:a4bd:2bcd:1455:6d15'

def read(path):return json.loads(path.read_text())
def local(pid):
 p=Path('/proc')/str(pid)
 try:
  state=(p/'stat').read_text().split()[2]
  return {'pid':pid,'alive':state!='Z','state':state,'cmd':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
 except FileNotFoundError:return {'pid':pid,'alive':False}
def remote(port,pid):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p',str(port),HOST,f'ps -p {pid} -o pid=,stat=,args='],capture_output=True,text=True,timeout=20)
 parts=p.stdout.strip().split(maxsplit=2)
 return {'pid':pid,'returncode':p.returncode,'alive':p.returncode==0 and len(parts)>=2 and not parts[1].startswith('Z'),'ps':p.stdout.strip(),'stderr':p.stderr.strip()}

def main(seconds):
 assert 1<=seconds<=1800
 lock=(R/'foreground_observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 end=time.monotonic()+seconds;last_remote=-1e9
 while time.monotonic()<end:
  now=datetime.datetime.now(datetime.timezone.utc);state=read(R/'PIPELINE_STATUS.json');pid=read(R/'COMPLETION_LAUNCH.json')['pid']
  item={'utc':now.isoformat(),'master':local(pid),'resource_monitor':local(1466490),'pipeline_state':state['state'],'pipeline_age_seconds':(now-datetime.datetime.fromisoformat(state['utc'])).total_seconds()}
  if time.monotonic()-last_remote>=300:
   with concurrent.futures.ThreadPoolExecutor(2) as pool:
    fs={s:pool.submit(remote,port,read(R/f'HEARTBEAT_{s}.json')['supervisor_pid']) for s,port in [('dual',11323),('single',10235)]}
    for s,f in fs.items():
     try:item[s]=f.result()
     except Exception as exc:item[s]={'observation_error':repr(exc)}
   last_remote=time.monotonic()
  with (R/'FOREGROUND_MONITOR_OBSERVATIONS.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
  print(json.dumps(item),flush=True)
  if state['state']=='complete':break
  if not item['master']['alive'] or not item['resource_monitor']['alive'] or item['pipeline_age_seconds']>180:
   raise SystemExit('observation needs investigation; no restart performed')
  time.sleep(min(60,max(0,end-time.monotonic())))
 print(json.dumps({'bounded_observation_finished':True,'research_supervisors_not_stopped':True}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=1800);main(p.parse_args().seconds)
