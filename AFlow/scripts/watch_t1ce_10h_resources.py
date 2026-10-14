"""Read-only minute GPU/resource sampling across the two approved workers."""
import os,sys,time,subprocess,concurrent.futures,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *
HOST='fdbd:dccd:cde2:1001:a4bd:2bcd:1455:6d15'

def sample(port):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p',str(port),HOST,'nvidia-smi --query-gpu=uuid,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
 if p.returncode:return {'error':p.stderr[-1000:],'returncode':p.returncode}
 out=[]
 for line in p.stdout.splitlines():
  uid,used,total,util,temp=[s.strip() for s in line.split(',')]
  out.append({'uuid':uid,'memory_used_mib':int(used),'memory_total_mib':int(total),'gpu_utilization_percent':int(util),'temperature_c':int(temp)})
 return {'gpus':out}

def main():
 lock=(R/'resource_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  while time.time()<timestamp(P['deadline_utc']):
   futures={name:pool.submit(sample,port) for name,port in [('dual',11323),('single',10235)]};d={}
   for name,f in futures.items():
    try:d[name]=f.result()
    except Exception as e:d[name]={'error':repr(e)}
   ids=[g['uuid'] for s in d.values() for g in s.get('gpus',[])];d['unique_gpu_count']=len(set(ids));d['duplicate_gpu_allocation']=len(set(ids))!=len(ids)
   if d['duplicate_gpu_allocation']:(R/'STOP').write_text('duplicate GPU allocation detected\n')
   for name,item in d.items():
    if not isinstance(item,dict):continue
    for gpu in item.get('gpus',[]):
     if gpu['memory_used_mib']>39*1024:event(R/'RESOURCE_ALERTS.jsonl',worker=name,reason='memory above39GiB',gpu=gpu)
   event(R/'RESOURCES.jsonl',**d);write(R/'RESOURCE_STATUS.json',{'utc':now(),**d});time.sleep(60)
 write(R/'RESOURCE_MONITOR_DONE.json',{'utc':now(),'status':'complete'})

if __name__=='__main__':main()
