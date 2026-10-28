import subprocess,fcntl
from scripts.segresnet_w32_test270_20261028.common import *
def main():
 verify();TMP.mkdir(parents=True,exist_ok=True)
 lock=(R/'PROBES.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in read(R/'PROBE_FREEZE.json')['files'].items():assert sha(ROOT/p)==h
 if (R/'PROBES_STARTED.json').exists():raise RuntimeError('probes already started')
 write(R/'PROBES_STARTED.json',{'at':now(),'pid':os.getpid(),'patient_model_forward':False})
 subprocess.run([str(PY),'-m','scripts.segresnet_w32_test270_20261028.preflight'],cwd=ROOT,env=env(),check=True)
 for batch in (2,4,8,12):
  jobs=[]
  for rank in range(2):
   e=env();e['CUDA_VISIBLE_DEVICES']=str(rank);log=(TMP/f'probe_r{rank}_b{batch}.log').open('w')
   p=subprocess.Popen([str(PY),'-m','scripts.segresnet_w32_test270_20261028.probe',str(rank),str(batch)],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT);jobs.append(p);log.close()
  for p in jobs:
   if p.wait(timeout=900)!=0:raise RuntimeError('GPU probe process failed')
 options=[]
 for batch in (2,4,8,12):
  x=[read(R/f'PROBE_r{i}_b{batch}.json') for i in range(2)]
  if all(v['status']=='passed' for v in x):options.append({'batch':batch,'slow_rank_scans_per_second':min(v['scans_per_second'] for v in x),'ranks':x})
 if not options:raise RuntimeError('no safe inference batch on both40GB cards')
 selected=max(options,key=lambda x:x['slow_rank_scans_per_second'])
 write(R/'BATCH_DECISION.json',{'at':now(),'status':'passed',**selected,'candidate_batches':[2,4,8,12],'selection_rule':'fastest slow-rank scans/s with peak reserved<=37.5GiB and exact synthetic full/tails; no patient scores','patient_model_forward':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(R/'PROBES_FAILURE.json',{'at':now(),'error':repr(e)});raise
