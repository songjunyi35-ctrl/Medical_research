import subprocess,fcntl,traceback
from scripts.segresnet_w32_test270_20261028.common import *
def main():
 lock=(R/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (R/'COORDINATOR_STARTED.json').exists():raise RuntimeError('coordinator already started')
 verify();write_new(R/'COORDINATOR_STARTED.json',{'at':now(),'pid':os.getpid(),'worker':WORKER})
 while not ((R/'CACHE_READY.json').exists() and (R/'BATCH_DECISION.json').exists()):
  deadline();disk()
  failures=list(R.glob('CACHE_FAILURE.json'))+list(R.glob('PROBES_FAILURE.json'))+list(R.glob('PROBE_FAILURE_*.json'))
  if failures:raise RuntimeError('preparation failed: '+str(failures))
  write(R/'COORDINATOR_HEARTBEAT.json',{'at':now(),'state':'waiting_cache_and_hardware_batch','cache_ready':(R/'CACHE_READY.json').exists(),'batch_ready':(R/'BATCH_DECISION.json').exists()});time.sleep(10)
 assert read(R/'CACHE_READY.json')['historical_payload_exact'] and read(R/'BATCH_DECISION.json')['status']=='passed'
 from scripts.segresnet_w32_test270_20261028.publish import progress,checks
 progress('evaluating','单N4缓存四载荷SHA与历史完全一致，两卡合成批量门通过；开始固定171模型的一次270推理。');checks()
 expected=rows('test');batch=read(R/'BATCH_DECISION.json')['batch']
 write_new(R/'MODEL_PREACCESS.json',{'at':now(),'epoch':171,'checkpoint_sha256':CHECKPOINT_SHA,'freeze_sha256':sha(R/'EXECUTION_FREEZE.json'),'batch_per_gpu':batch,'threshold':.5,'precision':'bf16','rank_case_ids':{str(i):[d['case_id'] for d in expected[i::2]] for i in range(2)},'cache_sha256':sha(R/'CACHE_READY.json'),'test_tuned_development':True,'current_test_retuning':False,'no_automatic_repeated_forward':True})
 jobs=[]
 for rank in range(2):
  e=env();e['CUDA_VISIBLE_DEVICES']=str(rank);log=(TMP/f'evaluate_rank{rank}.log').open('w')
  p=subprocess.Popen([str(PY),'-m','scripts.segresnet_w32_test270_20261028.evaluate',str(rank)],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(p);log.close()
 try:
  while any(p.poll() is None for p in jobs):
   deadline()
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError('patient inference failed; no automatic repeat')
   status('evaluating_test270',running=True,selected_batch=batch,rank_progress=[read(R/f'HEARTBEAT_{i}.json') if (R/f'HEARTBEAT_{i}.json').exists() else None for i in range(2)]);time.sleep(5)
  if any(p.returncode!=0 for p in jobs):raise RuntimeError('patient inference exit failed')
 except BaseException:
  for p in jobs:
   if p.poll() is None:p.terminate()
  raise
 status('raw_completed_statistics_and_publication',running=True)
 subprocess.run([str(PY),'-m','scripts.segresnet_w32_test270_20261028.publish'],cwd=ROOT,env=env(),check=True)
 write(R/'COORDINATOR_DONE.json',{'at':now(),'one_fixed_model_test_complete':True,'new_outer_confirmation_access':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(R/'COORDINATOR_FAILURE.json',{'at':now(),'error':repr(e),'traceback':traceback.format_exc(),'model_access_started':(R/'MODEL_PREACCESS.json').exists(),'raw_complete':(R/'RAW_TEST270.json').exists(),'no_automatic_inference_retry':True});status('failed',running=False,error=repr(e));raise
