"""Cut over at a complete epoch; two GPUs train one model and archive at epoch200."""
import os,sys,time,json,subprocess,signal,datetime,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import ROOT,RECORD as R,WORK as W,write,sha,now
PY=str(ROOT.parent/'sammed3d-venv/bin/python');ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NCCL_DEBUG':'WARN'}
def stage(name,**kw):write(R/'RUN_STATUS.json',{'updated_utc':now(),'stage':name,'train':972,'validation':108,'test':270,'target_epochs':200,'arms':['graymean'],'world_size':2,'batch_per_gpu':4,'global_batch':8,**kw})
def terminate(pid,expected):
 cmd=Path(f'/proc/{pid}/cmdline')
 if cmd.exists() and expected.encode() in cmd.read_bytes():os.kill(pid,signal.SIGTERM)
def audit():
 if (R/'SOURCE_AUDIT_COMPLETE.json').exists():return True
 if (R/'CACHE_INDEX.json').exists():
  data=json.loads((R/'CACHE_INDEX.json').read_text())
  if data['count']!=1080:raise RuntimeError('cache count')
  for row in data['cases']:
   if sha(ROOT/row['path'])!=row['sha256']:raise RuntimeError('cache byte hash changed')
  write(R/'SOURCE_AUDIT_COMPLETE.json',{'utc':now(),'cases':1080,'cache_index_sha256':sha(R/'CACHE_INDEX.json'),'note':'source audit overlapped early training, complete before model validation'});return True
 if list(R.glob('FAILURE_cache_*.json')):raise RuntimeError('background source verification failed')
 return False

def main():
 lock=(R/'DDP_RETRY_SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import torch
 for device in [0,1]:
  free,total=torch.cuda.mem_get_info(device)
  if free<30*(1<<30):raise RuntimeError('residual GPU allocation; need at least30GiB free per GPU')
 for n in ['DDP_FREEZE.json','DDP_ORCHESTRATION_FREEZE.json','DDP_RETRY_FREEZE.json']:
  for path,h in json.loads((R/n).read_text())['files'].items():
   if sha(ROOT/path)!=h:raise RuntimeError('frozen source changed')
 transition=json.loads((R/'DDP_TRANSITION.json').read_text());snapshot=ROOT/transition['checkpoint_path'];epoch=transition['from_epoch'];out=W/'graymean'
 if sha(snapshot)!=transition['checkpoint_sha256']:raise RuntimeError('checkpoint changed')
 write(R/'DDP_RETRY_PREACCESS.json',{'utc':now(),'reason':'passed synthetic smoke processes remained alive and occupied26GiB per GPU; terminated only those owned smoke processes','from_epoch':epoch,'unchanged_batch_per_gpu':4,'checkpoint_sha256':sha(snapshot),'previous_formal_failure_before_first_optimizer_step':True,'test_access':False})
 stage('单组双卡训练中，目标200轮',last_complete_epoch=epoch)
 log=(R/'ddp_train_retry.log').open('a');p=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_sammed3d_single972_ddp.py','train','--batch','4'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
 write(R/'DDP_LAUNCH.json',{'utc':now(),'pid':p.pid,'worker':'1097833','start_epoch':epoch+1,'world_size':2,'global_batch':8})
 publog=(R/'publication_ddp.log').open('a');pub=subprocess.Popen([PY,'scripts/publish_sammed3d_single972_ddp_status.py'],cwd=ROOT,env=ENV,stdout=publog,stderr=subprocess.STDOUT)
 try:
  while p.poll() is None:
   audited=audit();status=json.loads((out/'STATUS.json').read_text())
   write(R/'DDP_HEARTBEAT.json',{'utc':now(),'pid':p.pid,'source_audit_complete':audited,'last_complete_epoch':status['last_complete_epoch'],'alive':True});time.sleep(15)
  if p.returncode:raise RuntimeError(f'DDP training exited {p.returncode}')
  status=json.loads((out/'STATUS.json').read_text())
  if status['last_complete_epoch']!=200 or status['status']!='completed':raise RuntimeError('200 epochs not complete')
  stage('200轮完成，正在归档');result=subprocess.run([PY,'scripts/finalize_sammed3d_single972_ddp.py'],cwd=ROOT,env=ENV,capture_output=True,text=True);(R/'finalize_ddp.log').write_text(result.stdout+result.stderr)
  if result.returncode:raise RuntimeError('final archival failed')
  stage('200轮完成，报告和校验已归档')
 except BaseException:
  if p.poll() is None:
   p.terminate()
   try:p.wait(timeout=20)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  raise
 finally:log.close();publog.close()
if __name__=='__main__':
 try:main()
 except Exception as e:write(R/'DDP_RETRY_PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});stage('执行失败，200轮任务未完成',error=repr(e));raise
