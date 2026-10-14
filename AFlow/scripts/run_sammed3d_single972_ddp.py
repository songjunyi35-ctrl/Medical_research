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
 lock=(R/'DDP_SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if json.loads((R/'DDP_SMOKE.json').read_text())['status']!='passed':raise RuntimeError('DDP smoke incomplete')
 for n in ['DDP_FREEZE.json','DDP_ORCHESTRATION_FREEZE.json']:
  for path,h in json.loads((R/n).read_text())['files'].items():
   if sha(ROOT/path)!=h:raise RuntimeError('DDP frozen source mismatch')
 out=W/'graymean'
 while True:
  status=json.loads((out/'STATUS.json').read_text())
  if status['last_complete_epoch']>=2:break
  audit();time.sleep(1)
 for name,expected in [('STREAM_WATCH_LAUNCH.json','watch_sammed3d_single972_stream.py'),('PUBLICATION_LAUNCH.json','publish_sammed3d_single972_status.py')]:
  terminate(json.loads((R/name).read_text())['pid'],expected)
 terminate(json.loads((R/'STREAM_LAUNCH.json').read_text())['pid'],'train_sammed3d_single972_stream.py')
 time.sleep(2)
 status=json.loads((out/'STATUS.json').read_text());epoch=status['last_complete_epoch'];snapshot=out/f'single_gpu_e{epoch:03d}_transition.pth'
 if snapshot.exists():raise RuntimeError('transition snapshot already exists')
 os.link(out/'latest.pth',snapshot)
 write(R/'DDP_TRANSITION.json',{'utc':now(),'from_epoch':epoch,'start_epoch':epoch+1,'checkpoint_path':str(snapshot.relative_to(ROOT)),'checkpoint_sha256':sha(snapshot),'elapsed_seconds':status['elapsed_seconds'],'ledger_sha256':sha(out/'TRAIN.jsonl'),'partial_next_epoch':'any in-memory steps after final snapshot discarded; no completed epoch removed','batch_before':1,'batch_after':8,'gpu_count_after':2,'test_access':False})
 stage('单组双卡训练中，目标200轮',last_complete_epoch=epoch)
 log=(R/'ddp_train.log').open('a');p=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_sammed3d_single972_ddp.py','train','--batch','4'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
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
 except Exception as e:write(R/'DDP_PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});stage('执行失败，200轮任务未完成',error=repr(e));raise
