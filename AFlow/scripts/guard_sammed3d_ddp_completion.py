"""Recover archival if NCCL teardown remains alive after all200 epochs are saved."""
import os,sys,time,json,signal,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sammed3d_graymean_single972 import ROOT,RECORD as R,WORK as W,write,sha,now
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
def stop_owned(pid,needle):
 p=Path(f'/proc/{pid}/cmdline')
 if p.exists() and needle.encode() in p.read_bytes():os.kill(pid,signal.SIGTERM)
if __name__=='__main__':
 deadline=time.monotonic()+50*3600
 while time.monotonic()<deadline:
  st=json.loads((W/'graymean/STATUS.json').read_text())
  run=json.loads((R/'RUN_STATUS.json').read_text())
  if run['stage']=='200轮完成，报告和校验已归档':break
  if st.get('status')=='completed' and st.get('last_complete_epoch')==200:
   time.sleep(120)
   run=json.loads((R/'RUN_STATUS.json').read_text())
   if run['stage']=='200轮完成，报告和校验已归档':break
   # All mathematical training work and selected checkpoints must be complete.
   best=json.loads((W/'graymean/BEST.json').read_text())
   assert sha(W/'graymean/best.pth')==best['checkpoint_sha256']
   assert sha(W/'graymean'/best['report'])==best['report_sha256']
   write(R/'DDP_COMPLETION_TEARDOWN_RECEIPT.json',{'utc':now(),'reason':'200 epoch completion saved but normal shutdown/archive not complete after120s','status':st,'no_training_steps_interrupted':True})
   sup=json.loads((R/'DDP_RETRY_SUPERVISOR_LAUNCH.json').read_text())['pid'];stop_owned(sup,'run_sammed3d_single972_ddp_retry.py')
   launcher=json.loads((R/'DDP_LAUNCH.json').read_text())['pid'];stop_owned(launcher,'torch.distributed.run')
   time.sleep(5)
   p=subprocess.run([PY,'scripts/finalize_sammed3d_single972_ddp.py'],cwd=ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});(R/'completion_guard_finalize.log').write_text(p.stdout+p.stderr)
   if p.returncode:raise RuntimeError('200 complete but archive recovery failed')
   write(R/'RUN_STATUS.json',{'utc':now(),'stage':'200轮完成，报告和校验已归档','epochs':200,'train':972,'validation':108,'test':270,'world_size':2,'batch_per_gpu':4,'global_batch':8})
   break
  if '失败' in run['stage']:break
  time.sleep(30)
