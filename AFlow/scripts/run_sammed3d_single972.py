"""Durable, single-arm supervisor: smoke -> fixed cache -> 200 epochs -> archive."""
import datetime,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'archive/research_records/2026-09-16_sammed3d_graymean_single972';W=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/graymean_single972_20260916';PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(n,d):
 p=R/n;t=p.with_suffix('.tmp.json');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def stage(name,**kw):write('RUN_STATUS.json',{'updated_utc':now(),'stage':name,'train':972,'validation':108,'test':270,'target_epochs':200,'arms':['graymean'],'worker':'1097833',**kw})
def run(args,log,env=ENV):
 with (R/log).open('a') as f:
  p=subprocess.Popen([PY,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
  write('PROCESS.json',{'updated_utc':now(),'pid':p.pid,'args':args})
  while p.poll() is None:
   write('HEARTBEAT.json',{'updated_utc':now(),'pid':p.pid,'returncode':None});time.sleep(30)
  if p.returncode:raise RuntimeError(f'{args} exited {p.returncode}')
def main():
 lock=(R/'SUPERVISOR.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for n in ['PREREG.json','IMPLEMENTATION_FREEZE.json']:
  for name,digest in json.loads((R/n).read_text())['files'].items():
   if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'freeze changed: {name}')
 if not (R/'SMOKE.json').exists():
  stage('单组正式训练前合成GPU试跑');run(['scripts/sammed3d_graymean_single972.py','smoke'],'smoke.log')
 if not (R/'CACHE_INDEX.json').exists():
  stage('固定972/108缓存核验与补齐');run(['scripts/sammed3d_graymean_single972.py','cache'],'cache.log')
 a=W/'graymean';a.mkdir(exist_ok=True);args=['scripts/sammed3d_graymean_single972.py','train','--arm','graymean']
 if (a/'latest.pth').exists():args+=['--resume']
 stage('唯一灰度组训练中，目标200轮',gpu=0,second_gpu='idle')
 run(args,'train.log',{**ENV,'CUDA_VISIBLE_DEVICES':'0'})
 status=json.loads((a/'STATUS.json').read_text())
 if status.get('last_complete_epoch')!=200:raise RuntimeError('stopped before 200 epochs; task incomplete')
 stage('200轮完成，正在归档');run(['scripts/finalize_sammed3d_single972.py'],'finalize.log');stage('200轮完成，报告和校验已归档')
if __name__=='__main__':
 try:main()
 except Exception as e:
  write('FAILURE.json',{'updated_utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'test_access':False});stage('执行失败，200轮任务未完成',error=repr(e));raise
