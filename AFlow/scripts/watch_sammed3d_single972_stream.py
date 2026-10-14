"""Observe already launched single training; audit inputs and finalize at epoch200."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import concurrent.futures,datetime,hashlib,json,os,signal,subprocess,time,traceback
from scripts.train_sammed3d_single972_stream import materialize, ROOT, RECORD, WORK, sha, write
R=RECORD;PY=str(ROOT.parent/'sammed3d-venv/bin/python')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def alive(pid):
 p=Path(f'/proc/{pid}/stat')
 return p.exists() and p.read_text().split()[2]!='Z'
def stage(s,**kw):write(R/'RUN_STATUS.json',{'stage':s,'updated_utc':now(),'train':972,'validation':108,'test':270,'target_epochs':200,'arms':['graymean'],**kw})
def main():
 launch=json.loads((R/'STREAM_LAUNCH.json').read_text());pid=launch['pid']
 for p,h in json.loads((R/'STREAM_FREEZE.json').read_text())['files'].items():
  if sha(ROOT/p)!=h:raise RuntimeError('stream source/input changed')
 plan=json.loads((R/'INPUT_PLAN.json').read_text());missing=[r for r in plan['cases'] if not (ROOT/r['path']).exists()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(materialize,r) for r in missing]
  audit_done=False
  while True:
   for f in futures:
    if f.done():f.result()
   if (R/'CACHE_INDEX.json').exists() and not audit_done:
    index=json.loads((R/'CACHE_INDEX.json').read_text())
    if index['count']!=1080:raise RuntimeError('final cache count')
    for row in index['cases']:
     if sha(ROOT/row['path'])!=row['sha256']:raise RuntimeError('final cache hash changed')
    write(R/'SOURCE_AUDIT_COMPLETE.json',{'utc':now(),'cache_index_sha256':sha(R/'CACHE_INDEX.json'),'cases':1080,'note':'original source and cached bytes verified; source hash audit overlapped early training; no formal validation before completion'})
    audit_done=True
   failures=list(R.glob('FAILURE_cache_*.json'))
   if failures:raise RuntimeError('background source audit/cache failed: '+str(failures))
   statusfile=WORK/'graymean/STATUS.json';status=json.loads(statusfile.read_text()) if statusfile.exists() else {}
   write(R/'STREAM_HEARTBEAT.json',{'utc':now(),'pid':pid,'alive':alive(pid),'last_complete_epoch':status.get('last_complete_epoch',0),'source_audit_complete':audit_done,'prefill_remaining':sum(not f.done() for f in futures)})
   if not alive(pid):
    if status.get('status')!='completed' or status.get('last_complete_epoch')!=200:raise RuntimeError('training ended before 200 completed epochs')
    if not audit_done:raise RuntimeError('source audit missing at completion')
    break
   time.sleep(20)
 stage('200轮完成，正在归档')
 result=subprocess.run([PY,'scripts/finalize_sammed3d_single972.py'],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
 (R/'finalize.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('final archival failed')
 stage('200轮完成，报告和校验已归档')
if __name__=='__main__':
 try:main()
 except Exception as e:
  launch=json.loads((R/'STREAM_LAUNCH.json').read_text())
  if alive(launch['pid']):os.kill(launch['pid'],signal.SIGTERM)
  write(R/'STREAM_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});stage('执行失败，200轮任务未完成',error=repr(e));raise
