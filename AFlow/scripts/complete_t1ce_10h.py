"""Master-side continuous monitoring, cached-output transfer, and final audit."""
import sys,os,time,subprocess,shlex,fcntl,traceback,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.t1ce_10h_common import *
from scripts.finalize_t1ce_10h import select,publish

HOST='fdbd:dccd:cde2:1001:a4bd:2bcd:1455:6d15'
def ssh(port,command,**kwargs):
 return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-p',str(port),HOST,command],**kwargs)
def remote_stage(script,mode):
 command='cd '+shlex.quote(str(ROOT))+' && '+shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',PY,'scripts/'+script,mode])
 with (R/('completion_'+mode+'.log')).open('a') as f:
  p=ssh(11323,command,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 if p.returncode:raise RuntimeError(script+' '+mode+' failed')

def transfer():
 selected=read(R/'FINAL_PANEL_FREEZE.json')['selected'];receipts=[]
 for item in selected:
  ident=item['id'];assert '/' not in ident and all(c.isalnum() or c=='_' for c in ident)
  path=Path(item['logits_dir'])
  probe=ssh(11323,'test -d '+shlex.quote(str(path)),capture_output=True)
  if probe.returncode==0:continue
  dest=SCRATCH/'imported_logits'
  mkdir=ssh(11323,'mkdir -p '+shlex.quote(str(dest)),capture_output=True);assert mkdir.returncode==0
  source_cmd=shlex.join(['tar','-C',str(path.parent),'-cf','-',ident]);target_cmd=shlex.join(['tar','-C',str(dest),'-xf','-'])
  with (R/('transfer_'+ident+'.log')).open('w') as log:
   source_proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-p','10235',HOST,source_cmd],stdout=subprocess.PIPE,stderr=log)
   target_proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-p','11323',HOST,target_cmd],stdin=source_proc.stdout,stdout=log,stderr=log)
   source_proc.stdout.close();a=target_proc.wait(timeout=1200);b=source_proc.wait(timeout=30)
   assert a==b==0,(a,b)
  def hashes(port,p):
   code="import hashlib,json;from pathlib import Path;p=Path("+repr(str(p))+");print(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(p.glob('*.npy'))}))"
   result=ssh(port,shlex.join([PY,'-c',code]),capture_output=True,text=True,timeout=180)
   assert result.returncode==0,result.stderr;return json.loads(result.stdout)
  a=hashes(10235,path);b=hashes(11323,dest/ident);assert len(a)==270 and a==b
  receipts.append({'id':ident,'source_worker':1105022,'destination_worker':1104883,'count':270,'file_sha256':a,'no_persistent_image_relay':True})
 write(R/'LOGIT_TRANSFER_RECEIPT.json',{'utc':now(),'transfers':receipts})

def access_summary():
 reports=[read(p) for p in (R/'reports').glob('*.json')];receipts=list((R/'access').glob('*.json'))
 out={'utc':now(),'test_tuned':True,'test_labels_in_segmentation_training':False,'outer_confirmation_access':False,'model_access_receipts':{p.name:sha(p) for p in receipts},'per_candidate_unique_checkpoints':{c:len([p for p in receipts if p.name.startswith(c+'_')]) for c in 'ABC'},'output_panel_configurations':len([r for r in reports if r['id'].startswith('panel_')]),'fixed_reproduction_receipt':'RELOAD_PREACCESS.json','new_test_pixel_access':True,'test_role':'reused internal target development; not held-out confirmation'}
 assert all(n<=4 for n in out['per_candidate_unique_checkpoints'].values()) and out['output_panel_configurations']<=19
 write(R/'ACCESS_SUMMARY.json',out)

def checks():
 result={}
 for name in ['verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_brats_doc_layout.py','verify_aflow_root_artifacts.py','verify_t1ce_10h.py']:
  p=subprocess.run([PY,'scripts/'+name],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
  result[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 write(R/'DOC_CHECKS.json',result);assert all(x['returncode']==0 for x in result.values()),result

def evidence():
 files=list((R/'reports').glob('*.json'))+list((R/'access').glob('*.json'))
 files += [p for p in [R/'RESULT.json',R/'RESULT.md',R/'FINAL_PANEL_FREEZE.json',R/'FINAL_SELECTION.json',R/'ACCESS_SUMMARY.json',R/'DEPLOYMENT_CHECK.json',OUT/'deployment/manifest.json'] if p.exists()]
 files += list((OUT/'deployment').glob('*.pth'))+list((OUT/'label_relay').glob('*/index.json'))
 write(R/'EVIDENCE_SHA256.json',{'utc':now(),'files':{str(p.relative_to(ROOT)):sha(p) for p in files}})

def main():
 lock=(R/'completion_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);last_record=0;ready=(R/'RESULT.json').exists();failed=False
 write(R/'COMPLETION_LAUNCH.json',{'utc':now(),'pid':os.getpid(),'deadline':P['deadline_utc']})
 while time.time()<timestamp(P['deadline_utc']):
  statuses={c:read(OUT/c/'STATUS.json') if (OUT/c/'STATUS.json').exists() else None for c in 'ABC'}
  heartbeats={s:read(R/f'HEARTBEAT_{s}.json') if (R/f'HEARTBEAT_{s}.json').exists() else None for s in ['dual','single']}
  free=shutil.disk_usage(ROOT).free
  if free<20*2**30:(R/'STOP').write_text('persistent free disk below20GiB\n')
  state={'utc':now(),'state':'results_ready_monitoring' if ready else ('completion_needs_repair' if failed else 'active'),'elapsed_seconds':time.time()-timestamp(P['start_utc']),'seconds_remaining':timestamp(P['deadline_utc'])-time.time(),'training':statuses,'heartbeats':heartbeats,'persistent_free_bytes':free,'test_tuned':True}
  write(R/'PIPELINE_STATUS.json',state)
  if time.monotonic()-last_record>=1800:event(R/'RESEARCH_MONITOR.jsonl',**{k:v for k,v in state.items() if k!='utc'});last_record=time.monotonic()
  done=all((R/f'TRAINING_DONE_{s}.json').exists() for s in ['dual','single'])
  if done and not ready and not failed:
   try:
    select();transfer();remote_stage('finalize_t1ce_10h.py','panel');remote_stage('deploy_t1ce_10h.py','check');access_summary();publish();checks();evidence();ready=True
   except BaseException:
    write(R/'COMPLETION_FAILURE.json',{'utc':now(),'traceback':traceback.format_exc()});failed=True
  time.sleep(30)
 if not ready:raise RuntimeError('deadline reached without complete result; inspect phase failures')
 publish(True);access_summary();checks();evidence()
 write(R/'COMPLETION_AUDIT.json',{'utc':now(),'status':'complete','elapsed_seconds':time.time()-timestamp(P['start_utc']),'ten_hour_monitoring_completed':True,'all_checks_passed':True,'test_tuned':True,'target_passed':read(R/'FINAL_SELECTION.json')['target_passed'],'result_sha256':sha(R/'RESULT.json')})
 write(R/'PIPELINE_STATUS.json',{'utc':now(),'state':'complete','running':False,'test_tuned':True,'target_passed':read(R/'FINAL_SELECTION.json')['target_passed']})

if __name__=='__main__':
 try:main()
 except BaseException:
  write(R/f'COMPLETION_FATAL_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
