"""Stop only this run's verified supervisor and training process group."""
import os,signal,time,json,datetime
from pathlib import Path
ROOT=Path('/mlx_devbox/users/songjunyi.666/playground/AFlow')
R=ROOT/'archive/research_records/2026-09-17_sammed3d_resume139'
launch=json.loads((R/'LAUNCH.json').read_text())
def cmd(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
 except FileNotFoundError:return ''
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
supervisor=launch['supervisor_pid'];trainer=launch['pid']
s=cmd(supervisor);t=cmd(trainer)
assert not s or 'scripts/watch_sammed3d_resume139.py' in s,(supervisor,s)
assert not t or 'scripts/train_sammed3d_single972_resume139.py train --batch 4' in t,(trainer,t)
receipt={'requested_utc':now(),'reason':'user explicitly requested early stop after reviewing epoch50 test result','supervisor_pid':supervisor,'trainer_pid':trainer,'supervisor_command':s,'trainer_command':t,'worker':launch['worker'],'signals':[],'checkpoints_deleted':False}
(R/'USER_STOP_REQUEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
if s:
 os.kill(supervisor,signal.SIGTERM);receipt['signals'].append([supervisor,'SIGTERM'])
if t:
 assert os.getpgid(trainer)==trainer,'unexpected training process group'
 os.killpg(trainer,signal.SIGTERM);receipt['signals'].append(['group '+str(trainer),'SIGTERM'])
time.sleep(5)
if cmd(supervisor):
 os.kill(supervisor,signal.SIGKILL);receipt['signals'].append([supervisor,'SIGKILL'])
try:
 os.killpg(trainer,0)
except ProcessLookupError:pass
else:
 os.killpg(trainer,signal.SIGKILL);receipt['signals'].append(['group '+str(trainer),'SIGKILL'])
time.sleep(2)
alive=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  c=cmd(int(p.name))
  if 'scripts/train_sammed3d_single972_resume139.py' in c or 'scripts/watch_sammed3d_resume139.py' in c:alive.append({'pid':int(p.name),'command':c})
receipt.update(stopped_utc=now(),remaining_run_processes=alive,status='stopped' if not alive else 'stop_incomplete')
(R/'USER_STOP_PROCESS_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
assert not alive,alive
