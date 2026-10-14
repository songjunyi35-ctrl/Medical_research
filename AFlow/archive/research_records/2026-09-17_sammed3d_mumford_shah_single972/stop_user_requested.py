"""Stop only verified processes belonging to the explicitly cancelled MS run."""
import os,signal,time,json,datetime
from pathlib import Path
ROOT=Path('/mlx_devbox/users/songjunyi.666/playground/AFlow')
R=ROOT/'archive/research_records/2026-09-17_sammed3d_mumford_shah_single972'
launch=json.loads((R/'LAUNCH.json').read_text())
def cmd(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
 except FileNotFoundError:return ''
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
sup=launch['supervisor_pid'];pid=launch['pid'];a=cmd(sup);b=cmd(pid)
assert not a or 'scripts/watch_sammed3d_mumford_shah_972.py' in a
assert not b or 'scripts/train_sammed3d_mumford_shah_972.py train' in b
receipt={'requested_utc':now(),'reason':'user explicitly requested early stop after plateau beyond epoch25','worker':launch['worker'],'supervisor_pid':sup,'trainer_pid':pid,'verified_commands':[a,b],'signals':[],'deleted_files':[]}
(R/'USER_STOP_REQUEST.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
if a:os.kill(sup,signal.SIGTERM);receipt['signals'].append([sup,'SIGTERM'])
if b:
 assert os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM);receipt['signals'].append(['group '+str(pid),'SIGTERM'])
time.sleep(5)
if cmd(sup):os.kill(sup,signal.SIGKILL);receipt['signals'].append([sup,'SIGKILL'])
try:os.killpg(pid,0)
except ProcessLookupError:pass
else:os.killpg(pid,signal.SIGKILL);receipt['signals'].append(['group '+str(pid),'SIGKILL'])
time.sleep(2)
alive=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  c=cmd(int(p.name))
  if 'scripts/train_sammed3d_mumford_shah_972.py' in c or 'scripts/watch_sammed3d_mumford_shah_972.py' in c:alive.append({'pid':int(p.name),'command':c})
receipt.update(stopped_utc=now(),remaining_run_processes=alive,status='stopped' if not alive else 'stop_incomplete')
(R/'USER_STOP_PROCESS_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
assert not alive
