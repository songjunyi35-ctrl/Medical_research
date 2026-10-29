"""Head-node async recovery backup from immutable worker hardlink snapshots."""
import argparse
import shlex
import subprocess
import torch
from scripts.race_v2_20261029.common import *

HOST='tiger@fdbd:dccd:cde2:1086:109d:c0df:800c:b0e2'
KNOWN='/tmp/aflow_race_v2_20261029_known_hosts'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o',f'UserKnownHostsFile={KNOWN}','-p','10275',HOST]

def snapshot(name,epoch):
    # link() pins the checkpoint inode while the trainer atomically replaces
    # latest/best, so the copied file cannot be a mix of two epochs.
    script=("from pathlib import Path;import os,json,hashlib;"
       f"source=Path({str(TMP/'run')!r})/{(name+'.pth')!r};"
       f"folder=Path({str(TMP/'backup_snapshots')!r});folder.mkdir(parents=True,exist_ok=True);"
       f"path=folder/{(name+'_e'+str(epoch)+'.pth')!r};"
       "os.link(source,path) if not path.exists() else None;"
       "h=hashlib.sha256();f=path.open('rb');"
       "[(h.update(b)) for b in iter(lambda:f.read(8<<20),b'')];"
       "print(json.dumps({'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}))")
    output=subprocess.check_output(SSH+[shlex.join([str(PY),'-c',script])],text=True,timeout=60)
    return json.loads(output)

def main():
    BACKUP.mkdir(parents=True,exist_ok=True);last={};events=[]
    while True:
        deadline();state=read(R/'STATUS.json') if (R/'STATUS.json').exists() else None
        if state:
            best=read(R/'BEST.json') if (R/'BEST.json').exists() else {'new_epoch':0}
            for name,epoch in [('latest',state['new_epoch']),('best',best['new_epoch'])]:
                if last.get(name)==epoch:continue
                remote=snapshot(name,epoch);temporary=BACKUP/f'{name}.pth.copying'
                subprocess.run(['scp','-q','-P','10275','-o','BatchMode=yes','-o',f'UserKnownHostsFile={KNOWN}',
                    f'tiger@[fdbd:dccd:cde2:1086:109d:c0df:800c:b0e2]:{remote["path"]}',str(temporary)],check=True,timeout=180)
                assert temporary.stat().st_size==remote['bytes'] and sha(temporary)==remote['sha256']
                saved=torch.load(temporary,map_location='cpu',weights_only=False)
                assert saved['new_epoch']>=epoch
                assert 'optimizer_state_dict' in saved
                assert all(int(s['step'])==saved['new_updates'] for s in saved['optimizer_state_dict']['state'].values())
                target=BACKUP/f'{name}.pth';os.replace(temporary,target)
                last[name]=saved['new_epoch']
                event=dict(at=now(),name=name,new_epoch=saved['new_epoch'],new_updates=saved['new_updates'],
                    sha256=remote['sha256'],bytes=remote['bytes'],local_path=str(target),worker_snapshot=remote['path'],
                    sha_verified=True,optimizer_checked=True)
                events.append(event)
                write(R/'RECOVERY_BACKUP.json',dict(at=now(),status='active',head_node_scratch=str(BACKUP),
                    latest_by_name={row['name']:row for row in events},events=events))
            if (R/'DONE.json').exists() and last.get('latest')==read(R/'DONE.json')['new_epoch']:
                write(R/'RECOVERY_BACKUP_COMPLETE.json',dict(at=now(),status='complete',last=last,head_node_scratch=str(BACKUP)))
                break
        if list(R.glob('COORDINATOR_FAILURE_*.json')):
            write(R/'RECOVERY_BACKUP_STOPPED.json',dict(at=now(),reason='coordinator failed; copied complete checkpoints retained',last=last));break
        time.sleep(20)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(R/f'BACKUP_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error)));raise
