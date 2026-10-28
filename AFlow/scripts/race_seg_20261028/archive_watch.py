"""Main-host archival watcher. Uploads summary documents only, never weights/data.

This is administrative continuation of the user's archive request. It does not
start/resume training, select models or access patient arrays. Run on the main
host, where the existing repository deploy identity is already configured.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from scripts.race_seg_20261028.common import ROOT,R,read,write,now

RELEASE=Path('/tmp/medical_research_publish_20261014')
EXPECTED='https://github.com/songjunyi35-ctrl/Medical_research.git'
SUMMARY_FILES=('RESULT.md','RESULT.json','COMPLETION_AUDIT.json','PIPELINE_STATUS.json',
               'BUDGET_DECISION.json','DATA_READY.json','PREPARE_FAILURE.json','COORDINATOR_FAILURE.json')


def git(*args,environment=None):
    return subprocess.check_output(['git',*args],cwd=RELEASE,env=environment,text=True,stderr=subprocess.STDOUT)


def audited_result():
    required=[R/'RESULT.json',R/'RESULT.md',R/'COMPLETION_AUDIT.json']
    if not all(path.is_file() for path in required):return None
    result=read(required[0]);audit=read(required[2])
    digest=hashlib.sha256(required[0].read_bytes()).hexdigest()
    if audit.get('result_sha256')!=digest or audit.get('state')!=result.get('state'):return None
    return result


def publish():
    lock=(R/'ARCHIVE_PUBLICATION.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert git('remote','get-url','origin').strip()==EXPECTED
    if git('status','--porcelain').strip():raise RuntimeError('archive checkout has other pending changes')
    git('fetch','origin','main');git('merge','--ff-only','FETCH_HEAD')
    result=audited_result()
    copied=[]
    destination=RELEASE/'AFlow'/R.relative_to(ROOT);destination.mkdir(parents=True,exist_ok=True)
    for name in SUMMARY_FILES:
        if result is None and name in ('RESULT.md','RESULT.json','COMPLETION_AUDIT.json'):continue
        source=R/name
        if not source.is_file():continue
        if source.stat().st_size>2*2**20:raise RuntimeError('unexpectedly large summary '+name)
        target=destination/name;shutil.copy2(source,target);copied.append(target)
    summary='本次实验已结束；具体状态、完成轮数及验证指标见[结果记录](AFlow/archive/research_records/2026-10-28_race_seg/RESULT.md)。'
    if result is None:
        summary='本次实验尚无通过完整归档核查的结果；最后可见阶段见[状态记录](AFlow/archive/research_records/2026-10-28_race_seg/PIPELINE_STATUS.json)。'
    summary+=' 原972/108、单seed、复用验证；无新增test270或独立外部确认。'
    path=RELEASE/'README.md';text=path.read_text();marker='<!-- RACE_EXPERIMENT_OUTCOME -->'
    if marker not in text:
        section=f'\n{marker}\n{summary}\n<!-- END_RACE_EXPERIMENT_OUTCOME -->\n\n'
        text=text.replace('## 阅读入口\n',section+'## 阅读入口\n',1);path.write_text(text);copied.append(path)
    manifest=RELEASE/'MANIFEST.sha256.tsv';lines=manifest.read_text().splitlines()
    order=[line.split('\t',2)[2] for line in lines];entries={line.split('\t',2)[2]:line for line in lines}
    for file in copied:
        body=file.read_bytes();relative=file.relative_to(RELEASE).as_posix()
        entries[relative]=f'{hashlib.sha256(body).hexdigest()}\t{len(body)}\t{relative}'
    manifest.write_text('\n'.join([entries[k] for k in order]+[entries[k] for k in sorted(entries.keys()-set(order))])+'\n')
    subprocess.run(['python','verify_migration_bundle.py'],cwd=RELEASE,check=True)
    git('add',*[p.relative_to(RELEASE).as_posix() for p in copied],'MANIFEST.sha256.tsv')
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=RELEASE).returncode==0:return
    git('commit','-m','Archive RACE-Seg matched experiment outcome')
    publish_env=os.environ.copy()
    publish_env['GIT_SSH_COMMAND']='ssh -i /tmp/medical_research_archive_20261028_deploy_key -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/tmp/medical_research_github_known_hosts'
    git('push','origin','HEAD:main',environment=publish_env)
    git('fetch','origin','main')
    local=git('rev-parse','HEAD').strip();remote=git('rev-parse','FETCH_HEAD').strip()
    assert local==remote
    write(R/'FINAL_PUBLICATION.json',{'at':now(),'status':'uploaded_and_remote_verified','commit':local,
           'url':f'https://github.com/songjunyi35-ctrl/Medical_research/commit/{local}',
           'files':[p.relative_to(RELEASE).as_posix() for p in copied],'clinical_payloads_uploaded':False})


def main():
    deadline=read(R/'AUTHORIZATION.json')['deadline_unix']
    lock=(R/'ARCHIVE_WATCH.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(R/'ARCHIVE_WATCH.json',{'at':now(),'pid':os.getpid(),'state':'waiting_for_outcome','deadline_unix':deadline,'summaries_only':True})
    while time.time()<deadline:
        if audited_result() is not None or (R/'COORDINATOR_FAILURE.json').exists():
            time.sleep(30)  # Let checkpoint/result auditing and document checks finish.
            publish();return
        time.sleep(30)
    write(R/'ARCHIVE_WATCH_TIMEOUT.json',{'at':now(),'state':'budget_end','last_pipeline':read(R/'PIPELINE_STATUS.json')})
    publish()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(R/'ARCHIVE_WATCH_FAILURE.json',{'at':now(),'error':repr(error)});raise
