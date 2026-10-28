"""Bounded new-run controller: exact data, synthetic gates, matched arms, archive."""
import fcntl
import subprocess
import time
from scripts.race_seg_20261028.common import *


def status(stage,**kw):write(R/'PIPELINE_STATUS.json',{'at':now(),'state':stage,'test_access':False,**kw})


def decide_budget():
    import torch
    available={}
    for arm in ARMS:
        source=read(R/f'SMOKE_{arm}.json')
        available[arm]={r['microbatch']:r for r in source['results'] if r['status']=='passed'}
    choices=set(available['control'])&set(available['race'])
    if not choices: raise RuntimeError('no shared safe microbatch')
    speeds={micro:min(available[arm][micro]['raw_blocks_per_second'] for arm in ARMS) for micro in choices}
    best=max(speeds.values());micro=min(m for m in choices if speeds[m]>=.97*best)
    update_seconds=max(available[arm][micro]['seconds_per_update'] for arm in ARMS)
    # Conservative CPU augmentation/I/O + validation allowance, additional25%.
    per_epoch=(122*(update_seconds+.35)+35)*1.25
    remaining=read(R/'AUTHORIZATION.json')['deadline_unix']-time.time()-1800
    epochs=min(200,int(remaining/per_epoch))
    if epochs<50: raise RuntimeError('cannot fit minimum50 epochs inside budget')
    decision={'at':now(),'status':'passed','epochs':epochs,'microbatch':micro,'effective_batch':8,
              'updates_per_epoch':122,'total_updates_each':epochs*122,'estimated_epoch_seconds':per_epoch,
              'estimated_training_hours':epochs*per_epoch/3600,'remaining_seconds':remaining,
              'selection':'synthetic throughput only, no patient scores','memory_ceiling_gib':38,
              'source_sha256':{arm:sha(R/f'SMOKE_{arm}.json') for arm in ARMS}}
    write(R/'BUDGET_DECISION.json',decision)
    return decision


def main():
    import torch
    R.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    lock=(R/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources('EXECUTION_FREEZE.json');deadline();disk()
    assert torch.cuda.device_count()==2
    while not (R/'DATA_READY.json').exists():
        deadline();disk()
        if (R/'PREPARE_FAILURE.json').exists():raise RuntimeError('data preparation failed; no automatic rebuild')
        heartbeat=read(R/'PREPARE_HEARTBEAT.json') if (R/'PREPARE_HEARTBEAT.json').exists() else None
        status('preparing_data',preparation=heartbeat,training_started=False)
        time.sleep(20)
    assert read(R/'DATA_READY.json')['status']=='passed'
    budget=decide_budget()
    processes={};attempts={arm:0 for arm in ARMS}
    for gpu,arm in enumerate(ARMS):
        e=env();e['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=(TMP/f'train_{arm}.log').open('ab')
        process=subprocess.Popen([str(PY),'-m','scripts.race_seg_20261028.train',arm],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();processes[arm]=process
    write(R/'TRAIN_PIDS.json',{'at':now(),'processes':{arm:p.pid for arm,p in processes.items()},'worker':1152470})
    while True:
        deadline();disk();snapshot={}
        for arm,process in processes.items():
            for name in ('HEARTBEAT.json','STATUS.json'):
                if (R/arm/name).exists():snapshot.setdefault(arm,{})[name]=read(R/arm/name)
            code=process.poll()
            if code is not None and code!=0:
                logtext=(TMP/f'train_{arm}.log').read_text(errors='replace')[-6000:]
                recoverable=any(marker in logtext for marker in ('Input/output error','DataLoader worker','Bus error','Connection reset'))
                if recoverable and attempts[arm]<2 and (TMP/'runs'/arm/'latest.pth').exists():
                    attempts[arm]+=1;gpu=ARMS.index(arm);e=env();e['CUDA_VISIBLE_DEVICES']=str(gpu)
                    write(R/arm/f'ENGINEERING_RECOVERY_{attempts[arm]}.json',{'at':now(),'prior_exit':code,'retained_epoch':read(R/arm/'STATUS.json')['last_complete_epoch']})
                    log=(TMP/f'train_{arm}.log').open('ab')
                    processes[arm]=subprocess.Popen([str(PY),'-m','scripts.race_seg_20261028.train',arm,'--resume'],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
                    write(R/'TRAIN_PIDS.json',{'at':now(),'processes':{a:p.pid for a,p in processes.items()},'worker':1152470})
                else:
                    for other in processes.values():
                        if other.poll() is None:other.terminate()
                    raise RuntimeError('training arm failed; all matched arms stopped: '+arm)
        status('training',training_started=True,budget=budget,arms=snapshot,recoveries=attempts)
        if all(p.poll() is not None for p in processes.values()):break
        time.sleep(30)
    subprocess.run([str(PY),'-m','scripts.race_seg_20261028.finalize'],cwd=ROOT,env=env(),check=True)
    for script in ('verify_aflow_brats_doc_layout.py','verify_aflow_brats_method_catalog.py','verify_aflow_brats_leaderboard.py','verify_aflow_root_artifacts.py'):
        subprocess.run([str(PY),str(ROOT/'scripts'/script)],cwd=ROOT,env=env(),check=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        # Deadline/disk failures also stop only this new experiment's processes.
        if (R/'TRAIN_PIDS.json').exists():
            import signal
            for pid in read(R/'TRAIN_PIDS.json')['processes'].values():
                try:
                    command=Path(f'/proc/{pid}/cmdline').read_bytes()
                    if b'scripts.race_seg_20261028.train' in command:
                        os.killpg(pid,signal.SIGTERM)
                except (FileNotFoundError,ProcessLookupError):pass
        write(R/'COORDINATOR_FAILURE.json',{'at':now(),'error':repr(error)})
        status('failed',error=repr(error),training_started=(R/'TRAIN_PIDS.json').exists())
        if any((R/arm/'STATUS.json').exists() for arm in ARMS):
            subprocess.run([str(PY),'-m','scripts.race_seg_20261028.finalize'],cwd=ROOT,env=env(),check=False)
        raise
