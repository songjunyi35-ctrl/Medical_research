"""Exclusive bounded coordinator: data readiness -> synthetic gate -> one training -> finalization."""
import ast
import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *
from scripts.nnunet_resenc_20261013.publish import progress,checks

TRAINING_DEADLINE='2026-10-14T10:11:45+08:00'
OVERALL_DEADLINE='2026-10-14T12:41:45+08:00'

def source_files():
    todo=list((ROOT/'scripts/nnunet_resenc_20261013').glob('*.py'))
    todo.extend([ROOT/'scripts/build_segresnet_native_res_test_cache.py'])
    seen=set()
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(n,ast.Import):names=[x.name for x in n.names]
            elif isinstance(n,ast.ImportFrom) and n.module:names=[n.module]+[n.module+'.'+x.name for x in n.names]
            for name in names:
                if name.startswith(('scripts.','sammed3d.')):
                    q=ROOT/(name.replace('.','/')+'.py')
                    if q.exists():todo.append(q)
    seen.update(R/n for n in ['IDEA.md','FUSION_PLAN.md','AUTHORIZATION.json','DATA_FREEZE.json','ENVIRONMENT.json'])
    return sorted(seen)

def freeze(name,extra):
    write(R/name,{'at':now(),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in source_files()},**extra})

def run(command,logname,stage,seconds):
    with (R/logname).open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(R/f'LAUNCH_{stage}.json',{'at':now(),'stage':stage,'pid':p.pid,'command':command})
    start=time.monotonic();announced=False
    while p.poll() is None:
        state=read(OUT/'STATUS.json') if (OUT/'STATUS.json').exists() else {}
        write(R/'COORDINATOR_HEARTBEAT.json',{'at':now(),'stage':stage,'pid':os.getpid(),'child_pid':p.pid,
           'elapsed_seconds':time.monotonic()-start,'training_status':state})
        if stage.startswith('training') and state.get('last_complete_epoch',0)>=1 and not announced:
            progress('正式训练中',f'已完成第{state["last_complete_epoch"]}轮，真实进度见状态；继续冻结预算。');checks();announced=True
        if time.monotonic()-start>seconds or time.time()>datetime.datetime.fromisoformat(OVERALL_DEADLINE).timestamp():
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise RuntimeError(stage+' exceeded hard wall budget')
        time.sleep(10)
    if p.returncode:raise RuntimeError(stage+' exit '+str(p.returncode))

def main():
    lock=(R/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (R/'COORDINATOR_V2_STARTED.json').exists():raise RuntimeError('coordinator already attempted; inspect before recovery')
    write(R/'COORDINATOR_V2_STARTED.json',{'at':now(),'worker':1132660,'pid':os.getpid(),'training_deadline':TRAINING_DEADLINE,'overall_deadline':OVERALL_DEADLINE})
    progress('数据准备中','原生载荷重建与训练集独立规划尚未结束；正式患者训练0轮。');checks()
    while not (R/'DATA_READY.json').exists():
        if (R/'DATA_FAILURE.json').exists():raise RuntimeError('data preparation failed; no automatic rebuild')
        if time.time()>datetime.datetime.fromisoformat(TRAINING_DEADLINE).timestamp()-4*3600:raise RuntimeError('not enough remaining time for a meaningful run')
        write(R/'COORDINATOR_HEARTBEAT.json',{'at':now(),'stage':'waiting_data','pid':os.getpid()})
        time.sleep(15)
    assert read(R/'DATA_READY.json')['status']=='passed'
    if not (R/'CPU_CHECKS.json').exists():
        run([str(PY),'-m','scripts.nnunet_resenc_20261013.check_cpu'],'cpu.log','cpu',600)
    assert read(R/'CPU_CHECKS.json')['status']=='passed'
    freeze('SMOKE_FREEZE_V2.json',{'plan_sha256':sha(PRE/(PLANS+'.json')),'synthetic_only':True})
    command=[str(PY),'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'scripts/nnunet_resenc_20261013/run_train_v2.py')]
    run(command+['--mode','smoke'],'smoke_v2.log','smoke_v2',1800)
    smoke=read(R/'GPU_SMOKE.json');assert smoke['status']=='passed'
    step=max(r['mean_step_seconds'] for r in smoke['ranks'])
    available=min(20*3600,datetime.datetime.fromisoformat(TRAINING_DEADLINE).timestamp()-time.time())-900
    # 25% for pseudo-validation/steady-state overhead plus90min for full108 selection and saves.
    epochs=min(1000,int(max(0,available-5400)/(step*250*1.25)//25)*25)
    if epochs<250:raise RuntimeError('full-pipeline throughput permits fewer than250 epochs; no patient training')
    config=read(R/'TRAIN972_PLANS.json')['configurations']['3d_fullres']
    write(R/'BUDGET_DECISION.json',{'at':now(),'epochs':epochs,'updates_attempted_target':epochs*250,'slow_rank_seconds_per_update':step,
       'available_seconds':available,'estimated_seconds':epochs*250*step*1.25+5400,'rule':'max<=1000 rounded down25, 25% overhead+90min fixed reserve',
       'default1000_complete_budget':epochs==1000,'patch_size':config['patch_size'],'global_batch':config['batch_size'],'no_patient_model_forward_at_decision':True})
    # Hash all installed source files for the private nnU-Net implementation (in addition to local code).
    pkg=ENV/'lib/python3.11/site-packages'
    code={str(p):sha(p) for folder in ['nnunetv2','dynamic_network_architectures','batchgenerators','batchgeneratorsv2','acvl_utils'] for p in (pkg/folder).rglob('*.py')}
    write(R/'INSTALLED_SOURCE_SHA256.json',code)
    freeze('EXECUTION_FREEZE.json',{'epochs':epochs,'updates_target':epochs*250,'validation_epochs':'1, every25, final complete epoch',
       'plan_sha256':sha(PRE/(PLANS+'.json')),'split_sha256':sha(PRE/'splits_final.json'),
       'data_ready_sha256':sha(R/'DATA_READY.json'),'gpu_smoke_sha256':sha(R/'GPU_SMOKE.json'),
       'installed_source_manifest_sha256':sha(R/'INSTALLED_SOURCE_SHA256.json'),
       'training_deadline':TRAINING_DEADLINE,'overall_deadline':OVERALL_DEADLINE,'max_engineering_resumes':2,
       'validation_scan_gate':.7630921331986233,'test_tuned':True,'test_after_unique_validation_selection':True,'outer_confirmation':False})
    progress('工程门通过，接续训练',f'冻结{epochs}轮/{epochs*250}次更新尝试；尚不把启动当成完成。')
    for attempt in range(3):
        try:
            run(command+['--mode','train'],f'training_attempt{attempt}.log',f'training{attempt}',min(20*3600+600,available+900))
            break
        except Exception as e:
            write(R/f'TRAIN_ATTEMPT{attempt}_FAILURE.json',{'at':now(),'error':repr(e)})
            if attempt==2 or not (OUT/'model/fold_0/checkpoint_latest.pth').exists() or time.time()>datetime.datetime.fromisoformat(TRAINING_DEADLINE).timestamp()-1800:raise
            os.environ['AFLOW_ENGINEERING_RESUME']='1'
    assert (R/'TRAINING_DONE.json').exists()
    run([str(PY),'-m','scripts.nnunet_resenc_20261013.finalize_v2'],'finalize.log','finalization',3*3600)
    if read(R/'COMPLETION_AUDIT.json')['status']!='passed':raise RuntimeError('finalization incomplete')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(R/'COORDINATOR_FAILURE_V2.json',{'at':now(),'error':repr(e),'training_access':(R/'TRAIN_PREACCESS.json').exists(),
           'test_pixel_access':(R/'TEST_PIXEL_PREACCESS.json').exists(),'test_model_access':(R/'MODEL_PREACCESS.json').exists()})
        write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'needs_engineering_review','running':False,'error':repr(e)})
        raise
