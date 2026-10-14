"""Exclusive synthetic-gated cache -> frozen200-epoch arm -> archival coordinator."""
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'archive/research_records/2026-10-10_segresnet_valid_region_supervision'
OUT = ROOT / 'workspace/BraTS/segresnet_t1ce_wt/valid_region_20261010'
PY = ROOT.parent / 'sammed3d-venv/bin/python'
MAX_TOTAL_WALL_SECONDS = 18 * 3600
sys.path.insert(0, str(ROOT))
from scripts.finalize_valid_region_multiscale_20261010 import publish_progress


def now(): return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def write(path,data):
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def dependencies(seeds):
    pending=list(seeds);seen=set()
    while pending:
        p=pending.pop()
        if p in seen or not p.exists():continue
        seen.add(p)
        if p.suffix!='.py':continue
        for node in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module and not node.level:
                names=[node.module]+[node.module+'.'+x.name for x in node.names]
            for n in names:
                if n.startswith(('scripts.','sammed3d.')):
                    f=ROOT/(n.replace('.','/')+'.py');pkg=ROOT/n.replace('.','/')/'__init__.py'
                    if f.exists():pending.append(f)
                    elif pkg.exists():pending.append(pkg)
    return seen


def validate_synthetic_gate():
    frozen=read(R/'SMOKE_FREEZE_V2.json')
    for p,digest in frozen['source_sha256'].items():
        if sha(ROOT/p)!=digest:raise RuntimeError('synthetic source changed: '+p)
    smoke=read(R/'GPU_SMOKE.json')
    if smoke['status']!='passed' or smoke['worker']!=1127648 or smoke['global_batch']!=8:
        raise RuntimeError('GPU gate failed')
    if read(R/'CPU_CHECKS_V2.json')['status']!='passed':raise RuntimeError('CPU gate failed')
    for check in smoke['ddp_checks']:
        if check['max_parameter_difference_between_ranks']!=0 or check['optimizer_steps']!=[2]:
            raise RuntimeError('DDP gate failed')


def freeze_execution():
    validate_synthetic_gate()
    audit=read(R/'DATA_AUDIT.json')
    if audit['status']!='passed' or not audit['historical_pixel_payload_exact']:
        raise RuntimeError('native payload gate failed')
    if (R/'EXECUTION_FREEZE.json').exists() or (R/'TRAIN_PREACCESS.json').exists():
        raise RuntimeError('execution already frozen or started; inspect instead of new launch')
    seeds=[ROOT/'scripts'/n for n in ['train_valid_region_multiscale_20261010_v2.py',
           'valid_region_multiscale_20261010_common_v2.py','check_valid_region_multiscale_20261010_v2.py',
           'build_valid_region_multiscale_cache_20261010.py','supervise_valid_region_multiscale_20261010.py',
           'finalize_valid_region_multiscale_20261010.py','run_valid_region_multiscale_20261010.py']]
    files=dependencies(seeds)
    files.update(R/n for n in ['IDEA.md','AUTHORIZATION.json','SMOKE_FREEZE_V2.json','CACHE_FREEZE.json',
                               'CPU_CHECKS_V2.json','GPU_SMOKE.json','DATA_AUDIT.json','NUMERIC_ENGINEERING_AMENDMENT_001.json'])
    files.update([ROOT/'workspace/BraTS/segresnet_t1ce_wt/native_multiscale_resume50_20260929/validation_e166.json',
                  ROOT/'archive/research_records/2026-09-21_segresnet_pretrain_native_res/REFERENCE_NATIVE_VALIDATION.json'])
    source=ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/pretrain/final.pth';files.add(source)
    for name in ['train_t1ce_wt.jsonl','validation_t1ce_wt.jsonl','sealed_test/test_cases_t1ce.jsonl']:
        files.add(ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'/name)
    write(R/'EXECUTION_FREEZE.json',{'recorded_at':now(),'worker':1127648,'status':'frozen_before_patient_model_forward',
          'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
          'pretrain_sha256':sha(source),'data_audit_sha256':sha(R/'DATA_AUDIT.json'),
          'gpu_smoke_sha256':sha(R/'GPU_SMOKE.json'),'cpu_checks_sha256':sha(R/'CPU_CHECKS_V2.json'),
          'target_epochs':200,'global_batch':8,'updates_per_epoch':122,'per_gpu_batch':4,
          'no_epoch50_score_stop':True,'promotion_scan_min':.7630921331986233,
          'subject_floor':.7396687182784081,'point_floor':94,'paired_ci95_lower_min':0,
          'new_test_model_forward':False,'new_outer_confirmation_access':False,
          'validation_selection':'max native Scan, earliest tie; fixed108/50',
          'test_tuned_development':True,'training_wall_budget_hours':14,'max_engineering_resumes':2})


def latest_cache_progress():
    path=R/'cache_build.log'
    if not path.exists():return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-8192));lines=stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            d=json.loads(line)
            if 'done' in d:return d
        except ValueError:pass
    return None


def stop_owned_process(proc,stage):
    if stage=='training_and_finalization' and (R/'HEARTBEAT.json').exists():
        pid=read(R/'HEARTBEAT.json').get('train_pid')
        if pid:
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)


def run_stage(script,logname,stage,started):
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/aflow_valid_region_mpl',
                                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (R/logname).open('xb') as log:
        proc=subprocess.Popen([str(PY),'scripts/'+script],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(R/f'LAUNCH_{stage}.json',{'started_at':now(),'worker':1127648,'stage':stage,'pid':proc.pid,'script':script})
    published_first_epoch=False
    while proc.poll() is None:
        state=read(OUT/'STATUS.json') if (OUT/'STATUS.json').exists() else {}
        write(R/'COORDINATOR_HEARTBEAT.json',{'updated_at':now(),'worker':1127648,'coordinator_pid':os.getpid(),
              'stage':stage,'stage_pid':proc.pid,'stage_alive':True,'elapsed_seconds':time.monotonic()-started,
              'cache_progress':latest_cache_progress(),'training_status':state,'new_test_model_forward':False})
        if stage=='training_and_finalization' and state.get('last_complete_epoch',0)>=1 and not published_first_epoch:
            publish_progress('正式训练中','第1轮及108验证已完整完成；继续固定200轮上限。')
            published_first_epoch=True
        if time.monotonic()-started>MAX_TOTAL_WALL_SECONDS:
            stop_owned_process(proc,stage);raise RuntimeError('18h overall wall budget exhausted')
        time.sleep(10)
    if proc.returncode:raise RuntimeError(stage+' exited '+str(proc.returncode)+'; inspect log, no coordinator auto-retry')


def main():
    lock=(R/'COORDINATOR.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in read(R/'COORDINATOR_FREEZE.json')['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('coordinator source changed: '+path)
    validate_synthetic_gate()
    if (R/'COORDINATOR_STARTED.json').exists():raise RuntimeError('coordinator already attempted; explicit recovery audit required')
    write(R/'COORDINATOR_STARTED.json',{'started_at':now(),'worker':1127648,'pid':os.getpid(),'max_total_wall_hours':18,
          'gpu_smoke_sha256':sha(R/'GPU_SMOKE.json'),'patient_model_forward_at_launch':False})
    started=time.monotonic()
    publish_progress('合成门通过，重建原生缓存','CPU/bf16全1等价、增强/回退、global8与尾批4/参数同步均通过。')
    run_stage('build_valid_region_multiscale_cache_20261010.py','cache_build.log','cache',started)
    freeze_execution()
    write(R/'PIPELINE_STATUS.json',{'updated_at':now(),'state':'execution_frozen_awaiting_train','running':True,'worker':1127648,
          'training_started':False,'new_test_model_forward':False})
    publish_progress('载荷SHA通过，训练监督接续','缓存载荷与旧组完全一致，执行源码/数据/指标已冻结。')
    run_stage('supervise_valid_region_multiscale_20261010.py','supervisor.log','training_and_finalization',started)
    if read(R/'PIPELINE_STATUS.json')['state']!='complete' or read(R/'COMPLETION_AUDIT.json')['status']!='passed':
        raise RuntimeError('supervisor returned without complete archival boundary')
    write(R/'COORDINATOR_DONE.json',{'finished_at':now(),'worker':1127648,'completed_epochs':200,
          'result_sha256':sha(R/'FINAL_RESULT.json'),'elapsed_seconds':time.monotonic()-started,'new_test_model_forward':False})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(R/'COORDINATOR_FAILURE.json',{'updated_at':now(),'worker':1127648,'error':repr(exc),
              'training_started':(R/'TRAIN_PREACCESS.json').exists(),'new_test_model_forward':False})
        raise
