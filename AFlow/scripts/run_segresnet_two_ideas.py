"""Persistent sequential two-GPU runner; source preparation overlaps first training."""
import sys, os, json, time, subprocess, traceback, threading, fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.segresnet_two_ideas_common import *
from scripts.finalize_segresnet_two_ideas import update_block
PY=str(ROOT.parent/'sammed3d-venv/bin/python')
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2',
     'CUDA_VISIBLE_DEVICES':'0,1','AFLOW_BUILD_PROCESSES':'24'}


def stage(command,label,training=None):
    log=R/(label+'.log')
    with log.open('a') as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        while p.poll() is None:
            status=json.loads((OUT/training/'STATUS.json').read_text()) if training and (OUT/training/'STATUS.json').exists() else None
            write(R/(label+'_HEARTBEAT.json'),{'utc':now(),'stage':label,'pid':p.pid,'training':status})
            time.sleep(20)
    if p.returncode:raise RuntimeError(f'{label} exit={p.returncode}; see {log}')


def train(name):
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'training','stage':name,'worker':1104883,'test_tuning_used':False})
    rec=RECORDS.get(name,RECORDS['finetune_pretrained'])
    write(rec/'STATUS.json',{'utc':now(),'status':'running_'+name,'worker':1104883,'execution_frozen':True,
                            'training_started':True,'progress_path':str((OUT/name/'STATUS.json').relative_to(ROOT)),'test_access':False})
    brief=f'2026-09-21两候选顺序执行：当前{name}，worker1104883双A100；全局batch8、单T1ce、原972/108/270不变。自动记录进度并在每组完成后评估固定候选，用户test定向训练许可未动用。[实时进度](archive/research_records/2026-09-21_segresnet_two_experiments/PIPELINE_STATUS.json)。'
    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:
        update_block(ROOT/filename,'TWO_IDEAS_PIPELINE',brief)
    for attempt in range(3):
        try:
            stage([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_two_ideas.py','train',name],
                  'train_'+name,training=name)
            status=json.loads((OUT/name/'STATUS.json').read_text())
            if status['status']!='completed':raise RuntimeError('training did not pass completion/reload gate')
            return
        except BaseException:
            if attempt>=2 or not (OUT/name/'latest.pth').exists():raise
            write(R/f'ENGINEERING_RESUME_{name}_{attempt+1}.json',{'utc':now(),'reason':traceback.format_exc(),
                  'latest_sha256':sha(OUT/name/'latest.pth'),'attempt':attempt+1,'parameters_changed':False})


def prepare_external(result):
    try:
        stage([PY,'scripts/download_segresnet_two_ideas_pretrain.py'],'download_pretrain')
        stage([PY,'scripts/build_segresnet_two_ideas_pretrain.py'],'build_pretrain')
        result['status']='passed'
    except BaseException as e:
        result.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    write(R/'EXTERNAL_PREPARATION_STATUS.json',{'utc':now(),**result})


def pipeline():
    lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    if (R/'PIPELINE_START.json').exists():raise RuntimeError('pipeline already attempted; inspect before explicit recovery')
    assert torch.cuda.is_available() and torch.cuda.device_count()==2
    write(R/'PIPELINE_START.json',{'utc':now(),'pid':os.getpid(),'worker':1104883,'gpu_count':2,
          'freeze_sha256':sha(R/'FREEZE.json'),'order':['semantic_regions','pretrain','finetune_pretrained'],
          'test_tuning_used':False,'patient_pixels_access_at_start':False})
    stage([PY,'scripts/prepare_segresnet_two_ideas.py','audit'],'data_audit')
    for name in ['semantic_regions','pretrain']:
        stage([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train_segresnet_two_ideas.py','smoke',name],
              'smoke_'+name)
    stage([PY,'scripts/build_segresnet_two_ideas_classes.py'],'build_classes')
    stage([PY,'scripts/prepare_segresnet_two_ideas.py','gate'],'execution_gate')
    external={'status':'running'};thread=threading.Thread(target=prepare_external,args=(external,),daemon=False);thread.start()
    outcomes={}
    try:
        train('semantic_regions')
        stage([PY,'scripts/finalize_segresnet_two_ideas.py','semantic_regions'],'finalize_semantic_regions')
        outcomes['semantic_regions']='complete_test270'
    except BaseException as e:
        outcomes['semantic_regions']='failed';write(RECORDS['semantic_regions']/'PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
    while thread.is_alive():
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'waiting_external_cache','outcomes':outcomes,'worker':1104883});thread.join(20)
    if external['status']=='passed':
        try:
            train('pretrain');train('finetune_pretrained')
            stage([PY,'scripts/finalize_segresnet_two_ideas.py','finetune_pretrained'],'finalize_finetune_pretrained')
            outcomes['finetune_pretrained']='complete_test270'
        except BaseException as e:
            outcomes['finetune_pretrained']='failed';write(RECORDS['finetune_pretrained']/'PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
    else:
        outcomes['finetune_pretrained']='source_preparation_failed'
        write(RECORDS['finetune_pretrained']/'PIPELINE_FAILURE.json',{'utc':now(),**external,'training_started':False,'test_access':False})
    done=all(v=='complete_test270' for v in outcomes.values())
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete' if done else 'ended_with_failure','outcomes':outcomes,'worker':1104883,'running':False})
    body='2026-09-21两候选流水线已退出：'+str(outcomes)+'。无自动新增组或测试调优。[最终状态](archive/research_records/2026-09-21_segresnet_two_experiments/PIPELINE_STATUS.json)。'
    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:
        update_block(ROOT/filename,'TWO_IDEAS_PIPELINE',body)


def launch():
    verify()
    if (R/'LAUNCH.json').exists():raise RuntimeError('already launched')
    with (R/'pipeline.log').open('x') as f:
        p=subprocess.Popen([PY,'scripts/run_segresnet_two_ideas.py','pipeline'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    write(R/'LAUNCH.json',{'utc':now(),'pid':p.pid,'worker':1104883,'log':str((R/'pipeline.log').relative_to(ROOT))})
    print(json.dumps({'pid':p.pid,'worker':1104883,'launched':True}))


if __name__=='__main__':
    try:{'launch':launch,'pipeline':pipeline}[sys.argv[1]]()
    except BaseException as e:
        write(R/'PIPELINE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()})
        write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'failed','error':repr(e),'running':False});raise
