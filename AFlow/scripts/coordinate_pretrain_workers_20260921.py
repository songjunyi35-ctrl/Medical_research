"""Single-GPU source pretrain in parallel, then original two-GPU downstream finetune."""
import sys,os,json,time,shutil,subprocess,fcntl,traceback,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.pretrain_single_a100_20260921 import verify_single
from scripts.segresnet_two_ideas_common import *
from scripts import run_segresnet_two_ideas as legacy
RELAY=ROOT/'workspace/BraTS/cache/openneuro334_relay_20260921'
FILENAMES=['index.json','images.f32','labels.u8']


def expected():
    r=json.loads((R/'PRETRAIN_CACHE_RECEIPT.json').read_text())
    return dict(zip(FILENAMES,[r['index_sha256'],r['images_sha256'],r['labels_sha256']]))


def export_cache():
    verify_single();assert not RELAY.exists()
    assert shutil.disk_usage(ROOT).free>sum((PRETRAIN_CACHE/n).stat().st_size for n in FILENAMES)+2*(1<<30)
    RELAY.mkdir(parents=True)
    for n,h in expected().items():
        assert sha(PRETRAIN_CACHE/n)==h;shutil.copy2(PRETRAIN_CACHE/n,RELAY/n);assert sha(RELAY/n)==h
    write(R/'CACHE_EXPORT_1104883.json',{'utc':now(),'status':'passed','source_worker':1104883,'relay':str(RELAY),
          'sha256':expected(),'bytes':sum((RELAY/n).stat().st_size for n in FILENAMES),'hdfs_upload':False,'retained_for_recovery':True})


def import_cache():
    verify_single();assert json.loads((R/'CACHE_EXPORT_1104883.json').read_text())['status']=='passed'
    assert not PRETRAIN_CACHE.exists();assert shutil.disk_usage('/tmp').free>sum((RELAY/n).stat().st_size for n in FILENAMES)+2*(1<<30)
    PRETRAIN_CACHE.mkdir(parents=True)
    for n,h in expected().items():
        assert sha(RELAY/n)==h;shutil.copy2(RELAY/n,PRETRAIN_CACHE/n);assert sha(PRETRAIN_CACHE/n)==h
    write(R/'CACHE_TRANSFER_1105022.json',{'utc':now(),'status':'passed','source_worker':1104883,'destination_worker':1105022,
          'destination':str(PRETRAIN_CACHE),'destination_sha256':expected(),'source_receipt_sha256':sha(R/'PRETRAIN_CACHE_RECEIPT.json'),
          'raw_redownload':False,'reprocessing':False,'test_access':False})


def stop_old_queue():
    verify_single();state=json.loads((R/'EXTERNAL_REPAIR_STATUS.json').read_text())
    assert state['phase']=='waiting_for_existing_semantic_pipeline_to_exit'
    assert not (OUT/'pretrain').exists() and not (OUT/'finetune_pretrained').exists()
    pid=json.loads((R/'EXTERNAL_REPAIR_LAUNCH.json').read_text())['pid']
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'repair_segresnet_external_pretrain_queue.py run' in cmd
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('old queue did not stop; do not start another pretrain')
    write(R/'OLD_EXTERNAL_QUEUE_STOPPED.json',{'utc':now(),'stopped':True,'pid':pid,'previous_phase':state['phase'],
          'semantic_trainer_and_supervisor_untouched':True,'pretrain_had_not_started':True,'new_owner_worker':1105022,
          'reason':'user-provided working single A100; retain global8 and parallelize source pretraining'})
    write(R/'EXTERNAL_REPAIR_STATUS.json',{'utc':now(),'phase':'cache_complete_delegated_to_parallel_worker1105022',
          'worker':1104883,'cache_verified':True,'old_queue_stopped':True,'handoff_receipt':'OLD_EXTERNAL_QUEUE_STOPPED.json'})


def single_stage(mode):
    env={**legacy.ENV,'CUDA_VISIBLE_DEVICES':'0'}
    with (R/f'single_pretrain_{mode}.log').open('a') as log:
        proc=subprocess.Popen([legacy.PY,'scripts/pretrain_single_a100_20260921.py',mode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            progress=json.loads((OUT/'pretrain/STATUS.json').read_text()) if (OUT/'pretrain/STATUS.json').exists() else None
            write(R/'SINGLE_PRETRAIN_STATUS.json',{'utc':now(),'status':'running_'+mode,'worker':1105022,'pid':proc.pid,'progress':progress})
            time.sleep(20)
    if proc.returncode:raise RuntimeError(f'single pretrain {mode} exit {proc.returncode}')


def single_pipeline():
    verify_single();lock=(R/'pretrain_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((R/'OLD_EXTERNAL_QUEUE_STOPPED.json').read_text())['stopped']
    if (R/'SINGLE_PRETRAIN_START.json').exists():raise RuntimeError('already started')
    write(R/'SINGLE_PRETRAIN_START.json',{'utc':now(),'worker':1105022,'hardware_amendment_sha256':sha(R/'HARDWARE_AMENDMENT_003.json')})
    single_stage('smoke')
    for attempt in range(3):
        try:single_stage('train');break
        except BaseException:
            if attempt==2 or not (OUT/'pretrain/latest.pth').exists():raise
            write(R/f'SINGLE_PRETRAIN_RESUME_{attempt+1}.json',{'utc':now(),'reason':traceback.format_exc(),
                  'latest_sha256':sha(OUT/'pretrain/latest.pth'),'attempt':attempt+1,'same_single_gpu_recipe':True})
    status=json.loads((OUT/'pretrain/STATUS.json').read_text());assert status['status']=='completed' and status['last_complete_epoch']==150
    write(R/'SINGLE_PRETRAIN_DONE.json',{'utc':now(),'worker':1105022,'checkpoint_sha256':sha(OUT/'pretrain/final.pth'),
          'completed_epochs':150,'updates':6300,'elapsed_seconds':status['elapsed_seconds']})
    write(R/'SINGLE_PRETRAIN_STATUS.json',{'utc':now(),'status':'complete','worker':1105022,'progress':status})


def dual_queue():
    verify_single();queue_lock=(R/'finetune_owner.lock').open('a');fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(R/'pipeline.lock').open('a');locked=False
    while True:
        if (R/'SINGLE_PIPELINE_FAILURE.json').exists():raise RuntimeError('source pretraining failed; no downstream training')
        if not locked:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
            except BlockingIOError:pass
        ready=(R/'SINGLE_PRETRAIN_DONE.json').exists()
        if locked and ready:break
        write(R/'DUAL_FINETUNE_QUEUE_STATUS.json',{'utc':now(),'status':'waiting','worker':1104883,
              'semantic_pipeline_released_gpus':locked,'source_pretraining_complete':ready})
        time.sleep(20)
    verify_single();assert not (OUT/'finetune_pretrained').exists()
    done=json.loads((R/'SINGLE_PRETRAIN_DONE.json').read_text());assert sha(OUT/'pretrain/final.pth')==done['checkpoint_sha256']
    rec=RECORDS['finetune_pretrained']
    if (rec/'PIPELINE_FAILURE.json').exists():(rec/'PIPELINE_FAILURE.json').rename(rec/'PIPELINE_FAILURE_old_sequential_queue.json')
    write(rec/'PRETRAIN_HARDWARE_RECEIPT.json',{'utc':now(),'pretrain_worker':1105022,'pretrain_gpus':1,'pretrain_global_batch':8,
          'finetune_worker':1104883,'finetune_gpus':2,'finetune_global_batch':8,'stochastic_trajectory_change_disclosed':True,
          'hardware_amendment_sha256':sha(R/'HARDWARE_AMENDMENT_003.json'),'pretrain_final_sha256':done['checkpoint_sha256']})
    write(R/'DUAL_FINETUNE_QUEUE_STATUS.json',{'utc':now(),'status':'training','worker':1104883})
    legacy.train('finetune_pretrained')
    legacy.stage([legacy.PY,'scripts/finalize_segresnet_two_ideas.py','finetune_pretrained'],'finalize_finetune_pretrained')
    report=rec/'RESULT.md';report.write_text(report.read_text()+'\n硬件执行补记：公开预训练在1105022单A100/global8完成，微调在1104883双A100/global8完成；单卡预训练随机数消耗顺序与原双卡草案不同，不声称逐位同轨迹。[执行收据](PRETRAIN_HARDWARE_RECEIPT.json)。\n')
    outcomes={'semantic_regions':'complete_test270' if (RECORDS['semantic_regions']/'DONE.json').exists() else 'see_semantic_failure',
              'finetune_pretrained':'complete_test270'}
    write(R/'PIPELINE_STATUS.json',{'utc':now(),'status':'complete' if outcomes['semantic_regions']=='complete_test270' else 'ended_with_failure',
          'outcomes':outcomes,'running':False,'workers':[1104883,1105022]})
    write(R/'DUAL_FINETUNE_QUEUE_STATUS.json',{'utc':now(),'status':'complete','worker':1104883})
    body='两候选已结束：'+str(outcomes)+'。预训练单卡与语义组并行，微调双卡；未使用test定向训练。[最终状态](archive/research_records/2026-09-21_segresnet_two_experiments/PIPELINE_STATUS.json)。'
    for filename in ['AFLOW_BRATS_HANDOFF_CURRENT.md','AFLOW_BRATS_WEEKLY_REPORT.md']:legacy.update_block(ROOT/filename,'TWO_IDEAS_PIPELINE',body)


def launch(which):
    verify_single();receipt=R/f'{which.upper()}_LAUNCH.json';assert not receipt.exists()
    env={**legacy.ENV,'CUDA_VISIBLE_DEVICES':'0' if which=='single' else '0,1'}
    with (R/f'{which}_coordinator.log').open('x') as f:
        proc=subprocess.Popen([legacy.PY,'scripts/coordinate_pretrain_workers_20260921.py',which],cwd=ROOT,env=env,
              stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(receipt,{'utc':now(),'pid':proc.pid,'worker':1105022 if which=='single' else 1104883});print(json.dumps({'launched':which,'pid':proc.pid}))


if __name__=='__main__':
    try:
        mode=sys.argv[1]
        if mode=='launch':launch(sys.argv[2])
        else:{'export':export_cache,'import':import_cache,'stop-old':stop_old_queue,'single':single_pipeline,'dual':dual_queue}[mode]()
    except BaseException as e:
        name='SINGLE_PIPELINE_FAILURE.json' if sys.argv[1]=='single' else 'MULTIWORKER_'+sys.argv[1]+'_FAILURE.json'
        write(R/name,{'utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
