"""One DDP training run or full-recipe synthetic gate, with absolute budget and bounded cleanup."""
import argparse
import datetime
import hashlib
import pickle
import random
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *
os.environ.update(env())
import numpy as np
import torch
import torch.distributed as dist
from scripts.nnunet_resenc_20261013.trainer import AFlowTrainer,restore_margin
CURRENT_TRAINER=None

def model_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def committed_exit(marker,rank):
    dist.barrier(device_ids=[rank])
    if not Path(marker).is_file():raise RuntimeError('no success receipt')
    def force():
        write(R/f'CLEANUP_{Path(marker).stem}_rank{rank}.json',{'at':now(),'post_success_timeout':True})
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    timer=threading.Timer(20,force);timer.daemon=True;timer.start()
    dist.destroy_process_group();timer.cancel()
    sys.stdout.flush();sys.stderr.flush();os._exit(0)

def synthetic_dataset(plan,dataset):
    folder=TMP/'synthetic/data';folder.mkdir(parents=True,exist_ok=True)
    shape=tuple(plan['configurations']['3d_fullres']['patch_size'])
    for i in range(8):
        rng=np.random.default_rng(100+i)
        data=rng.normal(size=(1,*shape)).astype(np.float32)
        seg=np.zeros((1,*shape),dtype=np.int8)
        box=tuple(slice(s//3,s*2//3) for s in shape);seg[(0,*box)]=1
        locations=np.argwhere(seg==1)[::100].astype(np.int64)
        np.save(folder/f'synthetic{i}.npy',data);np.save(folder/f'synthetic{i}_seg.npy',seg)
        np.savez(folder/f'synthetic{i}.npz',placeholder=np.zeros(1))
        with (folder/f'synthetic{i}.pkl').open('wb') as f:pickle.dump({'class_locations':{1:locations},'spacing':[1,1,1]},f)
    write(TMP/'synthetic/dataset_fingerprint.json',{'synthetic':True})

def check_geometry(trainer):
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    shape=(32,40,48);box=[[1,31],[2,38],[3,45]];cropped=(30,36,42)
    props={'spacing':[1,1,1],'shape_before_cropping':shape,'shape_after_cropping_and_before_resampling':cropped,'bbox_used_for_cropping':box}
    logits=torch.randn((2,*cropped))
    official=convert_predicted_logits_to_segmentation_with_correct_shape(logits,trainer.plans_manager,
        trainer.configuration_manager,trainer.label_manager,props,False,num_threads_torch=1)
    margin=restore_margin(logits,trainer.plans_manager,trainer.configuration_manager,props)
    if not np.array_equal(margin>0,official.transpose(2,1,0)>0):raise RuntimeError('native restore synthetic parity')

def main(mode):
    global CURRENT_TRAINER
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15))
    random.seed(20261013);np.random.seed(20261013);torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    smoke=mode=='smoke'
    if rank==0:
        write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'full_recipe_synthetic_gate' if smoke else 'training_initialization',
            'running':True,'training_started':False,'new_test_access':False})
    if smoke:os.environ['AFLOW_RESENC_SMOKE']='1'
    else:os.environ.pop('AFLOW_RESENC_SMOKE',None)
    frozen=verify_source_freeze('SMOKE_FREEZE.json' if smoke else 'EXECUTION_FREEZE.json')
    plan=read(R/'TRAIN972_PLANS.json');dataset=read(PRE/'dataset.json')
    if smoke and rank==0:
        folder=PRE/plan['configurations']['3d_fullres']['data_identifier']
        files=sorted(p for p in folder.iterdir() if p.is_file())
        write(R/'PREPROCESSED_SHA256.json',{'at':now(),'files':{str(p):sha(p) for p in files},
            'roles':['train','validation'],'test_access':False})
        synthetic_dataset(plan,dataset)
    dist.barrier(device_ids=[rank])
    trainer=AFlowTrainer(plan,'3d_fullres',0,dataset,torch.device('cuda',rank))
    CURRENT_TRAINER=trainer
    trainer.initialize();check_geometry(trainer)
    if not smoke:
        ready=read(R/'DATA_READY.json');assert ready['status']=='passed'
        if rank==0:
            receipt=read(R/'PREPROCESSED_SHA256.json')
            for path,digest in receipt['files'].items():
                if sha(path)!=digest:raise RuntimeError('preprocessed cache changed: '+path)
            marker=R/'CACHE_MODEL_PREACCESS.json'
            if not marker.exists():
                write(marker,{'at':now(),'preprocessed_manifest_sha256':sha(R/'PREPROCESSED_SHA256.json'),
                    'native_audit_sha256':sha(R/'NATIVE_DATA_AUDIT.json'),'before_patient_model_forward':True})
            elif read(marker)['preprocessed_manifest_sha256']!=sha(R/'PREPROCESSED_SHA256.json'):
                raise RuntimeError('preprocessed source receipt changed')
        dist.barrier(device_ids=[rank])
        if sha(R/'INSTALLED_SOURCE_SHA256.json')!=frozen['installed_source_manifest_sha256']:raise RuntimeError('installed source manifest changed')
        for path,digest in read(R/'INSTALLED_SOURCE_SHA256.json').items():
            if sha(path)!=digest:raise RuntimeError('installed source changed: '+path)
        if sha(PRE/(PLANS+'.json'))!=frozen['plan_sha256'] or sha(PRE/'splits_final.json')!=frozen['split_sha256']:
            raise RuntimeError('plan or split changed')
        latest=Path(trainer.output_folder)/'checkpoint_latest.pth'
        if latest.exists():
            if os.environ.get('AFLOW_ENGINEERING_RESUME')!='1':raise RuntimeError('checkpoint exists; no implicit resume')
            trainer.load_checkpoint(str(latest))
        elif (R/'TRAIN_PREACCESS.json').exists():raise RuntimeError('prior training access without checkpoint')
        if rank==0 and not (R/'TRAIN_PREACCESS.json').exists():
            write(R/'TRAIN_PREACCESS.json',{'at':now(),'worker':1132660,'train_scans':972,'validation_scans':108,
               'test_pixels_access':False,'source_freeze_sha256':sha(R/'EXECUTION_FREEZE.json')})
        dist.barrier(device_ids=[rank])
        if rank==0:
            write(R/f'TRAIN_START_pid{os.getpid()}.json',{'at':now(),'pid':os.getpid(),'resume_epoch':trainer.current_epoch,
                'retained_optimizer_steps_at_start':trainer.actual_optimizer_steps,'attempted_updates_at_start':trainer.attempted_updates})
    # Official augmentation uses per-worker streams; no bitwise-resume claim.
    np.random.seed(20261013+rank);random.seed(20261013+rank)
    trainer.on_train_start()
    torch.cuda.reset_peak_memory_stats(rank)
    if smoke:
        trainer.on_train_epoch_start();timings=[];committed=[]
        for i in range(45):
            torch.cuda.synchronize(rank);tick=time.monotonic();before=trainer.actual_optimizer_steps
            trainer.train_step(next(trainer.dataloader_train));torch.cuda.synchronize(rank)
            if i>=10:timings.append(time.monotonic()-tick);committed.append(trainer.actual_optimizer_steps-before)
        if not all(x==1 for x in committed):raise RuntimeError('synthetic post-warmup optimizer skip')
        torch.cuda.synchronize(rank)
        digest=model_hash(trainer.network.module);all_hash=[None,None];dist.all_gather_object(all_hash,digest)
        if len(set(all_hash))!=1:raise RuntimeError('DDP full state SHA differs')
        path=Path(trainer.output_folder)/'checkpoint_smoke.pth';trainer.save_checkpoint(str(path));dist.barrier(device_ids=[rank])
        trainer.load_checkpoint(str(path))
        if model_hash(trainer.network.module)!=digest:raise RuntimeError('checkpoint reload differs')
        peak=torch.cuda.max_memory_reserved(rank)/2**30
        if peak>38:raise RuntimeError('reserved peak exceeds38GiB')
        local={'rank':rank,'patch_size':trainer.configuration_manager.patch_size,'batch_per_rank':trainer.batch_size,
          'attempted_updates':trainer.attempted_updates,'optimizer_steps':trainer.actual_optimizer_steps,
          'median_step_seconds':float(np.median(timings)),'mean_step_seconds':float(np.mean(timings)),
          'p90_step_seconds':float(np.quantile(timings,.9)),'peak_reserved_gib':peak,
          'state_sha256':digest,'strict_reload_exact':True,'geometry_parity':True,
          'full_official_augmentation_loss_optimizer':True}
        reports=[None,None];dist.all_gather_object(reports,local)
        if rank==0:write(R/'GPU_SMOKE.json',{'at':now(),'status':'passed','ranks':reports,'patient_model_forwards':0})
        trainer.stop_loaders();committed_exit(R/'GPU_SMOKE.json',rank)
    start=time.monotonic();train_access=read(R/'TRAIN_PREACCESS.json')
    absolute=datetime.datetime.fromisoformat(frozen['training_deadline']).timestamp()
    origin=datetime.datetime.fromisoformat(train_access['at']).timestamp()
    deadline=min(absolute,origin+20*3600)
    stopped=False
    while trainer.current_epoch<trainer.num_epochs:
        trainer.on_epoch_start();trainer.on_train_epoch_start();outputs=[];stop_after_epoch=False
        for step in range(trainer.num_iterations_per_epoch):
            outputs.append(trainer.train_step(next(trainer.dataloader_train)))
            if rank==0 and step%25==0:
                write(R/'HEARTBEAT.json',{'at':now(),'worker':1132660,'pid':os.getpid(),'epoch_in_progress':trainer.current_epoch+1,
                    'iteration':step+1,'last_complete_epoch':trainer.current_epoch,'actual_optimizer_steps':trainer.actual_optimizer_steps,
                    'attempted_updates':trainer.attempted_updates,'test_pixels_access':False})
                if trainer.attempted_updates==1:
                    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'training','running':True,'training_started':True,
                        'last_complete_epoch':0,'target_epochs':trainer.num_epochs,'actual_optimizer_steps':trainer.actual_optimizer_steps,'new_test_access':False})
        trainer.on_train_epoch_end(outputs)
        trainer.on_validation_epoch_start()
        with torch.no_grad():vals=[trainer.validation_step(next(trainer.dataloader_val)) for _ in range(trainer.num_val_iterations_per_epoch)]
        trainer.on_validation_epoch_end(vals)
        flag=torch.tensor([time.time()>=deadline-600],device=rank,dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        stop_after_epoch=bool(flag.item())
        trainer.finish_epoch(force_validation=stop_after_epoch)
        if stop_after_epoch:stopped=True;break
    final=read(OUT/'STATUS.json') if rank==0 else None
    if rank==0:
        final.update(at=now(),state='training_complete' if not stopped else 'budget_stopped',running=False,
           stopped_for_wall_budget=stopped,elapsed_this_attempt_seconds=time.monotonic()-start,
           wall_seconds_since_first_training_preaccess=time.time()-origin,
           update_count_scope='retained checkpoint trajectory; discarded work in per-attempt records')
        write(OUT/'STATUS.json',final);write(R/'TRAINING_DONE.json',final)
    trainer.stop_loaders();committed_exit(R/'TRAINING_DONE.json',rank)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    try:main(a.mode)
    except BaseException as e:
        data={'at':now(),'error':repr(e),'pid':os.getpid()}
        if CURRENT_TRAINER is not None:
            data.update(epoch_in_progress=CURRENT_TRAINER.current_epoch+1,actual_optimizer_steps=CURRENT_TRAINER.actual_optimizer_steps,
                attempted_updates=CURRENT_TRAINER.attempted_updates)
        write(R/f'{a.mode.upper()}_FAILURE_rank{os.environ.get("LOCAL_RANK","unknown")}.json',data)
        write(R/f'{a.mode.upper()}_FAILURE_pid{os.getpid()}.json',data)
        raise
