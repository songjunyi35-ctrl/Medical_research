"""One frozen innovation transfer after final new-best108 atoms, never16 again."""
import argparse,fcntl,time
import numpy as np
import torch
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_innovation_a_20261029.deployment import deploy
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks
from scripts.race_v2_inference_innovation_a_20261029.algorithm import equal8
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

FREEZE=R/'FINAL_SNAPSHOT_TRANSFER_FREEZE.json';OUT=R/'final_snapshot'
TRAIN=ROOT/'archive/research_records/2026-10-29_race_v2_training'
REGISTRATION=STAGE/'SNAPSHOT108_QUEUE/race_v2_final_best108.json'
SNAPSHOT=BASE/'snapshots/race_v2_final_best108'

def freeze():
    assert not FREEZE.exists();assert read(R/'DONE.json')['state']=='complete'
    paths=list((ROOT/'scripts/race_v2_inference_innovation_a_20261029').glob('*.py'))
    paths+=[ROOT/'scripts/race_v2_inference_a_20261029/statistics.py']
    write(FREEZE,dict(at=now(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        deployment_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),initial_result_sha256=sha(R/'RESULT.json'),
        original_execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),new_recipes=1,new_neural_forwards=0,
        expected_registration_path=str(REGISTRATION),only_completed_training_best=True,epoch0_skip=True,new270_access=False))

def verify_transfer():
    guard();f=read(FREEZE)
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(R/'DEPLOYMENT_RECIPE.json')==f['deployment_recipe_sha256']
    assert sha(R/'RESULT.json')==f['initial_result_sha256']
    return f

def skip_if_epoch0():
    marker=STAGE/'FINAL_HANDOFF_NO_NEW_MODEL.json'
    if not marker.exists():return False
    assert read(TRAIN/'DONE.json')['state']=='complete' and read(TRAIN/'HANDOFF.json')['actual_new_epochs']==0
    write(OUT/'SKIPPED.json',dict(at=now(),state='source_e164_remains_best_no_duplicate_forwards',
        root_marker_sha256=sha(marker),training_done_sha256=sha(TRAIN/'DONE.json'),new108_forwards=0,new270_forwards=0,new_decoding_search=False))
    return True

def execute():
    f=verify_transfer();registration=read(REGISTRATION)
    assert registration['registered_by']=='root' and registration['approval_scope']=='final_completed_training_snapshot108_only'
    h=read(registration['handoff_path']);assert sha(registration['handoff_path'])==registration['handoff_sha256']
    done=read(registration['training_done_path']);assert sha(registration['training_done_path'])==registration['training_done_sha256']
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs'] and h['actual_new_epochs']>0
    ready=read(SNAPSHOT/'ATOMS_READY.json');snapshot_result=read(SNAPSHOT/'RESULT.json')
    assert snapshot_result['state']=='complete' and ready['case_forwards']==864
    assert ready['checkpoint_sha256']==snapshot_result['checkpoint_sha256']==h['checkpoint_sha256']
    index=read(CACHE/'index.json');assert sha(CACHE/'index.json')==ready['index_sha256']
    cohort=rows();assert index['case_ids']==[r['case_id'] for r in cohort]
    lookup={(a['slot'],a['view']):a for a in ready['atoms']};assert len(lookup)==864
    entry=read(R/'DEPLOYMENT_RECIPE.json')['innovation_entry'];identifier=entry['id']
    reference=read(SNAPSHOT/'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05']
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*index['canvas']));labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*index['canvas']))
    lock_path=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock');lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    write(OUT/'PREACCESS.json',dict(at=now(),registration_sha256=sha(REGISTRATION),source_atoms_sha256=sha(SNAPSHOT/'ATOMS_READY.json'),
        frozen_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),new_recipes=1,new_network_forwards=0,new270_scoring=False))
    records=[];baseline=[];started=time.monotonic();gpu_seconds=0.
    for slot,row in enumerate(cohort):
        verify_transfer();sl=slices(index['geometry'][row['case_id']]);logits=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];logits.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
        output=deploy(logits,np.array(images[slot][sl],np.float32),entry);last.record();last.synchronize();gpu_seconds+=first.elapsed_time(last)/1000
        target=np.array(labels[slot][sl]>0);base_point=output['base_automatic_point'];base_mask=(equal8(logits)>0).cpu().numpy()
        tp=int((base_mask&target).sum());pred=int(base_mask.sum());positive=int(target.sum());den=pred+positive
        base=dict(case_id=row['case_id'],subject_id=row['subject_id'],tp=tp,fp=pred-tp,fn=positive-tp,locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,automatic_point=base_point,automatic_point_hits_gt=bool(target[tuple(base_point)]))
        assert all(base[k]==reference[slot][k] for k in base);baseline.append(base)
        record=record_mask(row,slot,output['mask'].cpu().numpy(),output['automatic_point'],base_point,target,
            TMP/'final_snapshot/predictions'/(row['case_id']+'.npz'),output['diagnostic']);records.append(record)
        write(OUT/'HEARTBEAT.json',dict(at=now(),completed=slot+1,total=108,new_recipes=1,new_network_forwards=0))
        del logits,output
    raw_path=OUT/'RAW_VALIDATION108.json';write(raw_path,dict(at=now(),per_configuration={identifier:records},recipe=entry,new_network_forwards=0,fixed_transfer=True))
    receipt_path=OUT/'MASK_AUDIT.json';audit_masks(raw_path,receipt_path,cohort,index,labels)
    metric=summary_ci(records);write(OUT/'RESULT.json',dict(at=now(),state='complete',checkpoint_sha256=h['checkpoint_sha256'],
        recipe_id=identifier,metrics=metric,Point_refined=metric['point_hits'],Point_base=sum(r['base_automatic_point_hits_gt'] for r in records),
        paired_vs_same_model_equal8=paired(records,baseline),source_sha256=f['source_sha256'],source_freeze_sha256=sha(FREEZE),
        deployment_recipe_sha256=sha(R/'DEPLOYMENT_RECIPE.json'),source_atoms_sha256=sha(SNAPSHOT/'ATOMS_READY.json'),
        raw_sha256=sha(raw_path),masks_audit_receipt_path=str(receipt_path),masks_audit_receipt_sha256=sha(receipt_path),
        gpu_algorithm_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        new_network_forwards=0,new_recipes=1,new270_scoring=False,test_tuned_history=True,independent_confirmation=False))

def watch():
    verify_transfer()
    while True:
        guard()
        if skip_if_epoch0():return
        if REGISTRATION.exists() and (SNAPSHOT/'RESULT.json').exists() and (SNAPSHOT/'ATOMS_READY.json').exists():execute();return
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:watch() if a.watch else execute()
        except BaseException as error:
            import traceback
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_network_forwards=0));raise
