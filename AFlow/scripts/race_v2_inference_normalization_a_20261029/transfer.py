"""Exactly one selected normalization transfer after completed final training."""
import argparse,fcntl,time
import numpy as np
import torch
from scripts.race_v2_inference_normalization_a_20261029.common import *
from scripts.race_v2_inference_normalization_a_20261029.cache import generate
from scripts.race_v2_inference_normalization_a_20261029.deployment import deploy,gate_network
from scripts.race_v2_inference_a_20261029.snapshot108 import load_handoff
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks

OUT=R/'final_snapshot';FREEZE=R/'FINAL_SNAPSHOT_TRANSFER_FREEZE.json'
RECIPE=R/'NORMALIZATION_DEPLOYMENT_RECIPE.json'
TRAIN=ROOT/'archive/research_records/2026-10-29_race_v2_training'
JOB=STAGE/'SNAPSHOT108_QUEUE/race_v2_final_best108.json'
SNAPSHOT=BASE/'snapshots/race_v2_final_best108'

def freeze():
    assert not FREEZE.exists() and read(R/'DONE.json')['state']=='complete'
    sources=dict(read(R/'PREPARATION_FREEZE.json')['source_sha256'])
    for p in (ROOT/'scripts/race_v2_inference_normalization_a_20261029').glob('*.py'):sources[str(p.relative_to(ROOT))]=sha(p)
    sources.update(read(BASE/'SNAPSHOT108_FREEZE.json')['source_sha256'])
    write(FREEZE,dict(at=now(),source_sha256=sources,recipe_path=str(RECIPE),recipe_sha256=sha(RECIPE),
        result_sha256=sha(R/'RESULT.json'),recipes_transferred=1,maximum_new_forwards=756,
        existing_seed_from='same-weight completed ordinary8 snapshot view0 only',complete_training_required=True,
        zero_epoch_skip=True,new270_access=False))

def verify_transfer():
    guard();f=read(FREEZE)
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(RECIPE)==f['recipe_sha256'] and sha(R/'RESULT.json')==f['result_sha256']
    return f

def skip():
    marker=STAGE/'FINAL_HANDOFF_NO_NEW_MODEL.json'
    if not marker.exists():return False
    assert read(TRAIN/'DONE.json')['state']=='complete' and read(TRAIN/'HANDOFF.json')['actual_new_epochs']==0
    write(OUT/'SKIPPED.json',dict(at=now(),state='source_e164_remains_best_no_duplicate_forwards',
        training_done_sha256=sha(TRAIN/'DONE.json'),root_marker_sha256=sha(marker),
        new_validation_forwards_registered=0,new108_forwards=0,new270_forwards=0))
    return True

def execute():
    f=verify_transfer();job=read(JOB);h,network=load_handoff(job)
    source=read(SNAPSHOT/'ATOMS_READY.json');assert read(SNAPSHOT/'RESULT.json')['state']=='complete'
    assert source['checkpoint_sha256']==h['checkpoint_sha256'] and source['case_forwards']==864
    index=read(CACHE/'index.json');cohort=rows();assert index['case_ids']==[r['case_id'] for r in cohort]
    assert sha(CACHE/'index.json')==source['index_sha256']
    assert sha(CACHE/'images.f32')==read(BASE/'CACHE_READY.json')['payload_sha256']['images.f32']
    entry=read(RECIPE)['innovation_entry'];metadata=dict(model_id='race_v2_final_healthnorm',base_model_id='race_v2_final',
        checkpoint_path=h['checkpoint_path'],checkpoint_sha256=h['checkpoint_sha256'],architecture=h['architecture'],construction=h['construction'],
        input_recipe=entry,input_recipe_sha256=sha(RECIPE),atoms_ready_path=str(OUT/'ATOMS_READY.json'))
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*index['canvas']))
    lock=LOCK.open('a')
    while True:
        guard()
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(2)
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='0';torch.set_num_threads(2)
    network=network.cuda();write(OUT/'GPU_GATE.json',dict(at=now(),**gate_network(network,tuple(index['canvas']),4)))
    ready=generate(network,metadata,index,cohort,images,source['manifest_sha256'],source['index_sha256'],
        OUT,TMP/'final_snapshot',SNAPSHOT/'ATOMS_READY.json')
    del network;torch.cuda.empty_cache()
    assert sha(CACHE/'labels.u8')==read(BASE/'CACHE_READY.json')['payload_sha256']['labels.u8']
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*index['canvas']));lookup={(a['slot'],a['view']):a for a in ready['atoms']}
    baseline=read(SNAPSHOT/'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05'];records=[]
    for slot,row in enumerate(cohort):
        guard();g=index['geometry'][row['case_id']];zs=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];zs.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
        output=deploy(zs,np.asarray(images[slot][slices(g)]),entry);target=np.array(labels[slot][slices(g)]>0)
        records.append(record_mask(row,slot,output['mask'].cpu().numpy(),output['automatic_point'],baseline[slot]['automatic_point'],target,
            TMP/'final_snapshot/predictions'/f'{slot:03}.npz',output['diagnostic']))
        del zs,output
    raw=OUT/'RAW_VALIDATION108.json';write(raw,dict(at=now(),per_configuration={entry['id']:records},recipe=entry,
        checkpoint_sha256=h['checkpoint_sha256'],source_atoms_ready_sha256=sha(OUT/'ATOMS_READY.json'),
        new_network_forwards=ready['actual_new_forwards'],fixed_transfer=True))
    audit=OUT/'MASK_AUDIT.json';audit_masks(raw,audit,cohort,index,labels)
    write(OUT/'RESULT.json',dict(at=now(),state='complete',checkpoint_sha256=h['checkpoint_sha256'],recipe_id=entry['id'],
        metrics=summary_ci(records),paired_vs_same_model_equal8=paired(records,baseline),source_sha256=f['source_sha256'],
        source_atoms_sha256=sha(OUT/'ATOMS_READY.json'),raw_sha256=sha(raw),masks_audit_receipt_path=str(audit),
        masks_audit_receipt_sha256=sha(audit),deployment_recipe_sha256=sha(RECIPE),new_network_forwards=ready['actual_new_forwards'],
        logical_deployment_forwards=8,ordinary_reference_network_forwards_separately=864,
        new270_scoring=False,test_tuned_history=True,independent_confirmation=False))

def watch():
    verify_transfer()
    while True:
        guard()
        if skip():return
        if JOB.exists() and (SNAPSHOT/'RESULT.json').exists() and (SNAPSHOT/'ATOMS_READY.json').exists():execute();return
        time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        try:watch() if a.watch else execute()
        except BaseException as error:
            import traceback
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
