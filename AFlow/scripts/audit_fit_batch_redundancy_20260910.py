"""Metadata-only replay of already-completed native/subject sampler minibatches."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts import structured_background_protocol as p
from scripts.audit_subject_sampler_replay import independent_weights


def main():
    record=p.ROOT/'archive/reviews/2026-09-10_data_processing_priority'
    p.check_freeze();fit,_=p.check_roles();p.checked(p.NATIVE,p.NATIVE_SHA)
    ids=[r['case_id'] for r in fit];subject_map={r['case_id']:r['subject_id'] for r in fit}
    slot={s:i for i,s in enumerate(sorted(set(subject_map.values())))}
    subject_ids=np.array([slot[subject_map[c]] for c in ids])
    volumes=json.loads(p.NATIVE.read_text())['native_wt_voxels'];weights=independent_weights(ids,subject_map,volumes)
    result={};source_hashes={}
    for policy,w in weights.items():
        unique_subjects=[];unique_cases=[];maxima=[]
        for seed in p.SEEDS:
            path=p.WORK/f'seed{seed}_control/TRAIN_DONE.json' if policy=='control' else p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed{seed}_subject_balanced/TRAIN_DONE.json'
            old=json.loads(path.read_text());source_hashes[str(path.relative_to(p.ROOT))]=p.sha(path)
            assert hashlib.sha256(w.numpy().tobytes()).hexdigest()==old['sampling_weight_sha256']
            generator=torch.Generator().manual_seed(seed)
            for epoch in range(168):
                draw=torch.multinomial(w,972,replacement=True,generator=generator).numpy()[:968]
                assert hashlib.sha256(draw.astype(np.int64).tobytes()).hexdigest()==old['history'][epoch]['sampler_sha256']
                batches=draw.reshape(121,8);subjects=subject_ids[batches]
                unique_subjects.extend((1+(np.diff(np.sort(subjects,axis=1),axis=1)!=0).sum(1)).tolist())
                unique_cases.extend((1+(np.diff(np.sort(batches,axis=1),axis=1)!=0).sum(1)).tolist())
                maxima.append(int(np.bincount(subject_ids[draw],minlength=313).max()))
        result[policy]={'batches':len(unique_subjects),'batches_with_repeated_subject':sum(x<8 for x in unique_subjects),'repeated_subject_fraction':float(np.mean(np.array(unique_subjects)<8)),'mean_unique_subjects_per_batch':float(np.mean(unique_subjects)),'redundant_patient_slots_fraction':float((8-np.mean(unique_subjects))/8),'batches_with_repeated_scan':sum(x<8 for x in unique_cases),'maximum_single_subject_draws_in_epoch':max(maxima)}
    p.write_new(record/'FIT_BATCH_REDUNDANCY.json',{'status':'completed_metadata_audit','created_utc':datetime.now(timezone.utc).isoformat(),'fit_sha256':p.FIT_SHA,'native_sha256':p.NATIVE_SHA,'source_training_hashes':source_hashes,'all672_epoch_index_hashes_exact':True,'results':result,'script_sha256':p.sha(Path(__file__)),'access':{'fit_metadata':True,'images_labels':False,'checkpoint_load':False,'new_development_evaluation':False,'test':False},'new_forwards':0,'decision':'batch_reordering_not_implemented_or_added_to_current_training'})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
