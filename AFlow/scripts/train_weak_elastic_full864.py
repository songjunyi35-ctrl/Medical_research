"""Conditional same-engine full864 pair, no development loader or warm start."""
import json
import sys
import torch
from scripts import weak_elastic_full864_protocol as p
from scripts import train_structured_background_aux as training
from scripts.train_t1ce_wt_capg3d import BrainCropDataset,augment_batch as base_augment,write_json_atomic
from scripts.build_t1ce_wt_native_volume_artifact import native_sampling_weights
from sammed3d.subject_balanced_sampling import normalize_subject_weights
from sammed3d.weak_elastic import WeakElastic


class FullData(BrainCropDataset):
    def __init__(self,fit):
        super().__init__(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in fit])
    def __getitem__(self,slot):
        item=super().__getitem__(slot)
        item['label']=torch.cat((item['label'],torch.zeros_like(item['label'])))
        return item


def full_weights(case_ids,volumes,policy):
    weights=native_sampling_weights(case_ids,volumes)
    if policy=='subject_elastic':
        weights=normalize_subject_weights(case_ids,weights,{r['case_id']:r['subject_id'] for r in p.rows(p.FIT)})
    elif policy!='native_elastic':raise ValueError(policy)
    return weights


def main():
    frozen=p.check_freeze()
    seed=int(sys.argv[sys.argv.index('--seed')+1]);arm=sys.argv[sys.argv.index('--arm')+1]
    if seed not in p.SEEDS or arm not in p.ARMS:raise ValueError('unregistered full864 run')
    deform=WeakElastic(seed)
    def augment(x,y,max_shift):
        x,y=base_augment(x,y,max_shift=max_shift)
        if arm=='full_elastic':
            x,y=deform(x,y)
            if deform.steps%121==0:write_json_atomic(p.WORK/f'seed{seed}_{arm}/ELASTIC_PROGRESS.json',deform.receipt())
        return x,y
    training.p=p;training.FitData=FullData;training.augment_batch=augment
    training.native_sampling_weights=lambda ids,volumes:full_weights(ids,volumes,frozen['selected_arm'])
    try:training.main()
    finally:
        p.write_new(p.WORK/f'seed{seed}_{arm}/FULL864_ROLE_RECEIPT.json',{'seed':seed,'arm':arm,'selected_sampling_policy':frozen['selected_arm'],'fit864_sha256':p.FIT_SHA,'development180_consumed_as_training':True,'validation_or_test_metrics_accessed':False,'auxiliary_coefficient':0,'auxiliary_targets':'zero_placeholder_not_used_in_primary_gradient','deformation':deform.receipt()})


if __name__=='__main__':main()
