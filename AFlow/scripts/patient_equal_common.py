"""One single-GPU full972 patient-weighted candidate; no test pixels."""
import json,os,collections
from pathlib import Path
import torch
from scripts.segresnet_plain972_common import ROOT,TRAIN_CACHE,VAL_CACHE,model_for,seed_all,safe_augment,dice_per_case,validate,summary_ci,sha,write,now,save,disk,BrainCropDataset,learning_rate_for_epoch
from scripts.segresnet_two_ideas_common import paired
from sammed3d.patient_equal_loss import PatientEqualDiceCE,patient_weights
R=ROOT/'archive/research_records/2026-09-21_segresnet_patient_equal_loss'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/patient_equal_loss_20260921'
SOURCE=ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/pretrain/final.pth'
SOURCE_SHA='3726eac66918b7afd32ae93500a554534d27528bc86a0fe9dd2a626ce5ac6b76'
BASELINE_REPORT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/plain972_baseline_20260917/validation_e193.json'
PRETRAINED_REPORT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/two_ideas_20260921/finetune_pretrained/validation_e179.json'
SEED=20260819
MANIFEST={k:ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1'/v for k,v in [('train','train_t1ce_wt.jsonl'),('validation','validation_t1ce_wt.jsonl'),('test','sealed_test/test_cases_t1ce.jsonl')]}


def read_rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def verify():
    d=json.loads((R/'FREEZE.json').read_text())
    for p,h in d['files'].items():assert sha(ROOT/p)==h,p
    return d
def batches(epoch):
    order=torch.randperm(972,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
    return [order[i:i+8] for i in range(0,972,8)]
def lr(epoch):return learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200)


class WeightedDataset(BrainCropDataset):
    def __init__(self,ids,weights):
        super().__init__(TRAIN_CACHE,expected_split='train',case_ids=ids);self.weights=weights
        assert set(self.case_ids)==set(weights)
    def __getitem__(self,i):
        d=super().__getitem__(i);d['patient_weight']=float(self.weights[d['case_id']]);return d
