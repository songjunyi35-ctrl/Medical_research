#!/usr/bin/env python3
"""One-time Africa-informed frozen checkpoint confirmation on CPTAC33."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Dataset
import scripts.evaluate_brats_africa95_frozen_baselines as metrics
import scripts.evaluate_ucsd_ptgbm_external_confirmation as model_io
from scripts import cptac33_protocol as p
metrics.p=p
class Data(Dataset):
 def __init__(self):
  r=json.load(open(p.CACHE_RECEIPT));p.checked(p.CACHE/'index.json',r['index_sha256']);self.index=json.load(open(p.CACHE/'index.json'))
  if not r['all_gates_passed'] or self.index['cases']!=33:raise ValueError('cache')
  p.checked(p.CACHE/'images.f32',r['images_sha256']);p.checked(p.CACHE/'labels.u8',r['labels_sha256']);self.i=np.memmap(p.CACHE/'images.f32',dtype=np.float32,mode='r',shape=(33,128,128,128));self.l=np.memmap(p.CACHE/'labels.u8',dtype=np.uint8,mode='r',shape=(33,128,128,128))
 def __len__(self):return 33
 def __getitem__(self,x):return {'image':torch.from_numpy(np.array(self.i[x]))[None],'label':torch.from_numpy(np.array(self.l[x]))[None],'case_id':self.index['case_ids'][x],'subject_id':self.index['subject_ids'][x]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args();p.docs();pre=json.load(open(p.MODEL_PREACCESS));cache=json.load(open(p.CACHE_RECEIPT))
 if pre.get('status')!='frozen_after_cache_before_model' or pre['access']['model_inference'] is not False or pre['evaluator_sha256']!=p.sha(Path(__file__)) or pre['cache']['receipt_sha256']!=p.sha(p.CACHE_RECEIPT):raise ValueError('preaccess')
 out=p.RECORD/'RAW_CPTAC33.json';started=p.RECORD/'RAW_CPTAC33_STARTED.json'
 if out.exists() or started.exists():raise FileExistsError('repeat')
 device=torch.device(a.device)
 if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA')
 torch.cuda.set_device(device);torch.set_num_threads(1);loader=DataLoader(Data(),batch_size=8,shuffle=False,num_workers=0);full,fp=model_io.load_model(p.FULL,p.FULL_SHA,'primary',device);gc,gp=model_io.load_model(p.GCSAM,p.GCSAM_SHA,'secondary',device);p.write_new(started,{'schema':'aflow.cptac33.started','created_at_utc':datetime.now(timezone.utc).isoformat(),'preaccess_sha256':p.sha(p.MODEL_PREACCESS),'external_access':True})
 torch.cuda.reset_peak_memory_stats(device);begin=time.monotonic();fs,nf=metrics.evaluate_full(full,loader,device);del full;torch.cuda.empty_cache();gs,ng=metrics.evaluate_identity(gc,loader,device);forwards=nf+ng
 if forwards!=297:raise RuntimeError('forwards')
 primary=metrics.compare(fs['identity'],gs,p.BOOTSTRAP_SEED+200,'gcsam_minus_full');secondary=metrics.compare(fs['identity'],fs['flip8'],p.BOOTSTRAP_SEED+100,'flip8_minus_identity');delta=primary['gcsam_minus_full']['scan_dice'];lo=primary['paired_scan_dice_95ci'][0];interpretation='strong_directional_support' if delta>0 and lo>0 else ('weak_uncertain_support' if delta>0 else 'not_replicated')
 report={'schema':'aflow.cptac33.africa_informed_confirmation','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'evidence_scope':'BraTS2021 public Training33 used external to this project; Africa-informed checkpoint-direction confirmation; not original validation/test or hidden blind; source-disjoint no patient bridge','preregistered':True,'not_test_tuned':True,'not_pristine_blind':True,'data':{'cases':33,'subjects':33,'cohort_sha256':p.COHORT_SHA,'cache_index_sha256':cache['index_sha256'],'official_sums_sha256':p.sha(p.SUMS),'validation_without_seg_excluded':6},'full972_kd168':{'checkpoint_sha256':p.FULL_SHA,'identity':fs['identity'],'identity_bootstrap':metrics.absolute(fs['identity'],p.BOOTSTRAP_SEED),'flip8':fs['flip8'],'flip8_bootstrap':metrics.absolute(fs['flip8'],p.BOOTSTRAP_SEED+10)},'gcsam184':{'checkpoint_sha256':p.GCSAM_SHA,'identity':gs,'identity_bootstrap':metrics.absolute(gs,p.BOOTSTRAP_SEED+20)},'comparisons':{'gcsam_identity_minus_full_identity':primary,'full_flip8_minus_identity':secondary},'africa_informed_hypothesis_interpretation':interpretation,'performance_promotion_gate':None,'result_must_not_trigger_tuning_or_repeat':True,'forwards':forwards,'elapsed_seconds':time.monotonic()-begin,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device),'access':{'cptac33':True,'internal_test270':False,'ucsd75':False,'africa95_new':False,'fixed_validation108':False,'outer':False,'confirmation':False},'training_source_caveat':'local BraTS2024 Kaggle mirror; official Synapse parity unresolved','evaluator_sha256':p.sha(Path(__file__))};p.write_new(out,report);print(json.dumps({'full_identity':{k:fs['identity'][k] for k in ('locator_dsc_mean','automatic_point_hits')},'full_flip8':{k:fs['flip8'][k] for k in ('locator_dsc_mean','automatic_point_hits')},'gcsam':{k:gs[k] for k in ('locator_dsc_mean','automatic_point_hits')},'comparisons':report['comparisons'],'interpretation':interpretation},indent=2))
if __name__=='__main__':main()
