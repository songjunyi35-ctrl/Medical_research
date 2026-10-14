#!/usr/bin/env python3
"""One-time WAW postoperative patient-disjoint31 frozen panel."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Dataset
import scripts.evaluate_brats_africa95_frozen_baselines as metrics
import scripts.evaluate_ucsd_ptgbm_external_confirmation as io
from scripts import waw_postop_only31_protocol as p
metrics.p=p
class Data(Dataset):
 def __init__(self):
  r=json.load(open(p.CACHE_RECEIPT));p.checked(p.CACHE/'index.json',r['index_sha256']);self.x=json.load(open(p.CACHE/'index.json'))
  if not r['all_gates_passed'] or self.x['cases']!=p.CASES:raise ValueError('cache')
  p.checked(p.CACHE/'images.f32',r['images_sha256']);p.checked(p.CACHE/'labels.u8',r['labels_sha256']);shape=(p.CASES,128,128,128);self.i=np.memmap(p.CACHE/'images.f32',dtype=np.float32,mode='r',shape=shape);self.l=np.memmap(p.CACHE/'labels.u8',dtype=np.uint8,mode='r',shape=shape)
 def __len__(self):return p.CASES
 def __getitem__(self,i):return {'image':torch.from_numpy(np.array(self.i[i]))[None],'label':torch.from_numpy(np.array(self.l[i]))[None],'case_id':self.x['case_ids'][i],'subject_id':self.x['subject_ids'][i]}
def interp(c,key):
 d=c[key]['scan_dice'];lo=c['paired_scan_dice_95ci'][0]
 return 'supported_on_WAW_postop31' if d>0 and lo>0 else ('positive_but_uncertain' if d>0 else 'not_supported_or_reversed')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args();p.docs();pre=json.load(open(p.MODEL_PREACCESS));cache=json.load(open(p.CACHE_RECEIPT));smoke=p.RECORD/'SMOKE_RECEIPT.json'
 if pre.get('status')!='frozen_after_cache_before_model' or pre['access']['model_inference'] is not False or pre['code']['evaluator_sha256']!=p.sha(Path(__file__)) or pre['cache_receipt_sha256']!=p.sha(p.CACHE_RECEIPT) or pre['smoke_receipt_sha256']!=p.sha(smoke):raise ValueError('preaccess')
 out=p.RECORD/'RAW_WAW_POSTOP31.json';started=p.RECORD/'RAW_WAW_POSTOP31_STARTED.json'
 if out.exists() or started.exists():raise FileExistsError('repeat')
 device=torch.device(a.device)
 if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA')
 torch.cuda.set_device(device);torch.set_num_threads(1);loader=DataLoader(Data(),batch_size=8,shuffle=False,num_workers=0);p.write_new(started,{'schema':'aflow.waw_postop_only31.started','created_at_utc':datetime.now(timezone.utc).isoformat(),'preaccess_sha256':p.sha(p.MODEL_PREACCESS),'checkpoint_loaded_at_write':False,'model_forwards_at_write':0})
 full,_=io.load_model(p.FULL,p.FULL_SHA,'primary',device);gc,_=io.load_model(p.GCSAM,p.GCSAM_SHA,'secondary',device);torch.cuda.reset_peak_memory_stats(device);begin=time.monotonic();fs,nf=metrics.evaluate_full(full,loader,device);del full;torch.cuda.empty_cache();gs,ng=metrics.evaluate_identity(gc,loader,device);forwards=nf+ng
 if forwards!=279:raise RuntimeError('forwards')
 flip=metrics.compare(fs['identity'],fs['flip8'],p.BOOTSTRAP_SEED+100,'flip8_minus_identity');gcomp=metrics.compare(fs['identity'],gs,p.BOOTSTRAP_SEED+200,'gcsam_minus_full')
 report={'schema':'aflow.waw_postop_only31.frozen_panel','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'evidence_scope':'WAW final postoperative earliest numeric day0-22 for31 patients absent from revealed preop152; institution-disjoint/no BraTS patient bridge; public not hidden blind','preregistered':True,'not_test_tuned':True,'not_pristine_blind':True,'data':{'cases':31,'subjects':31,'cohort_sha256':p.COHORT_SHA,'source_receipt_sha256':p.SOURCE_RECEIPT_SHA,'cache_index_sha256':cache['index_sha256'],'preop_patient_overlap':0,'later_or_blank_scans_excluded_before_performance':23,'target':'postop WT labels1/2/3 exclude RC4'},'full972_kd168':{'checkpoint_sha256':p.FULL_SHA,'identity':fs['identity'],'identity_bootstrap':metrics.absolute(fs['identity'],p.BOOTSTRAP_SEED),'flip8':fs['flip8'],'flip8_bootstrap':metrics.absolute(fs['flip8'],p.BOOTSTRAP_SEED+10)},'gcsam184':{'checkpoint_sha256':p.GCSAM_SHA,'identity':gs,'identity_bootstrap':metrics.absolute(gs,p.BOOTSTRAP_SEED+20)},'comparisons':{'full_flip8_minus_identity':flip,'gcsam_identity_minus_full_identity':gcomp},'interpretation_gates':{'full_flip8_minus_identity':interp(flip,'flip8_minus_identity'),'gcsam_identity_minus_full_identity':interp(gcomp,'gcsam_minus_full')},'performance_promotion_gate':None,'result_must_not_trigger_tuning_or_repeat':True,'forwards':forwards,'elapsed_seconds':time.monotonic()-begin,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device),'access':{'waw_postop_only31':True,'waw_preop152':False,'internal_test270':False,'fixed_validation108':False,'outer':False,'confirmation':False,'ucsd75':False,'africa95':False,'cptac33':False},'training_source_caveat':'local BraTS2024 Kaggle mirror; official Synapse parity unresolved','evaluator_sha256':p.sha(Path(__file__))};p.write_new(out,report);print(json.dumps({'full_identity':{k:fs['identity'][k] for k in ('locator_dsc_mean','automatic_point_hits')},'full_flip8':{k:fs['flip8'][k] for k in ('locator_dsc_mean','automatic_point_hits')},'gcsam':{k:gs[k] for k in ('locator_dsc_mean','automatic_point_hits')},'comparisons':report['comparisons'],'interpretation':report['interpretation_gates'],'forwards':forwards},indent=2))
if __name__=='__main__':main()
