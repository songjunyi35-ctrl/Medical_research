#!/usr/bin/env python3
"""Terminal recomputation and evidence receipt for WAW postop-only31."""
from __future__ import annotations
import hashlib,json,statistics
from pathlib import Path
import scripts.evaluate_brats_africa95_frozen_baselines as metrics
import scripts.evaluate_waw_postop_only31_panel as evaluator
from scripts import waw_postop_only31_protocol as p
metrics.p=p;RAW_SHA='72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98';START_SHA='b2485e1efe99fe7ffd98d840c1ce76a833220cbed983a321d5eb8c03662641bf';PRE_SHA='e8e13d35491672a641f4f7879a3f627a45438368fc04febcf341040c54a3acad';CACHE_SHA='acfc61760c654975fee89de0339dbcb39cdbaa0633064569a1adac84cb0d76bf';ARCHIVE_SHA='6236c97c81a3c1a02543d5bf48425946a481861f9dde96bb19b6482f69508080';EVAL_SHA='41a2fccc66b4730a55ab86ee5eca866505397a1d4cdbfb9940185ec514cba7d2'
def md5(path):
 h=hashlib.md5()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def req(x,msg):
 if not x:raise ValueError(msg)
def summary(x):
 rows=x['per_case'];req(len(rows)==len({r['case_id'] for r in rows})==31,'cases');req(abs(statistics.fmean(r['locator_dsc'] for r in rows)-x['locator_dsc_mean'])<1e-12,'mean');req(sum(r['automatic_point_hits_gt'] for r in rows)==x['automatic_point_hits'],'point')
def main():
 result=p.RECORD/'RESULT.json';receipt=p.RECORD/'VERIFICATION_RECEIPT.json';idxcopy=p.RECORD/'CACHE_INDEX_FINAL.json'
 if any(x.exists() for x in (result,receipt,idxcopy)):raise FileExistsError('output')
 p.docs();rawp=p.RECORD/'RAW_WAW_POSTOP31.json';startp=p.RECORD/'RAW_WAW_POSTOP31_STARTED.json';p.checked(rawp,RAW_SHA);p.checked(startp,START_SHA);p.checked(p.MODEL_PREACCESS,PRE_SHA);p.checked(p.CACHE_RECEIPT,CACHE_SHA);p.checked(p.FULL,p.FULL_SHA);p.checked(p.GCSAM,p.GCSAM_SHA);raw=json.load(open(rawp));req(raw['forwards']==279 and raw['evaluator_sha256']==EVAL_SHA,'raw')
 sections=[('full972_kd168','identity',p.BOOTSTRAP_SEED),('full972_kd168','flip8',p.BOOTSTRAP_SEED+10),('gcsam184','identity',p.BOOTSTRAP_SEED+20)]
 for model,view,seed in sections:
  x=raw[model][view];summary(x);req(metrics.absolute(x,seed)==raw[model][view+'_bootstrap'],'bootstrap')
 fi=raw['full972_kd168']['identity'];ff=raw['full972_kd168']['flip8'];gc=raw['gcsam184']['identity'];req(metrics.compare(fi,ff,p.BOOTSTRAP_SEED+100,'flip8_minus_identity')==raw['comparisons']['full_flip8_minus_identity'],'flip');req(metrics.compare(fi,gc,p.BOOTSTRAP_SEED+200,'gcsam_minus_full')==raw['comparisons']['gcsam_identity_minus_full_identity'],'gc');req(evaluator.interp(raw['comparisons']['full_flip8_minus_identity'],'flip8_minus_identity')==raw['interpretation_gates']['full_flip8_minus_identity'],'interp')
 archive=p.SOURCE/'2_2_postop_scans_coreg.zip';req(archive.stat().st_size==3321955416 and md5(archive)=='ab9da3eb2a3aff73b8f3b95e01508c1c','archive');p.checked(archive,ARCHIVE_SHA);source=json.load(open(p.SOURCE_RECEIPT));n=0;sourcebytes=0
 for x in source['selected']['files']:
  for pathkey,shakey in (('image_path','image_sha256'),('mask_path','mask_sha256')):
   path=Path(x[pathkey]);p.checked(path,x[shakey]);n+=1;sourcebytes+=path.stat().st_size
 req(n==62,'source count');cr=json.load(open(p.CACHE_RECEIPT));index=p.CACHE/'index.json';p.checked(index,cr['index_sha256']);p.checked(p.CACHE/'images.f32',cr['images_sha256']);p.checked(p.CACHE/'labels.u8',cr['labels_sha256']);p.write_new(idxcopy,json.load(open(index)))
 boardpath=p.ROOT/'AFLOW_BRATS_LEADERBOARD.json';board=json.load(open(boardpath));rows=board['external_waw_postop_only31_direct_locator_rank_by_scan'];mapping={'full972_random_init_voxel_kd_epoch168_flip8_tta':ff,'gcsam_inner_fit864_epoch184_identity':gc,'full972_random_init_voxel_kd_epoch168_identity':fi};req([x['rank'] for x in rows]==[1,2,3],'ranks')
 for x in rows:req(x['scan_dice']==mapping[x['method']]['locator_dsc_mean'] and x['point_hits']==mapping[x['method']]['automatic_point_hits'] and x['report_sha256']==RAW_SHA,'board')
 value={'schema':'aflow.waw_postop_only31.result','schema_version':1,'status':'completed_and_cohort_closed','cohort':{'cases':31,'subjects':31,'days_postop':[0.0,22.0],'preop_patient_overlap':0,'evidence':'public early postoperative; same WAW source/new patients; institution-disjoint/no BraTS patient bridge; not hidden blind'},'absolute':{'full_identity':{'dice':fi['locator_dsc_mean'],'ci':raw['full972_kd168']['identity_bootstrap']['scan_dsc_mean_95ci'],'point_hits':fi['automatic_point_hits']},'full_flip8':{'dice':ff['locator_dsc_mean'],'ci':raw['full972_kd168']['flip8_bootstrap']['scan_dsc_mean_95ci'],'point_hits':ff['automatic_point_hits']},'gcsam_identity':{'dice':gc['locator_dsc_mean'],'ci':raw['gcsam184']['identity_bootstrap']['scan_dsc_mean_95ci'],'point_hits':gc['automatic_point_hits']}},'comparisons':raw['comparisons'],'interpretation':raw['interpretation_gates'],'cost':{'training':0,'cache_seconds':cr['elapsed_seconds'],'formal_seconds':raw['elapsed_seconds'],'forwards':279,'peak_gpu_memory_bytes':raw['peak_gpu_memory_bytes'],'archive_bytes':archive.stat().st_size,'selected_source_bytes_rehashed':sourcebytes,'cache_bytes':(p.CACHE/'images.f32').stat().st_size+(p.CACHE/'labels.u8').stat().st_size},'access':raw['access'],'raw_sha256':RAW_SHA,'preaccess_sha256':PRE_SHA,'source_receipt_sha256':p.SOURCE_RECEIPT_SHA,'cache_index_sha256':cr['index_sha256'],'result_must_not_trigger_tuning_or_repeat':True};p.write_new(result,value)
 vr={'schema':'aflow.waw_postop_only31.verification','schema_version':1,'status':'passed','cases':31,'subjects':31,'archive_sha256':ARCHIVE_SHA,'official_md5_exact':True,'source_files_rehashed':n,'source_bytes_rehashed':sourcebytes,'source_receipt_sha256':p.SOURCE_RECEIPT_SHA,'cache_receipt_sha256':CACHE_SHA,'cache_index_sha256':cr['index_sha256'],'cache_index_archive_sha256':p.sha(idxcopy),'images_sha256':cr['images_sha256'],'labels_sha256':cr['labels_sha256'],'full_checkpoint_sha256':p.FULL_SHA,'gcsam_checkpoint_sha256':p.GCSAM_SHA,'preaccess_sha256':PRE_SHA,'started_sha256':START_SHA,'raw_sha256':RAW_SHA,'result_sha256':p.sha(result),'leaderboard_sha256':p.sha(boardpath),'recomputed':{'absolute':3,'paired':2,'leaderboard':3},'model_inference':False,'new_metrics':False,'verifier_sha256':p.sha(Path(__file__))};p.write_new(receipt,vr);print(json.dumps(vr,indent=2))
if __name__=='__main__':main()
