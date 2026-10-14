#!/usr/bin/env python3
"""Verify CPTAC33 sources, caches, raw metrics and leaderboard without inference."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
import scripts.evaluate_brats_africa95_frozen_baselines as m
import scripts.evaluate_cptac33_confirmation as e
from scripts import cptac33_protocol as p
m.p=p
RAW='4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd';START='58401d39fe82cbf06cf4c42392fe3b0ee85732508465428eff9f8d3146cd1ca0';SOURCE='8c23620f6800c4f6d885e346b0b7ccef35149f0dba3fba9b868af1dfed2e8bb8';CACHE='b14d789c9e5ca710a5509556eacc3a957a9cf4c7a8e595914f4db299594b10fa';INDEX='9eeebdf66779496259b3a909ae8262dc380f3eaf73544059383b0d6c003cd140';PRE='117cb7c38000107f44823f204d171bf13a4ce8a2bc84e96487b670abb1900f09'
def md5(path):
 h=hashlib.md5()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def req(x,msg):
 if not x:raise ValueError(msg)
def summary(x):
 rows=x['per_case'];req(len(rows)==len({r['case_id'] for r in rows})==33,'cases');mean=statistics.fmean(r['locator_dsc'] for r in rows);hits=sum(r['automatic_point_hits_gt'] for r in rows);return mean,hits
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--cache-index',type=Path,required=True);a=ap.parse_args()
 if a.receipt.exists() or a.cache_index.exists():raise FileExistsError('output')
 rawp=p.RECORD/'RAW_CPTAC33.json';startp=p.RECORD/'RAW_CPTAC33_STARTED.json';p.checked(rawp,RAW);p.checked(startp,START);p.checked(p.SOURCE_RECEIPT,SOURCE);p.checked(p.CACHE_RECEIPT,CACHE);p.checked(p.CACHE/'index.json',INDEX);p.checked(p.MODEL_PREACCESS,PRE);p.checked(p.FULL,p.FULL_SHA);p.checked(p.GCSAM,p.GCSAM_SHA)
 sums=json.load(open(p.SUMS_RECEIPT));p.checked(p.SUMS,sums['file']['sha256']);source=json.load(open(p.SOURCE_RECEIPT));n=0;total=0
 for copy in source['copies']:
  for item in copy['files']:
   path=Path(item['path']);req(path.stat().st_size==item['bytes']==item['expected_bytes'],'bytes');req(md5(path)==item['official_md5']==item['md5'],'md5');p.checked(path,item['sha256']);n+=1;total+=item['bytes']
 req(n==132 and total==2*p.BYTES,'source total')
 index=json.load(open(p.CACHE/'index.json'));p.checked(p.CACHE/'images.f32',index['images_sha256']);p.checked(p.CACHE/'labels.u8',index['labels_sha256']);p.write_new(a.cache_index,index)
 raw=json.load(open(rawp));req(raw['forwards']==297 and raw['evaluator_sha256']==p.sha(Path(e.__file__)),'raw contract');sections=[('full972_kd168','identity',p.BOOTSTRAP_SEED),('full972_kd168','flip8',p.BOOTSTRAP_SEED+10),('gcsam184','identity',p.BOOTSTRAP_SEED+20)]
 for model,view,seed in sections:
  x=raw[model][view];mean,hits=summary(x);req(abs(mean-x['locator_dsc_mean'])<1e-12 and hits==x['automatic_point_hits'],'summary');req(m.absolute(x,seed)==raw[model][view+'_bootstrap'],'bootstrap')
 fi=raw['full972_kd168']['identity'];ff=raw['full972_kd168']['flip8'];gc=raw['gcsam184']['identity'];req(m.compare(fi,gc,p.BOOTSTRAP_SEED+200,'gcsam_minus_full')==raw['comparisons']['gcsam_identity_minus_full_identity'],'g compare');req(m.compare(fi,ff,p.BOOTSTRAP_SEED+100,'flip8_minus_identity')==raw['comparisons']['full_flip8_minus_identity'],'flip compare')
 board=json.load(open(p.ROOT/'AFLOW_BRATS_LEADERBOARD.json'));rows=board['external_cptac33_direct_locator_rank_by_scan'];mapping={'full972_random_init_voxel_kd_epoch168_flip8_tta':ff,'gcsam_inner_fit864_epoch184_identity':gc,'full972_random_init_voxel_kd_epoch168_identity':fi};req([r['rank'] for r in rows]==[1,2,3],'rank')
 for row in rows:
  x=mapping[row['method']];req(row['scan_dice']==x['locator_dsc_mean'] and row['point_hits']==x['automatic_point_hits'] and row['report_sha256']==RAW,'board')
 receipt={'schema':'aflow.cptac33.verification','schema_version':1,'status':'passed','cases':33,'subjects':33,'sums_sha256':p.sha(p.SUMS),'sums_rows':sums['rows'],'source_receipt_sha256':SOURCE,'raw_source_files_rehashed':n,'raw_source_bytes_rehashed':total,'official_md5_and_dual_sha_exact':True,'cache_receipt_sha256':CACHE,'cache_index_sha256':INDEX,'cache_index_archive_sha256':p.sha(a.cache_index),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'preaccess_v2_sha256':PRE,'raw_sha256':RAW,'started_sha256':START,'full_checkpoint_sha256':p.FULL_SHA,'gcsam_checkpoint_sha256':p.GCSAM_SHA,'evaluator_sha256':raw['evaluator_sha256'],'recomputed':{'absolute':3,'paired':2,'leaderboard':3},'model_inference':False,'new_metrics':False,'verifier_sha256':p.sha(Path(__file__))};p.write_new(a.receipt,receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
