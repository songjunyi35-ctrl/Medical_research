"""Frozen contract for WAW postop-only patient-disjoint31."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RECORD=ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel';REVIEW=ROOT/'archive/reviews/2026-09-06_waw_postop_only31_source_audit';SOURCE=ROOT/'workspace/BraTS/waw_gliomai_source_20260906';CACHE=SOURCE/'cache_postop31';COHORT=REVIEW/'FROZEN_COHORT.json';AUDIT=REVIEW/'SOURCE_AUDIT.json';SOURCE_RECEIPT=RECORD/'SOURCE_RECEIPT.json';CACHE_RECEIPT=RECORD/'CACHE_RECEIPT.json';FIRST_PROBE=RECORD/'FIRST_CASE_PROBE.json';MODEL_PREACCESS=RECORD/'MODEL_PREACCESS.json'
PREREGISTERED_IDEA_SHA='4f21bcd4babbf229b683c84c9bb1a3992ec6cf3da90203beaaf0b9187d100cfc';IDEA_SHA='b28611c5861bc18d60f130dd065b21a606830901bcdfeccd09e3fbb4e53e427f';PREREG_SHA='078746900c854ee2b59972d49132bd3787c53dba902a4fd95009a14a1a4a6729';AUDIT_SHA='ca7d67892e42ae4d193f87ebdd983fb79314091b84d2e448021a71da18e0fe3d';COHORT_SHA='965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180';SOURCE_PREACCESS_SHA='0fffdc57e71febe972fae1c977939a01d50a18ad06b8492193f9c7e2b8d46c77';EXTRACTION_SHA='e08ef83528d8922bb82ce65ca188d058181e3606b9f8881615fd093ed8d43162';SOURCE_RECEIPT_SHA='ff255098ef7ea6f91f1f90b37193d8d8d7c611b99ba666919c7e3d95252997e9';CASES=31;BOOTSTRAP_SEED=20261531;BOOTSTRAP_SAMPLES=10000;AXES=((),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1))
FULL=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth';GCSAM=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth';FULL_SHA='f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b';GCSAM_SHA='acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def checked(path,d):
 if sha(path)!=d:raise ValueError('SHA '+str(path))
def docs():checked(RECORD/'IDEA.md',IDEA_SHA);checked(RECORD/'PREREG.md',PREREG_SHA);checked(AUDIT,AUDIT_SHA);checked(COHORT,COHORT_SHA);checked(RECORD/'SOURCE_PREACCESS.json',SOURCE_PREACCESS_SHA);checked(RECORD/'EXTRACTION_RECEIPT.json',EXTRACTION_SHA);checked(SOURCE_RECEIPT,SOURCE_RECEIPT_SHA)
def cases():
 docs();d=json.load(open(COHORT));r=d['cases']
 if len(r)!=CASES or d['case_count']!=CASES or d['subject_count']!=CASES:raise ValueError('cohort')
 return r
def write_new(path,v):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
