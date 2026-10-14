"""Frozen CPTAC33 confirmation paths and hashes."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RECORD=ROOT/'archive/research_records/2026-09-06_cptac33_africa_informed_confirmation';REVIEW=ROOT/'archive/reviews/2026-09-06_cptac_brats2021_source_audit'
COHORT=REVIEW/'FROZEN_TRAINING33_COHORT.json';INVENTORY=REVIEW/'OFFICIAL_TRAINING33_INVENTORY.json';SOURCE=Path('/tmp/aflow-cptac33-source-20260906');CACHE=Path('/tmp/aflow-cptac33-cache-20260906')
SUMS=RECORD/'source_metadata/RSNA-ASNR-MICCAI-BraTS-2021.sums';SUMS_RECEIPT=RECORD/'SUMS_RECEIPT.json';SOURCE_PREACCESS=RECORD/'SOURCE_PREACCESS.json';SOURCE_RECEIPT=RECORD/'SOURCE_DOWNLOAD.json';FIRST_PROBE=RECORD/'FIRST_CASE_PROBE.json';CACHE_RECEIPT=RECORD/'CACHE_RECEIPT.json';MODEL_PREACCESS=RECORD/'MODEL_PREACCESS_V2.json'
PREREGISTERED_IDEA_SHA='2a2cad7482feae8ddc51525f759cfab884c25b8daaa991cd5b430f3c6b3cbf1f';IDEA_SHA='c619ff3b376ac45ccff42cf670c5970d314afdddb001006ad291e13a616f9448';PREREG_SHA='b9f7c59d8376b2be6aa8144c42f30fa2d9312fb9efa648439cc5c3615bde3e66';COHORT_SHA='2e1391cfd110721788aa2ddaaa4f34827eaab49b9c354d775c80651afc292208';INVENTORY_SHA='228785f0434037295a2e65a0d1a031419b074aff918fae0bc8cbbd10e8395418';CASES=33;FILES=66;BYTES=78013574;BOOTSTRAP_SEED=2026090701;BOOTSTRAP_SAMPLES=10000;AXES=((),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1))
FULL=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth';GCSAM=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth';FULL_SHA='f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b';GCSAM_SHA='acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def checked(path,expected):
 if sha(path)!=expected:raise ValueError('SHA mismatch '+str(path))
def docs():checked(RECORD/'IDEA.md',IDEA_SHA);checked(RECORD/'PREREGISTRATION.json',PREREG_SHA);checked(COHORT,COHORT_SHA);checked(INVENTORY,INVENTORY_SHA)
def rows():
 docs();d=json.load(open(COHORT));r=d['cases_detail']
 if d['status']!='frozen_before_sums_payload_arrays_or_model' or len(r)!=CASES or d['files_selected']!=FILES or d['bytes_selected']!=BYTES:raise ValueError('cohort')
 return r
def write_new(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
