"""Frozen paths and constants for the MOTUM-HGG29 panel."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/'archive/research_records/2026-09-06_motum_hgg29_frozen_panel';REVIEW=ROOT/'archive/reviews/2026-09-06_motum_hgg_source_audit';SOURCE=ROOT/'workspace/BraTS/motum_source_20260906';CACHE=SOURCE/'cache128'
COHORT=REVIEW/'FROZEN_HGG29_COHORT.json';AUDIT=REVIEW/'SOURCE_AUDIT.json';FIRST_PROBE=RECORD/'FIRST_CASE_PROBE.json';CACHE_RECEIPT=RECORD/'CACHE_RECEIPT.json';MODEL_PREACCESS=RECORD/'MODEL_PREACCESS.json'
PREREGISTERED_IDEA_SHA='6327548760ddb153ec408a1562a1a10498ca724a4f55c2056dc2a611ffcb35f5';IDEA_SHA='1ad404a5cba66ebd0b8d56545b1882d17b247a029bff6979f74f669f4f13fcc4';PREREG_SHA='638767320eb77aba292968e686be7429056eb19e7552534ac923160024d5a162';AUDIT_SHA='69d59768cd54ed526aa1cc8b0f66e31884ea4aaad171a1a16bc17c4488412932';COHORT_SHA='f49e8f888b0fce8b7f34da1f9eadb49039d10aaed8e0538008f215df1b953379'
CASES=29;BOOTSTRAP_SEED=20261481;BOOTSTRAP_SAMPLES=10000;AXES=((),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1))
FULL=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/final_formal_full972_candidate/plain_fulltrajectory_full972_candidate_epoch168.pth';GCSAM=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/inner_formal_gcsam_candidate/post100_best.pth';FULL_SHA='f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b';GCSAM_SHA='acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def checked(path,expected):
 if sha(path)!=expected:raise ValueError('SHA mismatch '+str(path))
def docs():checked(RECORD/'IDEA.md',IDEA_SHA);checked(RECORD/'PREREG.md',PREREG_SHA);checked(AUDIT,AUDIT_SHA);checked(COHORT,COHORT_SHA)
def cases():
 docs();d=json.load(open(COHORT));r=d['cases']
 if d['status']!='frozen_after_source_arrays_before_model_or_cache' or d['case_count']!=CASES or d['subject_count']!=CASES or len(r)!=CASES:raise ValueError('cohort')
 return r
def write_new(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
