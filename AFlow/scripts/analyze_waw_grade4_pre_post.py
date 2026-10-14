#!/usr/bin/env python3
"""Unpaired grade4-only WAW pre/post descriptive comparison."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PRE_RAW=ROOT/'archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json';POST_RAW=ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json';PRE_C=ROOT/'archive/reviews/2026-09-06_waw_gliomai_source_audit/FROZEN_COHORT.json';POST_C=ROOT/'archive/reviews/2026-09-06_waw_postop_only31_source_audit/FROZEN_COHORT.json';EXPECTED={PRE_RAW:'ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91',POST_RAW:'72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98',PRE_C:'25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a',POST_C:'965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 for p,d in EXPECTED.items():
  if sha(p)!=d:raise ValueError('SHA')
 raw={'pre':json.load(open(PRE_RAW)),'post':json.load(open(POST_RAW))};co={'pre':json.load(open(PRE_C)),'post':json.load(open(POST_C))};out={}
 for stage in ('pre','post'):
  meta={x['case_id']:x for x in co[stage]['cases']};ids={i for i in meta if meta[i]['who_grade']==4};models={'full_identity':raw[stage]['full972_kd168']['identity']['per_case'],'full_flip8':raw[stage]['full972_kd168']['flip8']['per_case'],'gcsam_identity':raw[stage]['gcsam184']['identity']['per_case']};base=[x for x in models['full_identity'] if x['case_id'] in ids];target=np.array([x['target_wt_voxels'] for x in base]);out[stage]={'subjects':len(ids),'target_wt':{'mean':float(target.mean()),'median':float(np.median(target)),'min':int(target.min()),'max':int(target.max())},'raw_label4_present':sum(4 in meta[i].get('raw_values',meta[i].get('mask_labels',[])) for i in ids),'models':{name:{'dice':statistics.fmean(x['locator_dsc'] for x in rows if x['case_id'] in ids),'point_hits':sum(x['automatic_point_hits_gt'] for x in rows if x['case_id'] in ids),'point_denominator':len(ids)} for name,rows in models.items()}}
 result={'schema':'aflow.waw.grade4_pre_post_description','schema_version':1,'status':'unpaired_patient_disjoint_posthoc_description_no_causal_test','grade':4,'cohorts':out,'post_minus_pre':{'target_mean':out['post']['target_wt']['mean']-out['pre']['target_wt']['mean'],'target_median':out['post']['target_wt']['median']-out['pre']['target_wt']['median'],'model_dice':{m:out['post']['models'][m]['dice']-out['pre']['models'][m]['dice'] for m in out['pre']['models']},'model_point_rate':{m:out['post']['models'][m]['point_hits']/out['post']['subjects']-out['pre']['models'][m]['point_hits']/out['pre']['subjects'] for m in out['pre']['models']}},'interpretation':'Grade4-only postoperative performance and point hit remain much lower even though postop targets are smaller. Groups are different patients and differ in surgery, RC, timing, scanner/case mix; this rules out grade or larger target as sufficient explanations but is not a causal before-after estimate.','access':{'new_model_inference':False,'image_or_cache_access':False,'selection_or_tuning':False},'forbidden_use':'no causal surgery effect, day/grade/RC routing, threshold/postprocessing, or cohort retest','source_sha256':{str(p.relative_to(ROOT)):d for p,d in EXPECTED.items()},'analyzer_sha256':sha(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'output_sha256':sha(a.output),'post_minus_pre':result['post_minus_pre']},indent=2))
if __name__=='__main__':main()
