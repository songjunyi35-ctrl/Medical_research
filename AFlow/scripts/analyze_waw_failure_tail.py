#!/usr/bin/env python3
"""Describe WAW152 failure tails from saved rows and frozen label metadata."""
from __future__ import annotations
import argparse,collections,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json';COHORT=ROOT/'archive/reviews/2026-09-06_waw_gliomai_source_audit/FROZEN_COHORT.json';RAW_SHA='ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91';COHORT_SHA='25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 if sha(RAW)!=RAW_SHA or sha(COHORT)!=COHORT_SHA:raise ValueError('source SHA')
 raw=json.load(open(RAW));co=json.load(open(COHORT));meta={x['case_id']:x for x in co['cases']};rows={'full_identity':raw['full972_kd168']['identity']['per_case'],'full_flip8':raw['full972_kd168']['flip8']['per_case'],'gcsam_identity':raw['gcsam184']['identity']['per_case']};ids=[x['case_id'] for x in rows['full_identity']]
 if any([x['case_id'] for x in r]!=ids for r in rows.values()):raise ValueError('order')
 def summary(rs):
  dice=[x['locator_dsc'] for x in rs];hit=[x['automatic_point_hits_gt'] for x in rs]
  return {'dice_zero':sum(x==0 for x in dice),'dice_below_0_2':sum(x<.2 for x in dice),'dice_below_0_5':sum(x<.5 for x in dice),'point_misses':sum(not x for x in hit),'dice_mean_when_point_hits':statistics.fmean(x for x,h in zip(dice,hit) if h),'dice_mean_when_point_misses':statistics.fmean(x for x,h in zip(dice,hit) if not h),'worst8':[{'case_id':x['case_id'],'dice':x['locator_dsc'],'point_hit':x['automatic_point_hits_gt'],'who_grade':meta[x['case_id']]['who_grade'],'mask_labels':meta[x['case_id']]['mask_labels'],'scanner':meta[x['case_id']]['scanner_name']} for x in sorted(rs,key=lambda x:x['locator_dsc'])[:8]]}
 groups={}
 for name,pred in [('enhancing_label3_present',lambda m:3 in m['mask_labels']),('enhancing_label3_absent',lambda m:3 not in m['mask_labels']),('only_flair_abnormality_label2',lambda m:m['mask_labels']==[0,2])]:
  chosen={i for i in ids if pred(meta[i])};groups[name]={'cases':len(chosen),'grade_counts':dict(collections.Counter(str(meta[i]['who_grade']) for i in chosen)),'models':{model:{'dice_mean':statistics.fmean(x['locator_dsc'] for x in rs if x['case_id'] in chosen),'dice_zero':sum(x['locator_dsc']==0 for x in rs if x['case_id'] in chosen),'point_hits':sum(x['automatic_point_hits_gt'] for x in rs if x['case_id'] in chosen)} for model,rs in rows.items()}}
 zero_sets=[{x['case_id'] for x in rs if x['locator_dsc']==0} for rs in rows.values()];allzero=set.intersection(*zero_sets);base={x['case_id']:x for x in rows['full_identity']};rescues={}
 for model in ('full_flip8','gcsam_identity'):
  cand={x['case_id']:x for x in rows[model]};low={i for i,x in base.items() if x['locator_dsc']<.5};rescues[model]={'full_identity_below_0_5':len(low),'rescued_to_at_least_0_5':sum(cand[i]['locator_dsc']>=.5 for i in low),'newly_harmed_below_0_5':sum(base[i]['locator_dsc']>=.5 and cand[i]['locator_dsc']<.5 for i in ids),'point_gained':sum(not base[i]['automatic_point_hits_gt'] and cand[i]['automatic_point_hits_gt'] for i in ids),'point_lost':sum(base[i]['automatic_point_hits_gt'] and not cand[i]['automatic_point_hits_gt'] for i in ids)}
 result={'schema':'aflow.waw_gliomai152.posthoc_failure_tail','schema_version':1,'status':'posthoc_descriptive_saved_rows_and_label_metadata_no_selection','cases':152,'models':{k:summary(v) for k,v in rows.items()},'label_pattern_counts':{str(k):v for k,v in collections.Counter(tuple(x['mask_labels']) for x in co['cases']).items()},'label_observability_groups':groups,'all_three_zero_dice_cases':[{'case_id':i,'who_grade':meta[i]['who_grade'],'mask_labels':meta[i]['mask_labels'],'scanner':meta[i]['scanner_name']} for i in sorted(allzero)],'all_three_zero_count':len(allzero),'candidate_rescue_harm':rescues,'interpretation':'Four of five common zero-Dice cases contain only FLAIR-abnormality label2 and all five are grade2. T1 contrast-only input has a target observability limitation; this is not permission to use GT label composition for routing or exclusion.','forbidden_use':'no label/grade/scanner router, case exclusion, threshold/postprocessing, or cohort retest','new_model_inference':False,'image_or_cache_access':False,'source_sha256':{'raw':RAW_SHA,'cohort':COHORT_SHA},'analyzer_sha256':sha(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'all_three_zero':len(allzero),'output_sha256':sha(a.output)},indent=2))
if __name__=='__main__':main()
