#!/usr/bin/env python3
"""Post-hoc WAW152 scanner/field-strength description from saved rows."""
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
 if sha(RAW)!=RAW_SHA or sha(COHORT)!=COHORT_SHA:raise ValueError('SHA')
 raw=json.load(open(RAW));co=json.load(open(COHORT));meta={x['case_id']:x for x in co['cases']};full={x['case_id']:x for x in raw['full972_kd168']['identity']['per_case']};flip={x['case_id']:x for x in raw['full972_kd168']['flip8']['per_case']};gc={x['case_id']:x for x in raw['gcsam184']['identity']['per_case']}
 def group(ids):return {'cases':len(ids),'full_identity_dice':statistics.fmean(full[i]['locator_dsc'] for i in ids),'flip8_minus_full':statistics.fmean(flip[i]['locator_dsc']-full[i]['locator_dsc'] for i in ids),'gcsam_minus_full':statistics.fmean(gc[i]['locator_dsc']-full[i]['locator_dsc'] for i in ids),'point_hits':{'full':sum(full[i]['automatic_point_hits_gt'] for i in ids),'flip8':sum(flip[i]['automatic_point_hits_gt'] for i in ids),'gcsam':sum(gc[i]['automatic_point_hits_gt'] for i in ids)},'grade_counts':dict(collections.Counter(str(meta[i]['who_grade']) for i in ids))}
 ids=set(full);architect={i for i in ids if meta[i]['scanner_name']=='SIGNA Architect'};field={name:{i for i in ids if meta[i]['field_strength']==name} for name in ('3T','1.5T')};field['other_or_missing']=ids-field['3T']-field['1.5T']
 scanners=collections.Counter(meta[i]['scanner_name'] or 'missing' for i in ids);result={'schema':'aflow.waw_gliomai152.posthoc_scanner_description','schema_version':1,'status':'posthoc_descriptive_no_inference_or_selection','scanner_counts':dict(scanners),'dominant_scanner':{'SIGNA Architect':group(architect),'all_other':group(ids-architect)},'field_strength':{k:group(v) for k,v in field.items()},'interpretation':'GCSAM positive point estimate is concentrated in the dominant Architect/3T strata, while Flip8 remains positive in dominant versus other and 3T versus1.5T groups. Scanner, field strength, grade, labels, and case mix are confounded; no causal or routing claim.','forbidden_use':'no scanner/field/grade router, checkpoint selection, threshold, exclusion, or retest','new_model_inference':False,'image_or_cache_access':False,'source_sha256':{'raw':RAW_SHA,'cohort':COHORT_SHA},'analyzer_sha256':sha(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'output_sha256':sha(a.output),'architect_cases':len(architect),'other_cases':len(ids-architect)},indent=2))
if __name__=='__main__':main()
