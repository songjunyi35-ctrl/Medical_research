#!/usr/bin/env python3
"""Compare train972 and WAW152 label observability for T1-contrast-only input."""
from __future__ import annotations
import argparse,collections,hashlib,json,statistics
from pathlib import Path
import nibabel as nib,numpy as np
ROOT=Path(__file__).resolve().parents[1];MANIFEST=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/train_t1ce_wt.jsonl';WAW=ROOT/'archive/reviews/2026-09-06_waw_gliomai_source_audit/FROZEN_COHORT.json';TAIL=ROOT/'archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/POSTHOC_FAILURE_TAIL.json';EXPECTED={MANIFEST:'6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65',WAW:'25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a',TAIL:'cdfed37eb2e7586c65525cc25d7b1bf420e172b63a76798050d4a887675ae714'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def summarize(patterns,volumes,subjects):
 n=len(patterns);no3=[i for i,x in enumerate(patterns) if 3 not in x];only2=[i for i,x in enumerate(patterns) if x=={0,2}]
 return {'scans':n,'subjects':len(set(subjects)),'pattern_counts':{str(tuple(sorted(k))):v for k,v in collections.Counter(tuple(sorted(x)) for x in patterns).items()},'label3_absent':len(no3),'label3_absent_fraction':len(no3)/n,'only_label2':len(only2),'only_label2_fraction':len(only2)/n,'wt_voxels_median_all':float(np.median(volumes)),'wt_voxels_median_label3_absent':float(np.median([volumes[i] for i in no3])) if no3 else None,'subjects_with_any_label3_absent_scan':len({subjects[i] for i in no3}),'subjects_with_only_label2_scan':len({subjects[i] for i in only2})}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 for path,digest in EXPECTED.items():
  if sha(path)!=digest:raise ValueError('SHA '+str(path))
 rows=[json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
 if len(rows)!=972 or any(x['split']!='train' for x in rows):raise ValueError('manifest')
 patterns=[];volumes=[];subjects=[]
 for i,row in enumerate(rows,1):
  arr=np.asanyarray(nib.load(row['label_path'],mmap=True).dataobj);rounded=np.rint(arr);values={int(x) for x in np.unique(rounded)}
  if not np.allclose(arr,rounded,rtol=0,atol=1e-5) or not values<={0,1,2,3,4}:raise ValueError(row['case_id'])
  patterns.append(values);volumes.append(int(np.count_nonzero(np.isin(rounded,[1,2,3]))));subjects.append(row['subject_id'])
  if i%200==0:print(f'train labels {i}/972',flush=True)
 train=summarize(patterns,volumes,subjects);w=json.load(open(WAW));wpatterns=[set(x['mask_labels']) for x in w['cases']];wvol=[x['mask_positive_voxels'] for x in w['cases']];wsub=[x['subject_id'] for x in w['cases']];waw=summarize(wpatterns,wvol,wsub);tail=json.load(open(TAIL))
 result={'schema':'aflow.t1ce_only.label_observability_audit','schema_version':1,'status':'train_labels_plus_saved_waw_metadata_only_no_model_or_new_evaluation_role','train972_posttreatment':train,'waw152_preoperative':waw,'contrasts':{'waw_minus_train_label3_absent_fraction':waw['label3_absent_fraction']-train['label3_absent_fraction'],'waw_minus_train_only_label2_fraction':waw['only_label2_fraction']-train['only_label2_fraction'],'waw_common_zero_dice_cases':tail['all_three_zero_count'],'waw_common_zero_only_label2_cases':sum(x['mask_labels']==[0,2] for x in tail['all_three_zero_dice_cases'])},'interpretation':'T1-contrast-only input cannot directly observe FLAIR-defined abnormality. Distribution differences in enhancement-label presence can contribute to domain failure but do not prove causality or authorize GT-based routing.','access':{'train_role_labels':972,'waw_new_arrays':False,'validation':False,'outer':False,'confirmation':False,'internal_test':False,'model_inference':False},'forbidden_use':'no GT label-composition selection, router, exclusion, threshold, training method, or cohort retest','source_sha256':{str(k.relative_to(ROOT)):v for k,v in EXPECTED.items()},'analyzer_sha256':sha(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'train_label3_absent':train['label3_absent'],'train_only2':train['only_label2'],'waw_label3_absent':waw['label3_absent'],'waw_only2':waw['only_label2'],'output_sha256':sha(a.output)},indent=2))
if __name__=='__main__':main()
