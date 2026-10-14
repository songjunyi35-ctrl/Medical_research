#!/usr/bin/env python3
"""Freeze the labeled CPTAC BraTS2021 public Training33 after 39-case failure."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path

import scripts.audit_brats_africa_source_20260906 as api
import scripts.audit_cptac_brats2021_source_20260906 as prior

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'archive/reviews/2026-09-06_cptac_brats2021_source_audit';MAPPING=OUT/'source_metadata/BraTS2021_MappingToTCIA.xlsx'

def main():
 invout=OUT/'OFFICIAL_TRAINING33_INVENTORY.json';cohortout=OUT/'FROZEN_TRAINING33_COHORT.json'
 if invout.exists() or cohortout.exists() or MAPPING.exists():raise FileExistsError('output')
 api.PAGE=prior.PAGE;api.DATAINFO=prior.MAPPING;api.PACKAGE_ID=636;bearer,page,mapping,package=api.authorize()
 if prior.sha_bytes(mapping)!='223243c5821c282e90da980ba38d860ab2d2a597d1f4f1800ebe227329a2dc0b':raise ValueError('mapping SHA')
 rows=prior.parse_mapping(mapping);train=[r for r in rows if r['Segmentation (Task 1) Cohort']=='Training'];val=[r for r in rows if r['Segmentation (Task 1) Cohort']=='Validation']
 if len(train)!=33 or len(val)!=6:raise ValueError('role counts')
 def path(row):return f"/RSNA-ASNR-MICCAI-BraTS-2021/BraTS2021_{row['Segmentation (Task 1) Cohort']}Set/CPTAC-GBM/{row['BraTS2021 ID']}"
 def read(row):return row,*api.list_directory(path(row),bearer)
 results=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for item in pool.map(read,train+val):results.append(item)
 files=[];cases=[];validation=[]
 for row,items,pages in results:
  bid=row['BraTS2021 ID'];compact=[api.compact(x) for x in items];names={x['basename'] for x in compact};base={f'{bid}_{s}.nii.gz' for s in ('flair','t1','t1ce','t2')}
  if row['Segmentation (Task 1) Cohort']=='Training':
   expected=base|{f'{bid}_seg.nii.gz'}
   if pages!=1 or names!=expected or len(compact)!=5:raise ValueError('training contract '+bid)
   by={x['basename']:x for x in compact};cases.append({'brats2021_id':bid,'cptac_patient_id':row['PatientID on TCIA Radiology Portal'],'study_date':row.get('Study date (m/d/yyyy) per PatientID'),'original_role':'Training','t1ce':by[f'{bid}_t1ce.nii.gz'],'seg':by[f'{bid}_seg.nii.gz']});files.extend(compact)
  else:
   if pages!=1 or names!=base or len(compact)!=4:raise ValueError('validation expected unlabeled '+bid)
   validation.append({'brats2021_id':bid,'cptac_patient_id':row['PatientID on TCIA Radiology Portal'],'files':4,'seg':False})
  if any(x['type']!='symbolic_link' or x['target_type']!='file' or int(x['size'] or 0)<=0 for x in compact):raise ValueError('file type/size '+bid)
 cases.sort(key=lambda x:x['brats2021_id']);validation.sort(key=lambda x:x['brats2021_id']);files.sort(key=lambda x:x['path'])
 MAPPING.parent.mkdir(parents=True,exist_ok=True);MAPPING.write_bytes(mapping)
 inventory={'schema':'aflow.cptac_brats2021.training33_inventory','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'package_id':636,'files':files};invout.write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
 cohort={'schema':'aflow.cptac_brats2021.training33_frozen_cohort','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_sums_payload_arrays_or_model','cases':33,'subjects':33,'files_selected':66,'bytes_selected':sum(int(x[k]['size']) for x in cases for k in ('t1ce','seg')),'cases_detail':cases,'excluded_unlabeled_validation':{'cases':6,'details':validation},'evidence_role':'BraTS2021 public Training labels, external to this project; not original validation/test or hidden blind','source':{'package_id':636,'mapping_sha256':prior.sha_bytes(mapping),'page_sha256':prior.sha_bytes(page),'inventory_sha256':prior.sha(invout)},'institution':'CPTAC-GBM/CHOP-UPenn; not a named BraTS2024 post-treatment contributor; no patient-ID bridge','access':{'sums':False,'payload':False,'arrays':False,'label_values':False,'model_inference':False},'runner_sha256':prior.sha(Path(__file__))};cohortout.write_text(json.dumps(cohort,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':cohort['status'],'cases':33,'selected_files':66,'selected_bytes':cohort['bytes_selected'],'validation_unlabeled':6,'inventory_files':len(files),'inventory_sha256':prior.sha(invout),'cohort_sha256':prior.sha(cohortout)},indent=2,sort_keys=True))

if __name__=='__main__':main()
