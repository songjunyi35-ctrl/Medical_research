#!/usr/bin/env python3
"""Inventory only the 39 mapped CPTAC BraTS2021 NIfTI cases, no payload."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import scripts.audit_brats_africa_source_20260906 as api


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"archive/reviews/2026-09-06_cptac_brats2021_source_audit"
PAGE="https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/"
MAPPING="https://www.cancerimagingarchive.net/wp-content/uploads/BraTS2021_MappingToTCIA.xlsx"
PACKAGE_ID=636

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())

def parse_mapping(data:bytes):
 ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 import io
 with ZipFile(io.BytesIO(data)) as z:
  shared=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
  root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));raw=[]
  for row in root.findall('.//m:row',ns):
   values={}
   for cell in row.findall('m:c',ns):
    value=cell.find('m:v',ns);text='' if value is None else value.text
    if cell.get('t')=='s' and text:text=shared[int(text)]
    values[cell.get('r')[0]]=text
   raw.append(values)
 header=raw[0];rows=[{header.get(k,k):v for k,v in item.items()} for item in raw[1:]]
 return [r for r in rows if r.get('Data Collection (as on TCIA+additional)')=='CPTAC-GBM']

def main():
 inventory_path=OUT/'OFFICIAL_CPTAC39_INVENTORY.json';cohort_path=OUT/'FROZEN_COHORT.json';mapping_path=OUT/'source_metadata/BraTS2021_MappingToTCIA.xlsx'
 if any(x.exists() for x in (inventory_path,cohort_path,mapping_path)):raise FileExistsError('output')
 api.PAGE=PAGE;api.DATAINFO=MAPPING;api.PACKAGE_ID=PACKAGE_ID
 bearer,page,mapping,package=api.authorize();rows=parse_mapping(mapping)
 if len(rows)!=39 or len({r['PatientID on TCIA Radiology Portal'] for r in rows})!=39 or len({r['BraTS2021 ID'] for r in rows})!=39:raise ValueError('mapping identity')
 role_counts={role:sum(r.get('Segmentation (Task 1) Cohort')==role for r in rows) for role in ('Training','Validation')}
 if role_counts!={'Training':33,'Validation':6}:raise ValueError('roles')
 role_dir={'Training':'BraTS2021_TrainingSet','Validation':'BraTS2021_ValidationSet'}
 expected_dirs={f"/RSNA-ASNR-MICCAI-BraTS-2021/{role_dir[r['Segmentation (Task 1) Cohort']]}/CPTAC-GBM/{r['BraTS2021 ID']}":r for r in rows}
 for role,expected in role_counts.items():
  root=f"/RSNA-ASNR-MICCAI-BraTS-2021/{role_dir[role]}/CPTAC-GBM";items,pages=api.list_directory(root,bearer)
  got={x['path'] for x in items};want={x for x,r in expected_dirs.items() if r['Segmentation (Task 1) Cohort']==role}
  if pages!=1 or got!=want or any(x.get('type')!='directory' for x in items):raise ValueError('role directories')
 def read(path):return path,*api.list_directory(path,bearer)
 listings=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for path,items,pages in pool.map(read,sorted(expected_dirs)):listings.append((path,items,pages))
 files=[];cases=[]
 for path,items,pages in sorted(listings):
  row=expected_dirs[path];bid=row['BraTS2021 ID'];compact=[api.compact(x) for x in items];names={x['basename'] for x in compact};expected={f'{bid}_{suffix}.nii.gz' for suffix in ('flair','seg','t1','t1ce','t2')}
  if pages!=1 or names!=expected or len(compact)!=5 or any(x['type']!='symbolic_link' or x['target_type']!='file' or int(x['size'] or 0)<=0 for x in compact):raise ValueError('case file contract '+bid)
  by={x['basename']:x for x in compact};files.extend(compact);cases.append({'brats2021_id':bid,'cptac_patient_id':row['PatientID on TCIA Radiology Portal'],'role':row['Segmentation (Task 1) Cohort'],'study_date':row.get('Study date (m/d/yyyy) per PatientID'),'t1ce':by[f'{bid}_t1ce.nii.gz'],'seg':by[f'{bid}_seg.nii.gz']})
 files.sort(key=lambda x:x['path']);cases.sort(key=lambda x:x['brats2021_id']);OUT.mkdir(parents=True,exist_ok=True);mapping_path.parent.mkdir(parents=True,exist_ok=True);mapping_path.write_bytes(mapping)
 inventory={'schema':'aflow.cptac_brats2021.official_inventory','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'package_id':PACKAGE_ID,'files':files};inventory_path.write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
 cohort={'schema':'aflow.cptac_brats2021.frozen_cohort','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_sums_payload_arrays_or_model','cases':39,'subjects':39,'roles':role_counts,'hidden_test':False,'files_selected':78,'bytes_selected':sum(int(x[k]['size']) for x in cases for k in ('t1ce','seg')),'cases_detail':cases,'excluded_other_brats2021_cases':1441,'source':{'analysis_doi':'10.7937/JC8X-9874','package_id':PACKAGE_ID,'mapping_sha256':sha_bytes(mapping),'page_sha256':sha_bytes(page),'inventory_sha256':sha(inventory_path)},'institution':'CPTAC-GBM/CHOP-UPenn source; not a named BraTS2024 post-treatment contributor; no patient-ID bridge to BraTS2024','access':{'sums':False,'payload':False,'arrays':False,'label_values':False,'model_inference':False},'runner_sha256':sha(Path(__file__))};cohort_path.write_text(json.dumps(cohort,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':cohort['status'],'cases':39,'roles':role_counts,'inventory_files':len(files),'inventory_bytes':sum(int(x['size']) for x in files),'selected_files':78,'selected_bytes':cohort['bytes_selected'],'inventory_sha256':sha(inventory_path),'cohort_sha256':sha(cohort_path),'mapping_sha256':sha_bytes(mapping),'page_sha256':sha_bytes(page)},indent=2,sort_keys=True))

if __name__=='__main__':main()
