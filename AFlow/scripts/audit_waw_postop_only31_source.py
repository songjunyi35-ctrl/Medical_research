#!/usr/bin/env python3
"""Freeze WAW postoperative patients not present in the evaluated preop152."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib,numpy as np
ROOT=Path(__file__).resolve().parents[1];META=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/metadata';PRE=ROOT/'archive/reviews/2026-09-06_waw_gliomai_source_audit/FROZEN_COHORT.json';PRE_SHA='25de15904d8db5e6d6e304bc73058dba4f43e7534ae0a533f4f06636613a263a';OFFICIAL={'2_2_postop_scans_coreg_meta.csv':'55ed0b77b9dbe9a604843f0924897138','3_2_postop_masks.zip':'e1d3548ee99cdcf11e5f444be20dea73','1_0_clinical_molecular_data.csv':'7dd4f66797b768853898e77de76cda44'}
def hashf(path,alg='sha256'):
 h=hashlib.new(alg)
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path,delimiter=','):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=delimiter))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();audit=a.output_dir/'SOURCE_AUDIT.json';cohort=a.output_dir/'FROZEN_COHORT.json'
 if audit.exists() or cohort.exists():raise FileExistsError('output')
 if hashf(PRE)!=PRE_SHA:raise ValueError('pre cohort SHA')
 files={}
 for name,md5 in OFFICIAL.items():
  path=META/name
  if hashf(path,'md5')!=md5:raise ValueError(name)
  files[name]={'official_md5':md5,'sha256':hashf(path),'bytes':path.stat().st_size}
 prepatients={x['publisher_patient_id'] for x in json.load(open(PRE))['cases']};postrows=rows(META/'2_2_postop_scans_coreg_meta.csv');t1c={Path(r['file_name']).name.split('_brain_')[0]:r for r in postrows if r['file_name'].endswith('_t1c.nii.gz')};masks={p.name.split('_tumor_mask_post')[0]:p for p in META.glob('postop_masks/**/*.nii*')}
 if len(postrows)!=640 or len(t1c)!=160 or len(masks)!=160 or set(t1c)!=set(masks):raise ValueError('final counts/pairs')
 postpatients={e.split('_')[0] for e in t1c};postonly=postpatients-prepatients;postonly_scans={e:r for e,r in t1c.items() if e.split('_')[0] in postonly};numeric={e:r for e,r in postonly_scans.items() if r['days_postop'].strip()};numeric_subjects={e.split('_')[0] for e in numeric};clinical={r['id']:r for r in rows(META/'1_0_clinical_molecular_data.csv',';')};chosen=[]
 for subject in sorted(numeric_subjects,key=int):
  day,e,row=min((float(r['days_postop']),e,r) for e,r in numeric.items() if e.split('_')[0]==subject);path=masks[e];mask=nib.load(path);arr=np.asanyarray(mask.dataobj);rounded=np.rint(arr);values=sorted(int(x) for x in np.unique(rounded));meta_aff=np.asarray(json.loads(row['affine']),float);err=float(np.max(np.abs(mask.affine-meta_aff)));wt=int(np.count_nonzero(np.isin(rounded,[1,2,3])))
  if mask.shape!=tuple(json.loads(row['shape'])) or err>1e-5 or not np.allclose(arr,rounded,rtol=0,atol=1e-5) or not set(values)<={0,1,2,3,4} or wt==0:raise ValueError(e)
  chosen.append({'case_id':'WAW-POST-'+e,'subject_id':'WAW-'+subject,'publisher_patient_id':subject,'exam_id':e,'days_postop':day,'image_basename':row['file_name'],'mask_path':str(path),'mask_sha256':hashf(path),'native_shape':list(mask.shape),'spacing_mm':[float(x) for x in nib.affines.voxel_sizes(mask.affine)],'affine_error_vs_t1c_metadata':err,'raw_values':values,'raw_wt_voxels_excluding_label4':wt,'who_grade':int(clinical[subject]['who_grade']),'scanner_name':row['ScannerName'],'field_strength':row['FieldStrengthLabel']})
 if len(chosen)!=31 or len({x['subject_id'] for x in chosen})!=31 or {x['publisher_patient_id'] for x in chosen}&prepatients:raise ValueError('cohort')
 c={'schema':'aflow.waw.postop_only31.frozen_cohort','schema_version':1,'status':'frozen_before_postop_image_archive_access','selection':'patients absent from evaluated preop152; numeric days_postop required; earliest numeric exam per patient','case_count':31,'subject_count':31,'target':'postoperative WT raw labels {1,2,3}; exclude resection bed label4','cases':chosen};a.output_dir.mkdir(parents=True,exist_ok=True);cohort.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
 result={'schema':'aflow.waw.postop_only31.source_audit','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'zenodo_record':22205631,'source_files':files,'release_description_vs_final':{'described_postop_exams':161,'final_postop_exams':160,'final_postop_patients':92,'strict_t1c_mask_pairs':160},'patient_difference':{'evaluated_preop_patients':len(prepatients),'postop_patients_overlapping_preop':len(postpatients&prepatients),'postop_only_patients':len(postonly),'postop_only_scans':len(postonly_scans),'postop_only_numeric_days_scans':len(numeric),'postop_only_numeric_days_subjects':len(numeric_subjects),'blank_days_scans':len(postonly_scans)-len(numeric),'blank_days_only_subjects':sorted(postonly-numeric_subjects,key=int),'later_numeric_scans_excluded_by_earliest_per_subject':len(numeric)-len(chosen)},'frozen':{'cases':31,'subjects':31,'days_postop_counts':dict(Counter(str(x['days_postop']) for x in chosen)),'days_postop_min':min(x['days_postop'] for x in chosen),'days_postop_max':max(x['days_postop'] for x in chosen),'grade_counts':dict(Counter(str(x['who_grade']) for x in chosen)),'raw_label_case_presence':dict(Counter(str(v) for x in chosen for v in x['raw_values'])),'wt_voxels_min_median_max':[min(x['raw_wt_voxels_excluding_label4'] for x in chosen),float(np.median([x['raw_wt_voxels_excluding_label4'] for x in chosen])),max(x['raw_wt_voxels_excluding_label4'] for x in chosen)],'preop_patient_overlap':0,'all_geometry_exact':True},'evidence_scope':'new patients relative to revealed WAW preop152; public postoperative earliest numeric day0-22; institution-disjoint/no external patient bridge; not hidden blind','model_or_large_image_access':{'postop_coreg_archive':False,'cache':False,'checkpoint':False,'model_forwards':0,'performance':False},'cohort_sha256':hashf(cohort),'analyzer_sha256':hashf(Path(__file__))};audit.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'cases':31,'cohort_sha256':result['cohort_sha256'],'audit_sha256':hashf(audit)},indent=2))
if __name__=='__main__':main()
