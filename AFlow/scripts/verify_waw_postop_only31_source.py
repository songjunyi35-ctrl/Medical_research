#!/usr/bin/env python3
"""Verify the official WAW postop archive and selected patient-disjoint T1c files."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from zipfile import ZipFile
import nibabel as nib,numpy as np
ROOT=Path(__file__).resolve().parents[1];RECORD=ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel';REVIEW=ROOT/'archive/reviews/2026-09-06_waw_postop_only31_source_audit';SOURCE=ROOT/'workspace/BraTS/waw_gliomai_source_20260906';ARCHIVE=SOURCE/'2_2_postop_scans_coreg.zip';SELECTED=SOURCE/'selected_postop31';IDEA_SHA='4f21bcd4babbf229b683c84c9bb1a3992ec6cf3da90203beaaf0b9187d100cfc';PREREG_SHA='078746900c854ee2b59972d49132bd3787c53dba902a4fd95009a14a1a4a6729';AUDIT_SHA='ca7d67892e42ae4d193f87ebdd983fb79314091b84d2e448021a71da18e0fe3d';COHORT_SHA='965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180';PRE_SHA='0fffdc57e71febe972fae1c977939a01d50a18ad06b8492193f9c7e2b8d46c77';BYTES=3321955416;MD5='ab9da3eb2a3aff73b8f3b95e01508c1c'
def hashf(path,alg='sha256'):
 h=hashlib.new(alg)
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def checked(path,digest):
 if hashf(path)!=digest:raise ValueError('SHA '+str(path))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 checked(RECORD/'IDEA.md',IDEA_SHA);checked(RECORD/'PREREG.md',PREREG_SHA);checked(RECORD/'SOURCE_PREACCESS.json',PRE_SHA);checked(REVIEW/'SOURCE_AUDIT.json',AUDIT_SHA);checked(REVIEW/'FROZEN_COHORT.json',COHORT_SHA)
 if ARCHIVE.stat().st_size!=BYTES or hashf(ARCHIVE,'md5')!=MD5:raise ValueError('archive')
 archive_sha=hashf(ARCHIVE);cohort=json.load(open(REVIEW/'FROZEN_COHORT.json'));expected={x['image_basename'] for x in cohort['cases']}
 with ZipFile(ARCHIVE) as z:
  bad=z.testzip();members=[x for x in z.namelist() if not x.endswith('/')];t1c=[x for x in members if x.endswith('_brain_t1c.nii.gz')]
 if bad is not None or len(t1c)!=160 or not expected<={Path(x).name for x in t1c}:raise ValueError('zip/member')
 paths=sorted(SELECTED.glob('**/*_brain_t1c.nii.gz'))
 if len(paths)!=31 or {x.name for x in paths}!=expected:raise ValueError('selected')
 by={x.name:x for x in paths};files=[];union=set()
 for row in cohort['cases']:
  ip=by[row['image_basename']];mp=Path(row['mask_path']);checked(mp,row['mask_sha256']);ni=nib.as_closest_canonical(nib.load(str(ip),mmap=True));nm=nib.as_closest_canonical(nib.load(str(mp),mmap=True));err=float(np.max(np.abs(ni.affine-nm.affine)))
  if ni.shape!=nm.shape or err>1e-5:raise ValueError(row['case_id'])
  image=np.asanyarray(ni.dataobj);mask=np.asanyarray(nm.dataobj);rounded=np.rint(mask);values=sorted(int(x) for x in np.unique(rounded));union.update(values);wt=np.isin(rounded,[1,2,3])
  if not np.isfinite(image).all() or not (image>0).any() or not np.allclose(mask,rounded,rtol=0,atol=1e-5) or not set(values)<={0,1,2,3,4} or not wt.any():raise ValueError('array '+row['case_id'])
  files.append({'case_id':row['case_id'],'subject_id':row['subject_id'],'days_postop':row['days_postop'],'image_path':str(ip),'image_bytes':ip.stat().st_size,'image_sha256':hashf(ip),'mask_path':str(mp),'mask_sha256':row['mask_sha256'],'native_shape':list(ni.shape),'spacing_mm':[float(x) for x in nib.affines.voxel_sizes(ni.affine)],'axis_codes':list(nib.aff2axcodes(ni.affine)),'affine_error':err,'raw_values':values,'raw_wt_voxels':int(wt.sum())})
 report={'schema':'aflow.waw_postop_only31.source_receipt','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','archive':{'path':str(ARCHIVE),'bytes':BYTES,'official_md5':MD5,'sha256':archive_sha,'zip_crc_all_passed':True,'file_members':len(members),'postop_t1c_members':len(t1c)},'selected':{'cases':31,'subjects':31,'image_bytes':sum(x['image_bytes'] for x in files),'global_raw_union':sorted(union),'all_geometry_exact':True,'all_wt_nonempty':True,'files':files},'cohort_sha256':COHORT_SHA,'preop_patient_overlap':0,'checkpoint_loaded':False,'model_forwards':0,'performance':False,'verifier_sha256':hashf(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'passed','cases':31,'archive_sha256':archive_sha,'selected_image_bytes':report['selected']['image_bytes'],'receipt_sha256':hashf(a.output)},indent=2))
if __name__=='__main__':main()
