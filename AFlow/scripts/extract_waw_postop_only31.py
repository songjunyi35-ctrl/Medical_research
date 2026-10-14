#!/usr/bin/env python3
"""Extract only frozen WAW postop-only31 T1c members from the official zip."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[1];RECORD=ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel';COHORT=ROOT/'archive/reviews/2026-09-06_waw_postop_only31_source_audit/FROZEN_COHORT.json';ARCHIVE=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/2_2_postop_scans_coreg.zip';OUTPUT=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/selected_postop31';COHORT_SHA='965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180';PRE_SHA='0fffdc57e71febe972fae1c977939a01d50a18ad06b8492193f9c7e2b8d46c77';ARCHIVE_SHA='6236c97c81a3c1a02543d5bf48425946a481861f9dde96bb19b6482f69508080'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 if sha(COHORT)!=COHORT_SHA or sha(RECORD/'SOURCE_PREACCESS.json')!=PRE_SHA or sha(ARCHIVE)!=ARCHIVE_SHA:raise ValueError('SHA')
 rows=json.load(open(COHORT))['cases'];expected={r['image_basename']:r for r in rows}
 with ZipFile(ARCHIVE) as z:
  members={Path(x).name:x for x in z.namelist() if x.endswith('_brain_t1c.nii.gz')}
  if len(members)!=160 or not set(expected)<set(members):raise ValueError('member inventory')
  OUTPUT.mkdir();out=[]
  for name,row in expected.items():
   target=OUTPUT/row['publisher_patient_id']/name;target.parent.mkdir()
   with z.open(members[name]) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,8<<20)
   out.append({'case_id':row['case_id'],'member':members[name],'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
 if len(out)!=31:raise ValueError('count')
 receipt={'schema':'aflow.waw_postop_only31.extraction','status':'passed','cases':31,'archive_sha256':ARCHIVE_SHA,'cohort_sha256':COHORT_SHA,'files':sorted(out,key=lambda x:x['case_id']),'extractor_sha256':sha(Path(__file__)),'model_inference':False};path=RECORD/'EXTRACTION_RECEIPT.json'
 with path.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'cases':31,'bytes':sum(x['bytes'] for x in out),'receipt_sha256':sha(path)},indent=2))
if __name__=='__main__':main()
