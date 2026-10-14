#!/usr/bin/env python3
"""Two-process, no-interpolation shared-orientation bridge for MOTUM-HGG29."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import nibabel as nib,numpy as np
ROOT=Path(__file__).resolve().parents[1];RECORD=ROOT/'archive/research_records/2026-09-06_motum_hgg29_shared_orientation_bridge';COHORT=ROOT/'archive/reviews/2026-09-06_motum_hgg_source_audit/FROZEN_HGG29_COHORT.json';AUDIT=ROOT/'archive/reviews/2026-09-06_motum_hgg_source_audit/SOURCE_AUDIT.json';OLD=ROOT/'archive/research_records/2026-09-06_motum_hgg29_frozen_panel/CACHE_PROBE_FAILURE.json'
IDEA_SHA='6645975309475795eac231944b48f0365d0334c76b2f72b27bbab887ca2a7ec9';PREREG_SHA='bc6c188c736838c14a31ccbe0cec166c345b03debd9954f25d45708b01f60b78';COHORT_SHA='f49e8f888b0fce8b7f34da1f9eadb49039d10aaed8e0538008f215df1b953379';AUDIT_SHA='69d59768cd54ed526aa1cc8b0f66e31884ea4aaad171a1a16bc17c4488412932';OLD_SHA='394f02162a19b46516deaee1051efa4b7bb821ab0c0a7d28cf6bb79d4e27baa3'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checked(path,expected):
 if sha(path)!=expected:raise ValueError('SHA '+str(path))
def docs():checked(RECORD/'IDEA.md',IDEA_SHA);checked(RECORD/'PREREG.md',PREREG_SHA);checked(COHORT,COHORT_SHA);checked(AUDIT,AUDIT_SHA);checked(OLD,OLD_SHA)
def write_new(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def worker(output):
 docs();rows=json.load(open(COHORT))['cases'];target=nib.orientations.axcodes2ornt(('R','A','S'));cases=[]
 for row in rows:
  ims=[nib.load(row[k]) for k in ('image_path','flair_mask_path','t1ce_mask_path')]
  if not (ims[0].shape==ims[1].shape==ims[2].shape):raise ValueError('shape')
  native=max(float(np.max(np.abs(ims[0].affine-x.affine))) for x in ims[1:])
  if native>5e-7:raise ValueError('native affine')
  orientations=[nib.orientations.io_orientation(x.affine) for x in ims]
  if not all(np.array_equal(orientations[0],x) for x in orientations[1:]):raise ValueError('orientation')
  transform=nib.orientations.ornt_transform(orientations[0],target);raw=[np.asanyarray(x.dataobj) for x in ims];shared=[np.ascontiguousarray(nib.orientations.apply_orientation(x,transform)) for x in raw];independent=[np.ascontiguousarray(np.asanyarray(nib.as_closest_canonical(x).dataobj)) for x in ims]
  if not all(np.array_equal(a,b) for a,b in zip(shared,independent)):raise ValueError('canonical array parity')
  fv=sorted(int(x) for x in np.unique(shared[1]));tv=sorted(int(x) for x in np.unique(shared[2]));wt=(shared[1]>0)|(shared[2]>0)
  if not set(fv)<={0,1} or not set(tv)<={0,1,2} or not wt.any():raise ValueError('labels')
  affine=ims[0].affine@nib.orientations.inv_ornt_aff(transform,ims[0].shape)
  cases.append({'case_id':row['case_id'],'native_affine_max_abs':native,'orientation':orientations[0].tolist(),'transform':transform.tolist(),'shared_shape':list(shared[0].shape),'shared_affine_sha256':arrsha(affine.astype(np.float64)),'image_array_sha256':arrsha(shared[0]),'flair_mask_array_sha256':arrsha(shared[1]),'t1ce_mask_array_sha256':arrsha(shared[2]),'wt_union_array_sha256':arrsha(wt.astype(np.uint8)),'wt_voxels':int(wt.sum()),'arrays_equal_independent_canonical':[True,True,True]})
 report={'schema':'aflow.motum_hgg29.shared_orientation_copy','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','cases':cases,'case_count':len(cases),'native_affine_limit':5e-7,'shared_transform_from_image_only':True,'interpolation':False,'n4_or_cache':False,'checkpoint_loaded':False,'model_forwards':0,'performance_metrics':False,'runner_sha256':sha(Path(__file__))};write_new(output,report);print(json.dumps({'status':'passed','cases':len(cases),'output_sha256':sha(output)},indent=2))
def finalize(a,b,output):
 docs();ra=json.load(open(a));rb=json.load(open(b))
 if ra['status']!='passed' or rb['status']!='passed' or ra['case_count']!=29 or rb['case_count']!=29:raise ValueError('copy contract')
 if ra['cases']!=rb['cases']:raise ValueError('copy mismatch')
 result={'schema':'aflow.motum_hgg29.shared_orientation_bridge','schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','cases':29,'native_shape_exact':29,'native_affine_within_5e_7':29,'identical_discrete_orientation':29,'shared_arrays_equal_independent_canonical':29,'copy_a_b_case_payload_exact':True,'interpolation':False,'shared_affine_from_image_only':True,'copy_a_sha256':sha(a),'copy_b_sha256':sha(b),'cohort_sha256':COHORT_SHA,'old_failure_sha256':OLD_SHA,'cache_created':False,'checkpoint_loaded':False,'model_forwards':0,'performance_metrics':False,'runner_sha256':sha(Path(__file__))};write_new(output,result);print(json.dumps({**result,'result_sha256':sha(output)},indent=2))
def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True);w=sub.add_parser('worker');w.add_argument('--output',type=Path,required=True);f=sub.add_parser('finalize');f.add_argument('--copy-a',type=Path,required=True);f.add_argument('--copy-b',type=Path,required=True);f.add_argument('--output',type=Path,required=True);a=ap.parse_args();worker(a.output) if a.action=='worker' else finalize(a.copy_a,a.copy_b,a.output)
if __name__=='__main__':main()
