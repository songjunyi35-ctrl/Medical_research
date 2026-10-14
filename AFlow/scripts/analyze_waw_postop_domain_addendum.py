#!/usr/bin/env python3
"""Add WAW postop31 summary to the archived five-cache V2 audit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'archive/reviews/2026-09-06_fresh_evidence_capacity/INPUT_DOMAIN_SHIFT_AUDIT_V2.json';INDEX=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/cache_postop31/index.json';IMAGES=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/cache_postop31/images.f32';EXPECTED={BASE:'a9fab522fb1a98a2a13053f7ecb4e9e5022703327bef604df108eade626f7a44',INDEX:'879479ba8af6ed9c3d1c1324b02e6d3c0c5710b82bf4baab1bc616e42a46f1b5',IMAGES:'01074a25c223c39304f1053977593ecdd41d9b25282479fc0641a3c0d24ec6e3'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def desc(v):
 a=np.asarray(v,float);q=np.quantile(a,[0,.1,.25,.5,.75,.9,1]);return {'mean':float(a.mean()),**{k:float(x) for k,x in zip(('min','p10','p25','median','p75','p90','max'),q)}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);ap.add_argument('--png',type=Path,required=True);a=ap.parse_args()
 if a.json.exists() or a.png.exists():raise FileExistsError('output')
 for p,d in EXPECTED.items():
  if sha(p)!=d:raise ValueError('SHA '+str(p))
 base=json.load(open(BASE));idx=json.load(open(INDEX));images=np.memmap(IMAGES,dtype=np.float32,mode='r',shape=(31,128,128,128));rows=[]
 for x in images:
  nz=x!=0;rows.append({'std_all':float(np.std(x,dtype=np.float64)),'nonzero_fraction':float(nz.mean()),'mean_abs_all':float(np.mean(np.abs(x),dtype=np.float64))})
 target=[v['resized_wt'] for v in idx['details'].values()];post={'cases':31,'target_voxels':desc(target),'input':{k:desc([x[k] for x in rows]) for k in rows[0]}}
 train=base['cohorts']['train972'];pre=base['cohorts']['waw152'];result={'schema':'aflow.waw_postop31.input_domain_addendum','schema_version':1,'status':'postop_cache_plus_archived_preop_summary_no_preop_cache_rebuild','cohorts':{'train972_archived_summary':train,'waw_pre152_archived_summary':pre,'waw_postop31_current_summary':post},'contrasts':{'postop_to_train_target_mean_ratio':post['target_voxels']['mean']/train['target_voxels']['mean'],'postop_to_train_target_median_ratio':post['target_voxels']['median']/train['target_voxels']['median'],'postop_to_preop_target_mean_ratio':post['target_voxels']['mean']/pre['target_voxels']['mean'],'postop_to_preop_target_median_ratio':post['target_voxels']['median']/pre['target_voxels']['median'],'postop_minus_preop_std_median':post['input']['std_all']['median']-pre['input']['std_all']['median'],'postop_minus_preop_nonzero_fraction_median':post['input']['nonzero_fraction']['median']-pre['input']['nonzero_fraction']['median']},'interpretation':'Postop and preop are patient-disjoint groups from one source; differences are descriptive and confounded by patient mix/stage. Low postop point hit is chiefly GT-RC attraction, not explained by target extent alone.','access':{'waw_postop_cache':True,'waw_preop_cache_rebuilt_or_accessed':False,'model_inference':False,'selection_or_tuning':False},'source_sha256':{str(p.relative_to(ROOT)):d for p,d in EXPECTED.items()},'analyzer_sha256':sha(Path(__file__))};a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 labels=['Train972','WAW-pre152','WAW-post31'];co=[train,pre,post];fig,axes=plt.subplots(1,3,figsize=(11.5,4),constrained_layout=True);vals=[[x['target_voxels']['median'] for x in co],[x['input']['std_all']['median'] for x in co],[x['input']['nonzero_fraction']['median'] for x in co]];titles=['WT median voxels','Cached all-voxel std median','Nonzero fraction median']
 for ax,v,t in zip(axes,vals,titles):ax.bar(labels,v,color=['#4c72b0','#ccb974','#64b5cd']);ax.set_title(t);ax.tick_params(axis='x',rotation=15)
 fig.suptitle('WAW postop31 domain addendum (patient-disjoint groups; descriptive)');fig.savefig(a.png,dpi=180);plt.close(fig);print(json.dumps({'contrasts':result['contrasts'],'json_sha256':sha(a.json),'png_sha256':sha(a.png)},indent=2))
if __name__=='__main__':main()
