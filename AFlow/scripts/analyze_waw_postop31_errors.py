#!/usr/bin/env python3
"""Post-hoc WAW postop31 point-error and saved-count audit without inference."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib,numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];REC=ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel';RAW=REC/'RAW_WAW_POSTOP31.json';COHORT=ROOT/'archive/reviews/2026-09-06_waw_postop_only31_source_audit/FROZEN_COHORT.json';INDEX=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/cache_postop31/index.json';LABELS=ROOT/'workspace/BraTS/waw_gliomai_source_20260906/cache_postop31/labels.u8';EXPECTED={RAW:'72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98',COHORT:'965b2fe0f331d782dbb6bb136d6089c1181419a9a75dbd8f198cbeff33284180',INDEX:'879479ba8af6ed9c3d1c1324b02e6d3c0c5710b82bf4baab1bc616e42a46f1b5',LABELS:'98d96cd778f62ad941778f578df0b43e1c84a6712b15513ee11c0e11357968c0'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def resize(a):return F.interpolate(torch.from_numpy(np.ascontiguousarray(a)).float()[None,None],size=(128,128,128),mode='nearest-exact')[0,0].numpy()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);ap.add_argument('--png',type=Path,required=True);a=ap.parse_args()
 if a.json.exists() or a.png.exists():raise FileExistsError('output')
 for p,d in EXPECTED.items():
  if sha(p)!=d:raise ValueError('SHA '+str(p))
 raw=json.load(open(RAW));co=json.load(open(COHORT));idx=json.load(open(INDEX));meta={x['case_id']:x for x in co['cases']};cache=np.memmap(LABELS,dtype=np.uint8,mode='r',shape=(31,128,128,128));resized={}
 for i,case in enumerate(idx['case_ids']):
  m=meta[case];arr=np.rint(np.asanyarray(nib.as_closest_canonical(nib.load(m['mask_path'],mmap=True)).dataobj));g=idx['details'][case]['geometry'];crop=tuple(slice(o,o+e) for o,e in zip(g['origin'],g['extent']));view=arr[crop];wt=resize(np.isin(view,[1,2,3])).astype(bool);rc=resize(view==4).astype(bool)
  if not np.array_equal(wt,cache[i].astype(bool)) or np.any(wt&rc):raise ValueError('label reconstruction '+case)
  resized[case]={'wt':wt,'rc':rc,'rc_voxels':int(rc.sum()),'wt_voxels':int(wt.sum())}
 models={'full_identity':raw['full972_kd168']['identity']['per_case'],'full_flip8':raw['full972_kd168']['flip8']['per_case'],'gcsam_identity':raw['gcsam184']['identity']['per_case']};summary={}
 for name,rows in models.items():
  categories=[];precision=[];recall=[];ratios=[];details=[]
  for row in rows:
   point=row['automatic_point_flat_index'];case=row['case_id'];r=resized[case];cat='wt' if r['wt'].flat[point] else ('resection_cavity' if r['rc'].flat[point] else 'other_background');categories.append(cat);pred=row['predicted_wt_voxels'];inter=row['intersection_voxels'];target=row['target_wt_voxels'];precision.append(inter/pred if pred else 0);recall.append(inter/target);ratios.append(pred/target);details.append({'case_id':case,'days_postop':meta[case]['days_postop'],'dice':row['locator_dsc'],'point_category':cat,'target_wt':target,'rc_voxels':r['rc_voxels'],'prediction_target_ratio':pred/target})
  summary[name]={'point_categories':{x:categories.count(x) for x in ('wt','resection_cavity','other_background')},'precision_mean':statistics.fmean(precision),'recall_mean':statistics.fmean(recall),'prediction_target_ratio_mean':statistics.fmean(ratios),'dice_mean_day0_1':statistics.fmean(x['dice'] for x in details if x['days_postop']<=1),'dice_mean_day_gt1':statistics.fmean(x['dice'] for x in details if x['days_postop']>1),'details':details}
 result={'schema':'aflow.waw_postop_only31.posthoc_errors','schema_version':1,'status':'posthoc_saved_logits_index_plus_label_reconstruction_no_inference_or_selection','cases':31,'cache_wt_reconstruction_exact':31,'models':summary,'interpretation':'Most point misses land in ordinary background rather than GT resection cavity if counts show so; any category is descriptive only because GT RC is unavailable at deployment. Day, RC, precision and recall are confounded and cannot drive tuning.','forbidden_use':'no GT-RC avoidance, day-based routing, threshold/postprocessing, case/timepoint selection, or retest','new_model_inference':False,'checkpoint_or_image_access':False,'source_sha256':{str(p.relative_to(ROOT)):d for p,d in EXPECTED.items()},'analyzer_sha256':sha(Path(__file__))};a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 labels=list(models);cats=('wt','resection_cavity','other_background');colors=('#55a868','#c44e52','#8172b2');fig,axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True);bottom=np.zeros(3)
 for cat,color in zip(cats,colors):
  y=np.array([summary[x]['point_categories'][cat] for x in labels]);axes[0].bar(np.arange(3),y,bottom=bottom,label=cat.replace('_',' '),color=color);bottom+=y
 axes[0].set(xticks=np.arange(3),xticklabels=['Full 1x','Full Flip8','GCSAM 1x'],ylabel='Cases',title='Argmax point category');axes[0].legend(frameon=False,fontsize=8)
 x=np.arange(2);width=.24
 for i,(name,color) in enumerate(zip(labels,('#4c72b0','#dd8452','#55a868'))):axes[1].bar(x+(i-1)*width,[summary[name]['precision_mean'],summary[name]['recall_mean']],width,label=name,color=color)
 axes[1].set(xticks=x,xticklabels=['Precision','Recall'],ylim=(0,1),title='Saved-count overlap components');axes[1].legend(frameon=False,fontsize=8);fig.suptitle('WAW early-postop31 post-hoc audit (no tuning)');fig.savefig(a.png,dpi=180);plt.close(fig);print(json.dumps({'json_sha256':sha(a.json),'png_sha256':sha(a.png),'point_categories':{k:v['point_categories'] for k,v in summary.items()}},indent=2))
if __name__=='__main__':main()
