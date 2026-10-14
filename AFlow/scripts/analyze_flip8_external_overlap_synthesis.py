#!/usr/bin/env python3
"""Synthesize saved Full Flip8-vs-identity overlap components externally."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SOURCES={'Africa95':(ROOT/'archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json','c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3'),'CPTAC33':(ROOT/'archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json','4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd'),'WAW-pre152':(ROOT/'archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json','ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91'),'WAW-post31':(ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json','72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98')};SAMPLES=10000;SEED=20261621
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ci(v,seed):
 x=np.asarray(v,float);d=np.random.default_rng(seed).integers(0,len(x),size=(SAMPLES,len(x)));return [float(z) for z in np.quantile(x[d].mean(1),(.025,.975))]
def vals(rows):
 out={'precision':[],'recall':[],'predicted':[],'ratio':[]}
 for x in rows:
  pred=x['predicted_wt_voxels'];target=x['target_wt_voxels'];inter=x['intersection_voxels'];out['precision'].append(inter/pred if pred else 1);out['recall'].append(inter/target);out['predicted'].append(pred);out['ratio'].append(pred/target)
 return {k:np.asarray(v,float) for k,v in out.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);ap.add_argument('--png',type=Path,required=True);a=ap.parse_args()
 if a.json.exists() or a.png.exists():raise FileExistsError('output')
 rows=[]
 for i,(name,(path,digest)) in enumerate(SOURCES.items()):
  if sha(path)!=digest:raise ValueError('SHA')
  d=json.load(open(path));identity=d['full972_kd168']['identity']['per_case'];flip=d['full972_kd168']['flip8']['per_case']
  if [x['case_id'] for x in identity]!=[x['case_id'] for x in flip]:raise ValueError('order')
  f=vals(identity);g=vals(flip);delta={k:g[k]-f[k] for k in f};rows.append({'cohort':name,'cases':len(identity),'identity':{k:float(v.mean()) for k,v in f.items()},'flip8':{k:float(v.mean()) for k,v in g.items()},'flip8_minus_identity':{k:float(v.mean()) for k,v in delta.items()},'paired_95ci_posthoc':{k:ci(v,SEED+i*10+j) for j,(k,v) in enumerate(delta.items())},'point_delta':sum(x['automatic_point_hits_gt'] for x in flip)-sum(x['automatic_point_hits_gt'] for x in identity)})
 result={'schema':'aflow.flip8.external_overlap_synthesis','schema_version':1,'status':'posthoc_saved_counts_separate_cohorts_no_pooling_or_selection','precision_empty_prediction_convention':1,'cohorts':rows,'direction_summary':{'precision_positive':sum(x['flip8_minus_identity']['precision']>0 for x in rows),'recall_positive':sum(x['flip8_minus_identity']['recall']>0 for x in rows),'predicted_volume_positive':sum(x['flip8_minus_identity']['predicted']>0 for x in rows),'point_negative':sum(x['point_delta']<0 for x in rows),'cohorts':len(rows)},'interpretation':'Flip8 has higher precision point estimates in all four external cohorts, while recall is near zero/mixed and predicted volume is usually lower. This supports a conservative false-positive-reduction tendency, not universal Dice significance or free benefit.','bootstrap':{'samples':SAMPLES,'seed':SEED,'posthoc':True},'access':{'new_model_inference':False,'cohort_reaccess':False,'selection_or_tuning':False},'forbidden_use':'no view/weight/threshold search, pooling, router, postprocessing, or retest','source_sha256':{k:v[1] for k,v in SOURCES.items()},'analyzer_sha256':sha(Path(__file__))};a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 x=np.arange(len(rows));fig,axes=plt.subplots(1,3,figsize=(14.5,4.2),constrained_layout=True)
 for ax,key,title in ((axes[0],'precision','Precision delta'),(axes[1],'recall','Recall delta'),(axes[2],'predicted','Predicted voxel delta')):
  y=np.array([r['flip8_minus_identity'][key] for r in rows]);cis=np.array([r['paired_95ci_posthoc'][key] for r in rows]);ax.axhline(0,color='black',lw=1);ax.errorbar(x,y,yerr=np.vstack((y-cis[:,0],cis[:,1]-y)),fmt='o',capsize=4,color='#dd8452');ax.set(xticks=x,xticklabels=[r['cohort'] for r in rows],title=title);ax.tick_params(axis='x',rotation=20)
 fig.suptitle('Full Flip8 − identity external overlap components (post-hoc; no pooling)');fig.savefig(a.png,dpi=180);plt.close(fig);print(json.dumps({'direction_summary':result['direction_summary'],'json_sha256':sha(a.json),'png_sha256':sha(a.png)},indent=2))
if __name__=='__main__':main()
