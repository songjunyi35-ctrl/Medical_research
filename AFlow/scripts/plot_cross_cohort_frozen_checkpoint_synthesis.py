#!/usr/bin/env python3
"""Plot saved 1x Full972-vs-GCSAM evidence without new inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 "internal":(ROOT/"workspace/BraTS/sammed3d_t1ce_wt/gcsam_full972_stability_20260831/GCSAM_INNER_E184_TEST270.json","d0e2f48e1f499d65adddb3dbe30758caad27768bb75f737ea2cfe4a877c335d5"),
 "ucsd":(ROOT/"archive/research_records/2026-09-06_ucsd_official_float_source_correction/RAW_OFFICIAL_FLOAT.json","a1155914b16b648f97f11a7d7a20cb27096d7eed9f9662db73f4905860da04d2"),
 "africa":(ROOT/"archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json","c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3"),
 "cptac":(ROOT/"archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json","4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd"),
 "waw":(ROOT/"archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json","ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91"),
 "waw_postop":(ROOT/"archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json","72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98")}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--json',type=Path,required=True);ap.add_argument('--png',type=Path,required=True);ap.add_argument('--pdf',type=Path,required=True);a=ap.parse_args()
 if any(x.exists() for x in (a.json,a.png,a.pdf)):raise FileExistsError('output')
 a.json.parent.mkdir(parents=True,exist_ok=True)
 if a.png.parent!=a.json.parent or a.pdf.parent!=a.json.parent:raise ValueError('outputs must share one directory')
 # The internal source is checked against the canonical leaderboard report hash indirectly below because its raw hash is historical.
 full_report=ROOT/'workspace/BraTS/sammed3d_t1ce_wt/voxel_kd_fulltrajectory_972_20260824/FULLTRAJECTORY_TEST270.json'
 if any(sha(path)!=expected for path,expected in SOURCES.values()) or sha(full_report)!='9827a75b88dc9239902c7e473a7d75e2d6058f4f515bfd2b4d93003ba7bd34cd':raise ValueError('source SHA')
 internal=json.load(open(SOURCES['internal'][0]));ucsd=json.load(open(SOURCES['ucsd'][0]));africa=json.load(open(SOURCES['africa'][0]));cptac=json.load(open(SOURCES['cptac'][0]));waw=json.load(open(SOURCES['waw'][0]));waw_postop=json.load(open(SOURCES['waw_postop'][0]))
 rows=[
  {'cohort':'Reused internal test270','scope':'270 scans/119 subjects; reused, not blind/external','full_scan':internal['current_internal_best']['locator_dsc_mean'],'full_ci':json.load(open(full_report))['candidate_bootstrap']['scan_dsc_mean_95ci'],'full_points':262,'point_denominator':270,'gcsam_scan':internal['gcsam']['locator_dsc_mean'],'gcsam_ci':internal['gcsam_bootstrap']['scan_dsc_mean_95ci'],'gcsam_points':252,'delta':internal['gcsam_minus_current_internal_best']['ema_minus_raw']['scan'],'delta_ci':internal['gcsam_minus_current_internal_best']['ema_minus_raw']['scan_95ci']},
  {'cohort':'UCSD75 official-float','scope':'same-cohort source correction; reused public external','full_scan':ucsd['full972_kd168']['summary']['locator_dsc_mean'],'full_ci':ucsd['full972_kd168']['bootstrap']['scan_dsc_mean_95ci'],'full_points':70,'point_denominator':75,'gcsam_scan':ucsd['gcsam184']['summary']['locator_dsc_mean'],'gcsam_ci':ucsd['gcsam184']['bootstrap']['scan_dsc_mean_95ci'],'gcsam_points':70,'delta':-ucsd['official_float_model_comparison_full972_minus_gcsam']['primary_minus_secondary']['scan_dsc_mean'],'delta_ci':[-ucsd['official_float_model_comparison_full972_minus_gcsam']['paired_scan_dsc_mean_95ci'][1],-ucsd['official_float_model_comparison_full972_minus_gcsam']['paired_scan_dsc_mean_95ci'][0]]},
  {'cohort':'BraTS-Africa95','scope':'public post-challenge; four centers; institution-disjoint/no patient bridge','full_scan':africa['full972_kd168']['identity']['locator_dsc_mean'],'full_ci':africa['full972_kd168']['identity_bootstrap']['scan_dsc_mean_95ci'],'full_points':87,'point_denominator':95,'gcsam_scan':africa['gcsam184']['identity']['locator_dsc_mean'],'gcsam_ci':africa['gcsam184']['identity_bootstrap']['scan_dsc_mean_95ci'],'gcsam_points':84,'delta':africa['comparisons']['gcsam_identity_minus_full_identity']['gcsam_minus_full']['scan_dice'],'delta_ci':africa['comparisons']['gcsam_identity_minus_full_identity']['paired_scan_dice_95ci']},
  {'cohort':'CPTAC33 public Training','scope':'public labels external to this project; Africa-informed confirmation, not hidden','full_scan':cptac['full972_kd168']['identity']['locator_dsc_mean'],'full_ci':cptac['full972_kd168']['identity_bootstrap']['scan_dsc_mean_95ci'],'full_points':33,'point_denominator':33,'gcsam_scan':cptac['gcsam184']['identity']['locator_dsc_mean'],'gcsam_ci':cptac['gcsam184']['identity_bootstrap']['scan_dsc_mean_95ci'],'gcsam_points':33,'delta':cptac['comparisons']['gcsam_identity_minus_full_identity']['gcsam_minus_full']['scan_dice'],'delta_ci':cptac['comparisons']['gcsam_identity_minus_full_identity']['paired_scan_dice_95ci']},
  {'cohort':'WAW-GliomAI152','scope':'public preoperative final T1c-mask intersection; institution-disjoint/no patient bridge','full_scan':waw['full972_kd168']['identity']['locator_dsc_mean'],'full_ci':waw['full972_kd168']['identity_bootstrap']['scan_dsc_mean_95ci'],'full_points':141,'point_denominator':152,'gcsam_scan':waw['gcsam184']['identity']['locator_dsc_mean'],'gcsam_ci':waw['gcsam184']['identity_bootstrap']['scan_dsc_mean_95ci'],'gcsam_points':140,'delta':waw['comparisons']['gcsam_identity_minus_full_identity']['gcsam_minus_full']['scan_dice'],'delta_ci':waw['comparisons']['gcsam_identity_minus_full_identity']['paired_scan_dice_95ci']},
  {'cohort':'WAW early-postop31','scope':'public day0-22 earliest scan; patients absent from preop152; not hidden blind','full_scan':waw_postop['full972_kd168']['identity']['locator_dsc_mean'],'full_ci':waw_postop['full972_kd168']['identity_bootstrap']['scan_dsc_mean_95ci'],'full_points':15,'point_denominator':31,'gcsam_scan':waw_postop['gcsam184']['identity']['locator_dsc_mean'],'gcsam_ci':waw_postop['gcsam184']['identity_bootstrap']['scan_dsc_mean_95ci'],'gcsam_points':14,'delta':waw_postop['comparisons']['gcsam_identity_minus_full_identity']['gcsam_minus_full']['scan_dice'],'delta_ci':waw_postop['comparisons']['gcsam_identity_minus_full_identity']['paired_scan_dice_95ci']}
 ]
 result={'schema':'aflow.cross_cohort.frozen_checkpoint_synthesis','schema_version':4,'status':'descriptive_separate_cohorts_no_pooled_ranking','models':{'full972_sha256':'f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b','gcsam184_sha256':'acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'},'cohorts':rows,'interpretation':'GCSAM-minus-Full Scan Dice is uncertain on internal270, UCSD75, CPTAC33, WAW preop152, and WAW early-postop31; only Africa95 has an interval entirely above zero. Point differences are -10/0/-3/0/-1/-1. Cohorts must not be pooled; Africa95 remains a domain-specific checkpoint advantage, not a monotonic or universal GCSAM law.','new_model_inference':False,'selection_or_tuning':False,'analyzer_sha256':sha(Path(__file__))};a.json.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 x=np.arange(6);w=.32;fig,axes=plt.subplots(1,3,figsize=(20.5,4.4),constrained_layout=True)
 labels=['Internal270','UCSD75','Africa95','CPTAC33','WAW-pre152','WAW-post31']
 for offset,key,ci_key,label,color in [(-w/2,'full_scan','full_ci','Full972 identity','#4c72b0'),(w/2,'gcsam_scan','gcsam_ci','GCSAM identity','#55a868')]:
  y=np.array([r[key] for r in rows]);cis=np.array([r[ci_key] for r in rows]);axes[0].bar(x+offset,y,w,label=label,color=color);axes[0].errorbar(x+offset,y,yerr=np.vstack((y-cis[:,0],cis[:,1]-y)),fmt='none',ecolor='black',capsize=3,lw=1)
 axes[0].set(ylabel='Scan Dice',title='Absolute scores (separate cohorts)',xticks=x,xticklabels=labels,ylim=(0.45,.89));axes[0].legend(frameon=False)
 delta=np.array([r['delta'] for r in rows]);cis=np.array([r['delta_ci'] for r in rows]);axes[1].axhline(0,color='black',lw=1);axes[1].errorbar(x,delta,yerr=np.vstack((delta-cis[:,0],cis[:,1]-delta)),fmt='o',color='#7b3294',capsize=5);axes[1].set(ylabel='GCSAM − Full Scan Dice',title='Within-cohort paired differences',xticks=x,xticklabels=labels)
 fp=np.array([r['full_points']/r['point_denominator'] for r in rows]);gp=np.array([r['gcsam_points']/r['point_denominator'] for r in rows]);axes[2].bar(x-w/2,fp,w,label='Full972 identity',color='#4c72b0');axes[2].bar(x+w/2,gp,w,label='GCSAM identity',color='#55a868');axes[2].set(ylabel='Point hit rate',title='Point and Dice remain distinct',xticks=x,xticklabels=labels,ylim=(.4,1.01));axes[2].legend(frameon=False)
 fig.suptitle('Frozen 1× checkpoint synthesis (no pooled ranking or tuning)');fig.savefig(a.png,dpi=180);fig.savefig(a.pdf);plt.close(fig)
 print(json.dumps({'cohorts':rows,'json_sha256':sha(a.json),'png_sha256':sha(a.png),'pdf_sha256':sha(a.pdf)},indent=2))

if __name__=='__main__':main()
