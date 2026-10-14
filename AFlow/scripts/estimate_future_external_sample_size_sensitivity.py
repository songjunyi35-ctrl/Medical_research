#!/usr/bin/env python3
"""Planning-only paired sample-size sensitivity from frozen one-scan cohorts."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SOURCES={'africa':(ROOT/'archive/research_records/2026-09-06_brats_africa95_frozen_baselines/RAW_EXTERNAL95.json','c9fb07be4412c160129037104936e4c34b6390fa733b998b8a8f35524ccd30b3'),'cptac':(ROOT/'archive/research_records/2026-09-06_cptac33_africa_informed_confirmation/RAW_CPTAC33.json','4b9d8c90e1b5ec45b0e664e560301db20affd6a96387f416a62b561f8d57b4fd'),'waw':(ROOT/'archive/research_records/2026-09-06_waw_gliomai152_frozen_panel/RAW_WAW152.json','ef35301a78f6e47ac0574e5727aa7c5015b2f30cd73e1c8c45487fa8591dde91'),'waw_postop':(ROOT/'archive/research_records/2026-09-06_waw_postop_only31_frozen_panel/RAW_WAW_POSTOP31.json','72269a5a872389db9d6dd316d7d46b614bad8d81796b0bb72625814907d98a98')};Z=1.959963984540054+0.8416212335729143
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dice(summary):return np.asarray([x['locator_dsc'] for x in summary['per_case']],dtype=float)
def need(sd,effect):return int(math.ceil((Z*sd/effect)**2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 for path,digest in SOURCES.values():
  if sha(path)!=digest:raise ValueError('SHA')
 data={k:json.load(open(v[0])) for k,v in SOURCES.items()};observed={}
 for name,d in data.items():
  full=d['full972_kd168']['identity'];gc=d['gcsam184']['identity'];flip=d['full972_kd168']['flip8'];gd=dice(gc)-dice(full);fd=dice(flip)-dice(full);observed[name]={'subjects':len(gd),'gcsam_minus_full':{'mean':float(gd.mean()),'sample_sd':float(gd.std(ddof=1))},'flip8_minus_full':{'mean':float(fd.mean()),'sample_sd':float(fd.std(ddof=1))}}
 effects=[.005,.01,.02];planning={}
 for key in ('gcsam_minus_full','flip8_minus_full'):
  sds=[observed[x][key]['sample_sd'] for x in observed];median=statistics.median(sds);maximum=max(sds);planning[key]={'external_single_scan_sd_values':sds,'median_sd':median,'maximum_sd':maximum,'independent_subjects_needed':{str(effect):{'median_sd_scenario':need(median,effect),'maximum_sd_scenario':need(maximum,effect)} for effect in effects}}
 result={'schema':'aflow.future_external.sample_size_sensitivity','schema_version':2,'status':'planning_sensitivity_not_confirmatory_power_guarantee','formula':'ceil(((z_0.975 + z_0.80) * paired_delta_sd / target_mean_delta)^2)','alpha_two_sided':.05,'nominal_power':.8,'observed_one_scan_per_subject_cohorts':observed,'planning':planning,'excluded_from_sd_scenarios':{'internal270':'reused internal and multiple scans per subject','ucsd75':'75 scans from53 subjects; scan independence invalid'},'limitations':'Observed SD varies strongly by cohort and effects are not transportable. Normal approximation ignores nonnormal tails, clustering, multiplicity, attrition, source failures, and selection. Use only as a lower-bound sensitivity range and preregister the meaningful delta before new data.','access':{'new_model_inference':False,'cohort_reaccess':False,'selection_or_tuning':False},'source_sha256':{k:v[1] for k,v in SOURCES.items()},'analyzer_sha256':sha(Path(__file__))};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'planning':planning,'output_sha256':sha(a.output)},indent=2))
if __name__=='__main__':main()
