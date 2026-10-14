"""Independent readiness audit; ten-hour completion remains a separate live requirement."""
import sys,json,math,hashlib,time,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.t1ce_10h_common import ROOT,R,OUT,P,MANIFEST,MANIFEST_SHA,read,write,sha,now,timestamp

def main():
 requirements={};sets={};manifest_rows={}
 for role,(n,ns) in {'train':(972,444),'validation':(108,50),'test':(270,119)}.items():
  assert sha(MANIFEST[role])==MANIFEST_SHA[role]
  rr=[json.loads(x) for x in MANIFEST[role].read_text().splitlines() if x.strip()]
  assert len(rr)==len({r['case_id'] for r in rr})==n
  sets[role]={r['subject_id'] for r in rr};assert len(sets[role])==ns;manifest_rows[role]=rr
 assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
 requirements['fixed_cohorts_and_patient_isolation']={'status':'proven','hashes':MANIFEST_SHA}
 freeze_files=['FREEZE.json','EXECUTION_FREEZE.json','COMPLETION_CODE_FREEZE.json','ENGINEERING_AMENDMENT_001.json','ENGINEERING_AMENDMENT_002.json','ENGINEERING_AMENDMENT_003.json','FINAL_REPORT_SUPPLEMENT_FREEZE.json']
 verified=[]
 for name in freeze_files:
  f=read(R/name)
  for path,digest in f.get('files',{}).items():assert sha(ROOT/path)==digest;verified.append(path)
 requirements['frozen_code_and_declared_engineering_changes']={'status':'proven','files_checked':sorted(set(verified)),'receipts':freeze_files}
 training={}
 for arm,epochs in [('A',200),('B',25)]:
  state=torch.load(OUT/arm/'latest.pth',map_location='cpu',weights_only=False)
  assert state['epoch']==epochs
  steps={int(v['step']) for v in state['optimizer_state_dict']['state'].values()}
  assert steps=={epochs*122},(arm,steps)
  log=[json.loads(x) for x in (OUT/arm/'TRAIN.jsonl').read_text().splitlines()]
  assert [r['epoch'] for r in log]==list(range(1,epochs+1))
  assert all(r['unique_scans']==972 and r['updates']==122 for r in log)
  best=read(OUT/arm/'BEST.json');history=state['history'];expected=max(history,key=lambda h:(h['score'],-h['epoch']))
  assert best['epoch']==expected['epoch'] and best['score']==expected['score']
  training[arm]={'epochs':epochs,'actual_optimizer_state_steps':list(steps),'parameter_states':len(state['optimizer_state_dict']['state']),'coverage_rows':len(log),'full_scan_views':epochs*972,'selected_additional_epoch':best['epoch'],'latest_sha256':sha(OUT/arm/'latest.pth')}
  del state
 assert read(R/'C/STOP_RESULT.json')['state']=='not_started'
 assert read(R/'DIAGNOSTIC.json')['enable_C'] is False
 requirements['actual_training_and_selection']={'status':'proven','training':training,'C':'not_started_by_frozen_diagnostic_gate'}
 access={c:len(list((R/'access').glob(c+'_*.json'))) for c in 'ABC'}
 assert access=={'A':3,'B':1,'C':0}
 assert all(n<=P['max_test_checkpoints_per_candidate'] for n in access.values())
 receipts=[read(p) for p in (R/'access').glob('*.json')]
 assert all(r['test_tuned'] and r['segmentation_training_uses_test_labels'] is False for r in receipts)
 requirements['test_budget_and_disclosure']={'status':'proven','candidate_unique_model_accesses':access,'total_full_cohort_model_accesses':len(receipts),'fixed_reload_case_indices':read(R/'RELOAD_PREACCESS.json')['indices'],'new_outer_confirmation':False}
 write(R/'INDEPENDENT_METRIC_AUDIT_PREACCESS.json',{'utc':now(),'purpose':'verify saved counts, patient summaries and point flags against existing compressed native labels','new_model_forward':False,'new_configuration_or_selection':False,'test_tuned_campaign':True})
 target_index=read(OUT/'label_relay/test/index.json');assert sha(OUT/'label_relay/test/labels.npz')==target_index['labels_sha256']
 labels=np.load(OUT/'label_relay/test/labels.npz',allow_pickle=False)
 reports={p.stem:read(p) for p in (R/'reports').glob('*.json')}
 expected={r['case_id']:r['subject_id'] for r in manifest_rows['test']};byreport={}
 for name,d in reports.items():
  rr=d['per_case'];assert len(rr)==len({r['case_id'] for r in rr})==270
  assert {r['case_id']:r['subject_id'] for r in rr}==expected
  byreport[name]={r['case_id']:r for r in rr}
  groups=collections.defaultdict(list)
  for r in rr:
   denom=2*r['tp']+r['fp']+r['fn'];dice=2*r['tp']/denom if denom else 1.
   assert abs(dice-r['locator_dsc'])<1e-12
   assert r['tp']+r['fn']==target_index['geometry'][r['case_id']]['positive_voxels']
   groups[r['subject_id']].append(dice)
  scan=math.fsum(r['locator_dsc'] for r in rr)/270;subject=math.fsum(math.fsum(v)/len(v) for v in groups.values())/119
  assert abs(scan-d['metrics']['scan_dice'])<1e-12 and abs(subject-d['metrics']['subject_dice'])<1e-12
  assert sum(r['automatic_point_hits_gt'] for r in rr)==d['metrics']['point_hits']
 for cid in expected:
  lab=labels[cid]
  for name,rr in byreport.items():
   r=rr[cid];point=tuple(r['automatic_point']);assert len(point)==3 and all(0<=v<n for v,n in zip(point,lab.shape))
   assert bool(lab[point])==r['automatic_point_hits_gt'],(name,cid)
 labels.close()
 subjects=sorted(set(expected.values()));counts=np.array([sum(s==v for v in expected.values()) for s in subjects],dtype=np.float64)
 absolute_ix=np.random.default_rng(20260917).integers(0,119,(10000,119));paired_ix=np.random.default_rng(20260918).integers(0,119,(10000,119))
 sums={}
 for name,rr in byreport.items():
  sums[name]=np.array([math.fsum(v['locator_dsc'] for v in rr.values() if v['subject_id']==s) for s in subjects])
  ci=np.quantile(sums[name][absolute_ix].sum(1)/counts[absolute_ix].sum(1),[.025,.975])
  assert np.max(np.abs(ci-np.array(reports[name]['metrics']['scan_ci95'])))<1e-12
  sci=np.quantile((sums[name]/counts)[absolute_ix].mean(1),[.025,.975])
  assert np.max(np.abs(sci-np.array(reports[name]['metrics']['subject_ci95'])))<1e-12
  for field,ref in [('vs_strongest_reference','ref_native173'),('vs_fixed193','ref_pure193')]:
   if field not in reports[name]:continue
   dif=sums[name]-sums[ref] if ref in sums else sums[name]-np.array([math.fsum(v['locator_dsc'] for v in byreport[ref].values() if v['subject_id']==s) for s in subjects])
   ci=np.quantile(dif[paired_ix].sum(1)/counts[paired_ix].sum(1),[.025,.975])
   assert np.max(np.abs(ci-np.array(reports[name][field]['scan_delta_ci95'])))<1e-12
 requirements['independent_metric_recalculation']={'status':'proven','reports':len(reports),'point_flags_verified_from_native_labels':len(reports)*270,'counts_and_scan_subject_means':'exact within1e-12','patient_cluster_scan_subject_intervals':'independently reconstructed within1e-12'}
 f=read(R/'FINAL_PANEL_FREEZE.json');assert f['compute_cap']==8 and len(f['recipes'])==12
 assert not (R/'INFERENCE_COST_AMENDMENT.json').exists()
 selection=read(R/'FINAL_SELECTION.json');ranked=sorted([reports[r['id']] for r in f['recipes']],key=lambda d:(-d['metrics']['scan_dice'],d['forward_equivalents'],d['id']))
 assert selection['overall']==ranked[0]['id']
 assert selection['best_single_model']==next(d['id'] for d in ranked if len(d['components'])==1)
 win=ranked[0];passed=win['metrics']['scan_dice']>=.8 and win['vs_strongest_reference']['scan_delta']>=.03
 assert selection['target_passed']==passed==False
 requirements['fixed_output_panel_and_honest_promotion']={'status':'proven','configurations':12,'cap':8,'selected':selection,'target_achieved':False}
 contract=read(OUT/'deployment/manifest.json');single=read(OUT/'deployment/best_single_manifest.json')
 for cfg in [contract,single]:
  assert cfg['only_external_input']=='T1ce' and cfg['no_labels_or_case_lookup'] is True
  for c in cfg['components']:assert sha(ROOT/c['checkpoint_path'])==c['checkpoint_sha256']
 assert read(R/'FINAL_PANEL_CHECKS.json')['status']=='passed' and read(R/'DEPLOYMENT_CHECK.json')['status']=='passed'
 requirements['deployment_and_numeric_limits']={'status':'proven','exported_model_hashes_verified':True,'frozen4case_reload_passed':True,'batch1_batch2_bitwise_identity_claimed':False,'image_only_interface':'scripts/deploy_t1ce_10h.py:infer_views/nifti accepts image and frozen manifest, no label or case-routing input'}
 persistent=sum(p.stat().st_size for base in [R,OUT] for p in base.rglob('*') if p.is_file())
 assert persistent<8*2**30
 for side in ['dual','single']:
  d=read(R/f'HEARTBEAT_{side}.json')['resources'];assert d['persistent_free']>=20*2**30 and d['scratch_free']>=100*2**30 and d['scratch_bytes']<=100*2**30
 requirements['storage_limits']={'status':'proven_at_audit','persistent_bytes':persistent,'historical_material_deleted':False,'temporary_data_on_worker_scratch':True}
 elapsed=time.time()-timestamp(P['start_utc']);requirements['ten_hour_monitoring']={'status':'proven' if elapsed>=36000 and (R/'COMPLETION_AUDIT.json').exists() else 'pending','elapsed_seconds':elapsed,'deadline_utc':P['deadline_utc'],'do_not_mark_goal_complete_before_final_process_audit':True}
 write(R/'COMPLETION_READINESS_AUDIT.json',{'utc':now(),'classification_of_previous_goal_turn':'progress','requirements':requirements,'optimization_work_proven':True,'full_goal_complete':requirements['ten_hour_monitoring']['status']=='proven','remaining_requirement':'finish monitored window and verify terminal processes at deadline'})
 print(json.dumps({k:v['status'] for k,v in requirements.items()},indent=2))

if __name__=='__main__':main()
