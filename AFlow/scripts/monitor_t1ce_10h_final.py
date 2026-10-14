"""Completed optimization, continuing the original ten-hour monitoring deadline."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scripts.complete_t1ce_10h_v3
from scripts import complete_t1ce_10h as original
from scripts.supplement_t1ce_10h_report import supplement
from scripts.t1ce_10h_common import *
publish=original.publish;access=original.access_summary;evidence=original.evidence

def publish_complete(final=False):publish(final);supplement()
def access_complete():
 access();p=R/'ACCESS_SUMMARY.json';d=read(p);d.update(whole_campaign_test_access=True,candidate_C={'target_diagnostic_used':True,'candidate_model_forward':False},checksum_access_scope_clarification='ACCESS_BOUNDARY_CLARIFICATION.json',clean_fit_diagnostic={'cohorts':['train972','validation108'],'new_test_access':False,'selection_changed':False});write(p,d)
def evidence_complete():
 evidence();p=R/'EVIDENCE_SHA256.json';d=read(p)
 names=['TRAIN_FIT_DIAGNOSTIC.json','TRAIN_FIT_DIAGNOSTIC.md','TRAIN_FIT_PREACCESS.json','RAW_CLEAN_TRAIN.json','RAW_CLEAN_VALIDATION.json','DEPLOYMENT_BENCHMARK.json','FINAL_PANEL_CHECKS.json','EXPORT_REPAIR_RECEIPT.json','ACCESS_BOUNDARY_CLARIFICATION.json','EVALUATION_CONTRACT.json','NUMERIC_COMPARABILITY.json','CHECKPOINT_IDENTITIES.json','output_panel/RESULT.json','output_panel/RESULT.md','B/INITIALIZATION_IDENTITY_CHECK.json','ENVIRONMENT_dual.json','ENVIRONMENT_single.json','LIVE_CURVES.png']
 for name in names:d['files'][str((R/name).relative_to(ROOT))]=sha(R/name)
 p2=OUT/'deployment/best_single_manifest.json';d['files'][str(p2.relative_to(ROOT))]=sha(p2);write(p,d)
original.publish=publish_complete;original.access_summary=access_complete;original.evidence=evidence_complete

if __name__=='__main__':
 try:
  assert read(R/'COMPLETION_RECOVERED.json')['status']=='passed'
  supplement();access_complete();original.checks();evidence_complete();original.main()
 except BaseException:write(R/f'FINAL_MONITOR_FAILURE_{int(time.time())}.json',{'utc':now(),'traceback':traceback.format_exc()});raise
