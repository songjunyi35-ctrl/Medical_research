"""Post-test user-directed diagnostic; fixed comparisons with baseline and epoch50."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_sammed3d_e050_test270 import paired_stats
from scripts.sammed3d_graymean_single972 import ROOT,sha,write,now
R=ROOT/'archive/research_records/2026-09-17_sammed3d_graymean_e161_test270'

def main():
    frozen=json.loads((R/'ANALYSIS_FREEZE.json').read_text())
    for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
    sel=json.loads((R/'SELECTION_RECEIPT.json').read_text());raw=json.loads((R/'RAW_TEST270.json').read_text())
    basepath=ROOT/sel['baseline_report'];baseline=json.loads(basepath.read_text());p50=ROOT/'archive/research_records/2026-09-17_sammed3d_graymean_e050_test270/RAW_TEST270.json';early=json.loads(p50.read_text())
    assert raw['checkpoint_epoch']==161 and raw['status']=='complete_270_case_test' and len(raw['cases'])==270
    assert len({r['subject_id'] for r in raw['cases']})==119 and raw['baseline_tensor_byte_parity']
    d={'status':'complete_270_case_test','utc':now(),'epoch':161,'cohort':{'scans':270,'subjects':119,'role':'reused internal test','external':False,'pristine_blind':False},'prior_test_seen':True,'post_test_diagnostic':True,'candidate_locked_before_this_evaluation':True,'test_informed_checkpoint_request':True,'hyperparameters_test_tuned':False,'training_stopped':True,'completed_training_epochs':161,'best5':paired_stats(raw['cases'],baseline['cases'],'best_of_5_dice'),'final5':paired_stats(raw['cases'],baseline['cases'],'final_click_5_dice'),'versus_epoch50_best5':paired_stats(raw['cases'],early['cases'],'best_of_5_dice'),'versus_epoch50_final5':paired_stats(raw['cases'],early['cases'],'final_click_5_dice'),'bootstrap':{'resamples':10000,'seed':20260917,'unit':'patient cluster paired','interval':'percentile95'},'checkpoint_sha256':sha(R/'checkpoint_epoch161.pth'),'checkpoint_bytes':(R/'checkpoint_epoch161.pth').stat().st_size,'runtime':raw['runtime'],'point_hit':None,'point_hit_reason':'GT-assisted interactive task; automatic locator point hit not applicable','evidence':{str(p.relative_to(ROOT)):sha(p) for p in [R/'RAW_TEST270.json',R/'FREEZE.json',R/'ANALYSIS_FREEZE.json',R/'PREACCESS.json',R/'MODEL_PREACCESS.json',R/'CACHE_INDEX.json',R/'checkpoint_epoch161.pth',R/'SELECTION_RECEIPT.json',basepath,p50]}}
    write(R/'RESULT.json',d);print(json.dumps({k:d[k] for k in ['best5','final5','versus_epoch50_best5','runtime']}))
if __name__=='__main__':main()
