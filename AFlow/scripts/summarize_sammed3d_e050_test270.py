"""Frozen patient-cluster paired analysis of the explicit epoch50 test candidate."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.sammed3d_graymean_single972 import ROOT,sha,write,now
R=ROOT/'archive/research_records/2026-09-17_sammed3d_graymean_e050_test270'


def paired_stats(candidate,baseline,field):
    if [(r['case_id'],r['subject_id']) for r in candidate]!=[(r['case_id'],r['subject_id']) for r in baseline]:raise ValueError('case/subject alignment mismatch')
    subjects=sorted({r['subject_id'] for r in candidate})
    n=np.array([sum(r['subject_id']==s for r in candidate) for s in subjects])
    c=np.array([sum(r[field] for r in candidate if r['subject_id']==s) for s in subjects])
    b=np.array([sum(r[field] for r in baseline if r['subject_id']==s) for s in subjects])
    ix=np.random.default_rng(20260917).integers(0,len(subjects),(10000,len(subjects)))
    def stats(x):
        return {'scan':float(x.sum()/n.sum()),'subject':float((x/n).mean()),'scan_ci95':np.quantile(x[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_ci95':np.quantile((x/n)[ix].mean(1),[.025,.975]).tolist()}
    diff=np.array([a[field]-b[field] for a,b in zip(candidate,baseline)])
    return {'candidate':stats(c),'baseline':stats(b),'paired_delta':stats(c-b),'scan_improved':int((diff>0).sum()),'scan_worse':int((diff<0).sum()),'scan_tied':int((diff==0).sum())}


def main():
    receipt=json.loads((R/'SELECTION_RECEIPT.json').read_text());raw=json.loads((R/'RAW_TEST270.json').read_text());basepath=ROOT/receipt['baseline_report']
    assert sha(basepath)==receipt['baseline_report_sha256']
    baseline=json.loads(basepath.read_text())
    assert raw['status']=='complete_270_case_test' and len(raw['cases'])==270 and len({r['subject_id'] for r in raw['cases']})==119
    assert raw['checkpoint_epoch']==50 and raw['baseline_tensor_byte_parity']
    result={'status':'complete_270_case_test','utc':now(),'cohort':{'scans':270,'subjects':119,'role':'reused internal test','external':False,'pristine_blind':False},'epoch':50,'best5':paired_stats(raw['cases'],baseline['cases'],'best_of_5_dice'),'final5':paired_stats(raw['cases'],baseline['cases'],'final_click_5_dice'),'bootstrap':{'unit':'patient cluster paired','resamples':10000,'seed':20260917,'interval':'percentile95'},'runtime':raw['runtime'],'checkpoint_sha256':sha(R/'checkpoint_epoch050.pth'),'checkpoint_bytes':(R/'checkpoint_epoch050.pth').stat().st_size,'cache_bytes':json.loads((R/'CACHE_INDEX.json').read_text())['bytes'],'test_used_for_selection_or_adaptation':False,'single_factor_control':False,'baseline_tensor_byte_parity':True,'point_hit':None,'point_hit_reason':'GT-assisted interactive task; automatic locator point hit not applicable','evidence':{str(p.relative_to(ROOT)):sha(p) for p in [R/'RAW_TEST270.json',R/'SELECTION_RECEIPT.json',R/'PREACCESS.json',R/'MODEL_PREACCESS.json',R/'CACHE_INDEX.json',R/'checkpoint_epoch050.pth',R/'FREEZE.json',R/'ANALYSIS_FREEZE.json',basepath]}}
    result['point_estimate_exceeds_baseline']=result['best5']['paired_delta']['scan']>0
    result['paired_scan_ci_entirely_positive']=result['best5']['paired_delta']['scan_ci95'][0]>0
    write(R/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['best5','final5','runtime']},ensure_ascii=False))

if __name__=='__main__':main()
