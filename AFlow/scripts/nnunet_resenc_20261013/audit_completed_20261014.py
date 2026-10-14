"""Independent report/score integrity audit using existing predictions, no model forwards."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *
import numpy as np

def main():
    verify_source_freeze('EXECUTION_FREEZE.json');verify_source_freeze('FINALIZATION_RECOVERY_FREEZE.json')
    expected=[(r['case_id'],r['subject_id']) for r in rows('test')]
    reports=[read(p) for p in sorted((R/'reports').glob('*.json'))]
    assert len(reports)==7
    for report in reports:
        cases=report['per_case'];m=report['metrics']
        assert [(r['case_id'],r['subject_id']) for r in cases]==expected
        scores=np.array([r['locator_dsc'] for r in cases]);assert np.isfinite(scores).all()
        for r in cases:
            den=2*r['tp']+r['fp']+r['fn']
            d=float(np.float32(2*r['tp'])/np.float32(den)) if den else 1.
            assert d==r['locator_dsc']
        assert abs(float(scores.mean())-m['scan_dice'])<1e-12
        subjects=sorted({r['subject_id'] for r in cases})
        n=np.array([sum(r['subject_id']==s for r in cases) for s in subjects])
        sums=np.array([sum(r['locator_dsc'] for r in cases if r['subject_id']==s) for s in subjects])
        assert len(subjects)==119 and abs(float((sums/n).mean())-m['subject_dice'])<1e-12
        assert sum(r['automatic_point_hits_gt'] for r in cases)==m['point_hits']
        idx=np.random.default_rng(20260917).integers(0,119,(10000,119))
        ci=np.quantile(sums[idx].sum(1)/n[idx].sum(1),[.025,.975])
        assert np.allclose(ci,m['scan_ci95'],rtol=0,atol=1e-12)
    single=next(r for r in reports if r['single_model'])
    journal=read(R/'TEST_INFERENCE_JOURNAL.json');assert journal['completed_scans']==270
    assert journal['per_case']==single['per_case']
    index=read(TMP/'test_native/index.json');shape=tuple(index['canvas'])
    labels=np.memmap(TMP/'test_native/labels.u8',dtype=np.uint8,mode='r',shape=(270,*shape))
    for i,(prediction,r) in enumerate(zip(journal['predictions'],single['per_case'])):
        path=ROOT/prediction['path'];assert sha(path)==prediction['sha256']
        with np.load(path) as stored:margin=stored['margin']
        geo=index['geometry'][r['case_id']];sl=tuple(slice(o,o+e) for o,e in zip(geo['canvas_offset'],geo['extent']))
        target=np.asarray(labels[(i,*sl)])>0;assert margin.shape==target.shape
        mask=margin>0
        assert int(np.count_nonzero(mask&target))==r['tp']
        assert int(np.count_nonzero(mask&~target))==r['fp']
        assert int(np.count_nonzero(~mask&target))==r['fn']
        point=tuple(int(v) for v in np.unravel_index(np.argmax(margin),margin.shape))
        assert list(point)==r['automatic_point'] and bool(target[point])==r['automatic_point_hits_gt']
    board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json');ranked=board['test270_tuned_native_rank_by_scan']
    assert [r['rank'] for r in ranked]==list(range(1,len(ranked)+1))
    assert all(a['scan_dice']>=b['scan_dice'] for a,b in zip(ranked,ranked[1:]))
    lookup={r['method']:r for r in ranked}
    for report in reports:
        row=lookup[report['method']]
        assert sha(ROOT/row['report_path'])==row['report_sha256']
        assert row['scan_dice']==report['metrics']['scan_dice']
    write(R/'INDEPENDENT_RESULT_AUDIT.json',{'at':now(),'status':'passed','reports_verified':7,
       'single_model_raw_predictions_verified':270,'cohort_scans':270,'cohort_subjects':119,
       'all_point_tp_fp_fn_and_dice_counts_exact':True,'bootstrap_ci_recomputed':True,
       'new_model_forwards':0,'new_prediction_configurations':0,'test_pixels_reread_for_audit':True,
       'original_frozen_sources_unchanged':True,'source_sha256':sha(Path(__file__))})

if __name__=='__main__':main()
