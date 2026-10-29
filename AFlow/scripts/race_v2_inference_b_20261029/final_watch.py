"""Quiet root-triggered final queue; each GPU stage runs in its own process."""
import subprocess
import sys
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.final270 import F,WREADY,final_selection,test_data
import numpy as np

def run(stage,*extra):
    subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.final270',stage,*extra],check=True)

def audit_and_publish(candidate):
    selection,_=final_selection(candidate['candidate_id']);folder=F/'candidates'/candidate['candidate_id']
    raw_path=folder/'RAW_TEST270.json';raw=read(raw_path);selection_sha=sha(STAGE/'FINAL_TEST_SELECTION.json')
    assert raw['selection_sha256']==selection_sha
    index,cohort,_,labels=test_data(selection,True);records=raw['per_case']
    assert len(records)==270 and [r['case_id'] for r in records]==[r['case_id'] for r in cohort]
    audits=[]
    for slot,(row,record) in enumerate(zip(cohort,records)):
        path=Path(record['prediction_path']);assert sha(path)==record['prediction_sha256']
        saved=np.load(path,allow_pickle=False);shape=tuple(saved['shape'].tolist())
        assert list(shape)==record['prediction_shape']==index['geometry'][row['case_id']]['extent']
        mask=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
        target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
        tp=int((mask&target).sum());fp=int((mask&~target).sum());fn=int((~mask&target).sum());den=2*tp+fp+fn
        dice=float(np.float32(2*tp)/np.float32(den)) if den else 1.
        assert (tp,fp,fn)==(record['tp'],record['fp'],record['fn']) and dice==record['locator_dsc']
        point=saved['point'].tolist();assert point==record['automatic_point']
        assert bool(target[tuple(point)])==record['automatic_point_hits_gt']
        audits.append(dict(case_id=row['case_id'],prediction_path=str(path),prediction_sha256=sha(path),tp=tp,fp=fp,fn=fn))
    receipt=folder/'MASK_AUDIT.json'
    write(receipt,dict(at=now(),status='passed',candidate_id=candidate['candidate_id'],scans=270,
        selection_sha256=selection_sha,raw_sha256=sha(raw_path),all_masks_counts_points_exact=True,per_case=audits,model_forwards_added=0))
    published=dict(raw,candidate_id=candidate['candidate_id'],masks_audit_receipt_path=str(receipt),
        masks_audit_receipt_sha256=sha(receipt),mask_audit_path=str(receipt))
    write(F/'RAW_FUSION.json',published)
    write(F/'RESULT_FUSION.json',dict(read(folder/'RESULT.json'),masks_audit_receipt_path=str(receipt),
        aggregate_raw_path=str(F/'RAW_FUSION.json'),aggregate_raw_sha256=sha(F/'RAW_FUSION.json')))

def main():
    own=read(R/'FINAL_WATCH_FREEZE.json')
    for source,digest in own['source_sha256'].items():assert sha(ROOT/source)==digest,source
    assert sha(R/'FINAL_ENTRY_FREEZE.json')==own['final_entry_freeze_sha256']
    exclusive(F/'WATCH_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=1,
        behavior='quiet until true root selection; no artificial GPU load; no automatic neural retry'))
    selected=STAGE/'FINAL_TEST_SELECTION.json'
    while not selected.exists():
        guard();time.sleep(30)
    selection,_=final_selection();fusion=[c for c in selection['candidates'] if c['kind']=='fusion'];assert len(fusion)<=1
    if not fusion:
        write(F/'WATCH_STATUS.json',dict(at=now(),state='not_selected',selection_sha256=sha(selected)));return
    candidate=fusion[0];candidate_id=candidate['candidate_id'];folder=F/'candidates'/candidate_id
    write(F/'WATCH_STATUS.json',dict(at=now(),state='root_selected',candidate_id=candidate_id,selection_sha256=sha(selected)))
    if not (folder/'RAW_TEST270.json').exists():
        if any(m['model_id']=='w32e171' for m in candidate['models']) and not WREADY.exists():run('forward-w32')
        while not all(Path(m['atoms_ready_path']).exists() for m in candidate['models']):
            guard()
            owner_failures=list((ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/final270').glob('FAILURE*.json'))
            if owner_failures:raise RuntimeError('A final model queue failed; no B fallback/repeated RACE forward')
            time.sleep(10)
        run('score','--candidate-id',candidate_id)
    audit_and_publish(candidate)
    write(F/'WATCH_STATUS.json',dict(at=now(),state='complete',candidate_id=candidate_id,
        raw_path=str(F/'RAW_FUSION.json'),raw_sha256=sha(F/'RAW_FUSION.json')))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(F/'WATCH_FAILURE.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
