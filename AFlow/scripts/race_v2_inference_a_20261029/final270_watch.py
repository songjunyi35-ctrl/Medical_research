"""Publish the frozen270 entry with a canonical-RAW-bound mask receipt.

Prediction, scoring, and full saved-mask arithmetic remain in final270.py. This
adapter delays the public RAW alias until the independent receipt is complete.
"""
import argparse,time
from pathlib import Path
from scripts.race_v2_inference_a_20261029 import final270 as base
from scripts.race_v2_inference_a_20261029.common import *

def execute(statistics_only=False):
    assert base.SELECTION.exists()
    selection=read(base.SELECTION);digest=sha(base.SELECTION)
    # Validation is repeated by the frozen entry before any arrays are opened.
    base.validate_selection(selection)
    singles=[c for c in selection['candidates'] if c['kind']=='single']
    assert len(singles)==1 and singles[0]['candidate_id']=='single'
    canonical=base.F/'candidates/single/RAW_TEST270.json'
    public=base.F/'RAW_SINGLE.json'
    original_write=base.write;original_audit=base.audit_saved_masks
    def write_after_audit(path,value):
        if Path(path)==public:return
        original_write(path,value)
    def audit_canonical(path,configuration,selection_sha,metrics):
        assert Path(path)==public
        return original_audit(canonical,configuration,selection_sha,metrics)
    base.write=write_after_audit;base.audit_saved_masks=audit_canonical
    try:base.execute(statistics_only)
    finally:base.write=original_write;base.audit_saved_masks=original_audit
    assert sha(base.SELECTION)==digest
    raw=read(canonical);assert raw['selection_sha256']==digest
    source_receipt=base.F/'MASK_AUDIT_SINGLE.json';receipt=read(source_receipt)
    assert receipt['status']=='passed' and receipt['scans']==270
    assert receipt['raw_sha256']==sha(canonical) and receipt['selection_sha256']==digest
    assert receipt['all_saved_masks_counts_dice_point_exact'] is True
    published_receipt=base.F/'MASKS_AUDIT_RECEIPT_SINGLE.json'
    receipt=dict(receipt,all_masks_counts_points_exact=True,raw_path=str(canonical),source_mask_audit_path=str(source_receipt),source_mask_audit_sha256=sha(source_receipt),publication_source_sha256=sha(Path(__file__)))
    original_write(published_receipt,receipt)
    published=dict(raw,masks_audit_receipt_path=str(published_receipt),masks_audit_receipt_sha256=sha(published_receipt),mask_audit_path=str(published_receipt),canonical_raw_path=str(canonical),canonical_raw_sha256=sha(canonical))
    original_write(public,published)
    result=read(base.F/'candidates/single/RESULT.json')
    original_write(base.F/'RESULT_SINGLE.json',dict(result,aggregate_raw_path=str(public),aggregate_raw_sha256=sha(public),masks_audit_receipt_path=str(published_receipt),masks_audit_receipt_sha256=sha(published_receipt),canonical_raw_path=str(canonical),canonical_raw_sha256=sha(canonical)))
    original_write(base.F/'WATCH_STATUS.json',dict(at=now(),state='complete',selection_sha256=digest,raw_path=str(public),raw_sha256=sha(public),masks_audit_receipt_path=str(published_receipt),masks_audit_receipt_sha256=sha(published_receipt),automatic_neural_retry=False))

def main(watch,statistics_only):
    if watch:
        while not base.SELECTION.exists():guard();time.sleep(30)
    execute(statistics_only)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--statistics-only',action='store_true');args=parser.parse_args()
    try:main(args.watch,args.statistics_only)
    except BaseException as error:
        import traceback
        write(base.F/f'WATCH_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
