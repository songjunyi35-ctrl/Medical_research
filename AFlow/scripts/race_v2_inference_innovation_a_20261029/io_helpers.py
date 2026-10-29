"""Packed-mask persistence and independent full-crop arithmetic audit."""
import numpy as np
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.statistics import summary_ci

def record_mask(row,slot,mask,point,base_point,target,path,diagnostic=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        stored=np.load(path,allow_pickle=False);previous=np.unpackbits(stored['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
        assert np.array_equal(previous,mask) and stored['point'].tolist()==point and stored['base_point'].tolist()==base_point
    else:
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('xb') as out:np.savez_compressed(out,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point),base_point=np.array(base_point))
        os.replace(temporary,path);path.chmod(0o444)
    tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
    return dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,
        locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=pred-tp,fn=positive-tp,
        automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),
        base_automatic_point=base_point,base_automatic_point_hits_gt=bool(target[tuple(base_point)]),
        prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape),diagnostic=diagnostic or {})

def audit_masks(raw_path,receipt_path,cohort,index,labels,selection_sha256=None):
    raw=read(raw_path);configurations=raw.get('per_configuration',{'single':raw.get('per_case')});audits={}
    for identifier,records in configurations.items():
        assert len(records)==len(cohort);checks=[]
        for record,row in zip(records,cohort):
            slot=record['source_slot'];assert (record['case_id'],record['subject_id'])==(row['case_id'],row['subject_id']) and index['case_ids'][slot]==row['case_id']
            assert sha(record['prediction_path'])==record['prediction_sha256'];stored=np.load(record['prediction_path'],allow_pickle=False)
            shape=tuple(stored['shape']);assert list(shape)==index['geometry'][row['case_id']]['extent']==record['prediction_shape']
            mask=np.unpackbits(stored['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
            point=stored['point'].tolist();base=stored['base_point'].tolist();assert point==record['automatic_point'] and base==record['base_automatic_point']
            target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
            tp=int(np.count_nonzero(mask&target));fp=int(np.count_nonzero(mask&~target));fn=int(np.count_nonzero(~mask&target));den=2*tp+fp+fn
            dice=float(np.float32(2*tp)/np.float32(den)) if den else 1.;hit=bool(target[tuple(point)]);base_hit=bool(target[tuple(base)])
            assert (tp,fp,fn,dice,hit,base_hit)==(record['tp'],record['fp'],record['fn'],record['locator_dsc'],record['automatic_point_hits_gt'],record['base_automatic_point_hits_gt'])
            checks.append(dict(case_id=row['case_id'],tp=tp,fp=fp,fn=fn,dice=dice,Point_refined_hit=hit,Point_base_hit=base_hit))
        audits[identifier]=dict(scans=len(records),metrics=summary_ci(records),per_case=checks)
    receipt=dict(at=now(),status='passed',state='passed',scans=len(cohort),configurations=len(configurations),
        all_masks_counts_points_exact=True,all_masks_counts_dice_and_both_points_exact=True,
        raw_path=str(raw_path),raw_sha256=sha(raw_path),selection_sha256=selection_sha256,
        audit_source_path=str(Path(__file__).relative_to(ROOT)),audit_source_sha256=sha(Path(__file__)),
        per_configuration=audits,whole_original_crop_including_outside_support=True,model_forwards_added=0)
    write(receipt_path,receipt);return receipt
