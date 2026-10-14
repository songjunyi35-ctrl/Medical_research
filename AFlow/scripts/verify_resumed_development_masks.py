"""Independent mask counts plus descriptive class errors; zero new forward calls."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import importlib
import json
from pathlib import Path
import numpy as np
from sammed3d.packed_prediction_masks import read_packed_case
from sammed3d.development_mask_contract import validate_index,validate_case


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['small_rotation','image_blur'],required=True);args=ap.parse_args()
    p=importlib.import_module('scripts.'+args.study+'_protocol');p.check_freeze();rec=p.RECORD
    if (rec/'DEV_MASK_VERIFICATION.json').exists():raise FileExistsError('already verified')
    raw=json.loads((rec/'RAW_DEVELOPMENT.json').read_text());main_verification=json.loads((rec/'VERIFICATION_RECEIPT.json').read_text())
    assert main_verification['status']=='passed';p.checked(rec/'RAW_DEVELOPMENT.json',main_verification['raw_sha256']);p.checked(rec/'RESULT.json',main_verification['result_sha256'])
    done=json.loads((rec/'DEV_MASK_ARCHIVAL_DONE.json').read_text());assert done['status']=='completed'
    p.checked(rec/'DEV_MASK_PREACCESS.json',done['preaccess_sha256']);p.checked(rec/'DEV_MASK_INDEX.json',done['index_sha256']);p.checked(rec/'RAW_DEVELOPMENT.json',done['raw_sha256'])
    pre=json.loads((rec/'DEV_MASK_PREACCESS.json').read_text());index=json.loads((rec/'DEV_MASK_INDEX.json').read_text())
    for n,d in pre['files'].items():p.checked(p.ROOT/n,d)
    p.checked(rec/'DEV_MASK_ADDENDUM.md',pre['addendum_sha256'])
    p.checked(rec/'IMPLEMENTATION_FREEZE.json',pre['original_implementation_freeze_sha256']);p.checked(rec/'DEVELOPMENT_IMPLEMENTATION_FREEZE.json',pre['original_development_freeze_sha256'])
    for info in pre['checkpoints'].values():
        p.checked(info['checkpoint']['path'],info['checkpoint']['sha256']);p.checked(info['train_report_path'],info['train_report_sha256'])
    assert index['associated_raw_sha256']==done['raw_sha256'] and index['observer_preaccess_sha256']==done['preaccess_sha256']
    assert index['case_ids']==pre['case_ids'] and set(index['files'])==set(raw['results'])==set(pre['keys'])
    assert index['observed_forwards']==raw['new_forwards']==pre['expected_forwards']
    validate_index(index,pre['case_ids'],pre['keys'],rec/'dev_prediction_masks')
    for info in index['files'].values():p.checked(info['path'],info['sha256'])
    cache_index=json.loads((p.CACHE/'index.json').read_text());p.checked(p.CACHE/'index.json',p.INDEX_SHA);lookup={c:i for i,c in enumerate(cache_index['case_ids'])}
    aux=json.loads((rec/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    p.checked(pre['class_cache_receipt_path'],pre['class_cache_receipt_sha256']);cr=json.loads(Path(pre['class_cache_receipt_path']).read_text());p.checked(cr['path'],pre['class_cache_sha256'])
    assert cr['case_ids']==pre['case_ids'];classes=np.memmap(cr['path'],mode='r',dtype=np.uint8,shape=(180,128,128,128))
    count=0;anatomy={}
    for key,summary in raw['results'].items():
        target_counts=np.zeros(5,dtype=np.int64);prediction_counts=np.zeros(5,dtype=np.int64);point_classes=Counter()
        for i,row in enumerate(summary['per_case']):
            assert row['case_id']==pre['case_ids'][i]
            pred=read_packed_case(index,key,i);truth=labels[lookup[row['case_id']]]>0;cl=np.asarray(classes[i])
            validate_case(row,pred,truth)
            assert np.array_equal(np.isin(cl,[1,2,3]),truth)
            ntrue=int(truth.sum());npred=int(pred.sum());inter=int((pred&truth).sum())
            assert (inter,npred,ntrue)==(row['intersection'],row['predicted_voxels'],row['target_voxels'])
            assert row['dice']==2*inter/(ntrue+npred)
            point=row['point_flat_index'];assert 0<=point<truth.size and bool(truth.reshape(-1)[point])==row['point_hit']
            target_counts+=np.bincount(cl.ravel(),minlength=5);prediction_counts+=np.bincount(cl[pred],minlength=5);point_classes[str(int(cl.reshape(-1)[point]))]+=1;count+=1
        assert sum(point_classes.get(str(k),0) for k in [1,2,3])==summary['point_hits']
        anatomy[key]={'class_target_voxels':target_counts.tolist(),'class_predicted_voxels':prediction_counts.tolist(),'false_negatives_by_class':{str(k):int(target_counts[k]-prediction_counts[k]) for k in [1,2,3]},'false_positives_by_class':{str(k):int(prediction_counts[k]) for k in [0,4]},'point_class_histogram':dict(point_classes),'descriptive_only_not_selection':True}
    assert count==pre['expected_forwards']
    p.write_new(rec/'DEV_MASK_VERIFICATION.json',{'status':'passed_actual_masks','created_utc':datetime.now(timezone.utc).isoformat(),'raw_sha256':p.sha(rec/'RAW_DEVELOPMENT.json'),'result_sha256':p.sha(rec/'RESULT.json'),'observer_preaccess_sha256':p.sha(rec/'DEV_MASK_PREACCESS.json'),'mask_index_sha256':p.sha(rec/'DEV_MASK_INDEX.json'),'actual_masks_checked':count,'new_model_forwards':0,'extra_selection_or_parameter_change':False,'independent_counts_and_point_hit':True,'argmax_not_independently_recomputed':True,'class_error_description':anatomy,'class_cache_sha256':pre['class_cache_sha256'],'verifier_sha256':p.sha(__file__),'test_access':False})
    print(json.dumps({'status':'passed_actual_masks','actual_masks_checked':count,'class_error_description':anatomy},indent=2))


if __name__=='__main__':main()
