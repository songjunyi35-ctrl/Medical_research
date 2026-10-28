"""Fresh270 neural execution of the portable API; GT is outside the predictor."""
import argparse,hashlib
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v1 import FrozenCorePredictor


def main(candidate,recipe):
    folder=R/'live_verification'/candidate;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'DONE.json').exists(),'live verification already complete'
    source=R/'candidates'/candidate;reference=read(source/'RAW_TEST270.json')['per_case']
    rows,geometry,images,labels=data();assert [r['case_id'] for r in rows]==[r['case_id'] for r in reference]
    torch.cuda.set_per_process_memory_fraction(.75)
    predictor=FrozenCorePredictor(recipe);config=read(recipe)
    write(folder/'FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),predictor_sha256=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/deploy_core_v1.py'),
          portable_recipe_sha256=sha(recipe),reference_raw_sha256=sha(source/'RAW_TEST270.json'),checkpoint_sha256=CHECKPOINT_SHA,
          GT_input_to_predictor=False,case_ids_input_to_predictor=False,training=False,test_tuned=True,role='fresh neural deployment equivalence, not retuning'))
    probes=[]
    for batch in [8,6]:
        guard();torch.cuda.reset_peak_memory_stats();synthetic=torch.zeros((batch,1,*CANVAS))
        values=predictor.predict(synthetic,[dict(canvas_offset=[0,0,0],extent=list(CANVAS))]*batch)
        assert len(values)==batch and all(v['mask'].shape==CANVAS and v['mask'].dtype==bool for v in values)
        probes.append(dict(batch=batch,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        del synthetic,values
    write(folder/'SYNTHETIC_GATE.json',dict(at=now(),status='passed',probes=probes,patient_model_forward=False))
    append_log('ACCESS_LEDGER.jsonl',dict(at=now(),job='portable_live_'+candidate,role='fresh270 deployment equivalence',
          model_case_forwards_expected=270*len(config['aggregation']['views']),training=False,test_tuned=True,GT_input=False))
    records=[];start=time.monotonic()
    for first in range(0,270,8):
        guard();slots=list(range(first,min(first+8,270)))
        image=torch.from_numpy(np.array(images[slots],np.float32))[:,None]
        result=predictor.predict(image,[geometry[rows[i]['case_id']] for i in slots])
        for i,value in zip(slots,result):
            cid=rows[i]['case_id'];mask=value['mask'];point=value['automatic_point'];saved=np.load(source/'predictions'/(cid+'.npz'))
            previous=np.unpackbits(saved['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
            different=int((mask!=previous).sum());equal_point=point==reference[i]['automatic_point']
            target=np.array(labels[i][slices(geometry[cid])]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
            score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
            path=folder/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
            records.append(dict(case_id=cid,subject_id=rows[i]['subject_id'],locator_dsc=score,automatic_point=point,
                  automatic_point_hits_gt=bool(target[tuple(point)]),mask_different_voxels=different,point_equal=equal_point,
                  dice_difference=score-reference[i]['locator_dsc'],prediction_sha256=sha(path),tp=tp,fp=pred-tp,fn=positive-tp))
        write(folder/'JOURNAL.json',dict(at=now(),per_case=records,seconds=time.monotonic()-start))
    assert len(records)==270
    differences=sum(r['mask_different_voxels'] for r in records);points=all(r['point_equal'] for r in records);max_delta=max(abs(r['dice_difference']) for r in records)
    result=dict(at=now(),state='passed' if differences==0 and points and max_delta==0 else 'numerical_mismatch',
          scans=270,mask_different_voxels=differences,all_points_equal=points,max_dice_difference=max_delta,
          scan_dice=float(np.mean([r['locator_dsc'] for r in records])),point_hits=sum(r['automatic_point_hits_gt'] for r in records),
          model_case_forwards=270*len(config['aggregation']['views']),seconds=time.monotonic()-start,GT_or_case_ID_input_to_API=False,
          raw_reference_sha256=sha(source/'RAW_TEST270.json'),training=False,test_tuned=True)
    write(folder/'RESULT.json',result)
    if result['state']=='passed':write(folder/'DONE.json',result)
    print(result);assert result['state']=='passed',result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--recipe',required=True);a=p.parse_args();main(a.candidate,a.recipe)
