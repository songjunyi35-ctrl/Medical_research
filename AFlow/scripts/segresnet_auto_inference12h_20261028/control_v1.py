"""Rebuild the unchanged old five-view protocol as a same-resource control."""
import time
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *


def execute(model,job):
    verify_job(job)
    from scripts.segresnet_inference_search_20261028 import deploy
    deploy.CANVAS=CANVAS
    selected=deploy.SelectedPredictor()
    resource=read(R/'RESOURCE_CURRENT.json')
    gate_dir=R/'resource_baselines'/resource['token'];gate_dir.mkdir(parents=True,exist_ok=True)
    if not (gate_dir/'SYNTHETIC_GATE.json').exists():
        probes=[]
        with torch.inference_mode():
            for batch in [8,6]:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                synthetic=torch.zeros((batch,1,*CANVAS),device='cuda')
                outputs=selected.predict(synthetic,[dict(canvas_offset=[0,0,0],extent=list(CANVAS))]*batch)
                assert len(outputs)==batch and all(d['mask'].shape==CANVAS for d in outputs)
                probes.append(dict(batch=batch,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
                del synthetic,outputs
        write(gate_dir/'SYNTHETIC_GATE.json',dict(at=now(),status='passed',tests=probes,patient_model_forward=False,
              checkpoint_sha256=CHECKPOINT_SHA,resource_token=resource['token']))
    rows,geometry,images,labels=data();folder=R/'resource_baselines'/resource['token'];folder.mkdir(parents=True,exist_ok=True)
    old={d['case_id']:d for d in read(OLD/'RAW_TEST270.json')['per_case']}
    append_log('ACCESS_LEDGER.jsonl',dict(at=now(),job=job['id'],role='same_resource_old_protocol_control',
          case_forwards_expected=1350,training=False,test_tuned=True,reason='resource migration, not a new method'))
    records=[];files=[];start=time.monotonic()
    with torch.inference_mode():
        for begin in range(0,270,8):
            guard();slots=list(range(begin,min(begin+8,270)))
            image=torch.from_numpy(np.array(images[slots],np.float32))[:,None]
            result=selected.predict(image,[geometry[rows[i]['case_id']] for i in slots])
            for i,value in zip(slots,result):
                cid=rows[i]['case_id'];mask=value['mask'];target=np.array(labels[i][slices(geometry[cid])]>0)
                point=value['automatic_point'];tp,fp,fn=[int(a.sum()) for a in [mask&target,mask&~target,~mask&target]]
                den=2*tp+fp+fn;score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
                saved=np.load(OLD/'predictions'/(cid+'.npz'))
                previous=np.unpackbits(saved['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
                path=folder/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
                records.append(dict(case_id=cid,subject_id=rows[i]['subject_id'],locator_dsc=score,
                      automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),tp=tp,fp=fp,fn=fn,
                      source_slot=i,historical_dice_difference=score-old[cid]['locator_dsc'],
                      historical_mask_different_voxels=int((mask!=previous).sum()),historical_point_equal=point==old[cid]['automatic_point']))
                files.append(dict(case_id=cid,path=str(path.relative_to(ROOT)),sha256=sha(path)))
            write(folder/'JOURNAL.json',dict(at=now(),per_case=records,predictions=files))
            status('same_resource_control',completed=len(records),total=270,seconds=time.monotonic()-start)
    raw=dict(at=now(),per_case=records,predictions=files,resource=resource,original_recipe_sha256=sha(OLD/'SELECTED_RECIPE.json'),
          role='unchanged old protocol on current resource',model_case_forwards=1350,test_tuned=True)
    write(folder/'RAW_TEST270.json',raw)
    write(R/'CURRENT_BASELINE.json',dict(at=now(),folder=str(folder.relative_to(ROOT)),raw_sha256=sha(folder/'RAW_TEST270.json'),
          scan_dice=float(np.mean([d['locator_dsc'] for d in records])),
          max_historical_dice_difference=max(abs(d['historical_dice_difference']) for d in records),
          historical_mask_different_voxels=sum(d['historical_mask_different_voxels'] for d in records),
          all_historical_points_equal=all(d['historical_point_equal'] for d in records),resource_token=resource['token']))
    write(R/'jobs'/job['id']/'DONE.json',dict(at=now(),state='complete',role='same_resource_control',seconds=time.monotonic()-start,
          scans=270,case_forwards=1350,resource_token=resource['token']))
