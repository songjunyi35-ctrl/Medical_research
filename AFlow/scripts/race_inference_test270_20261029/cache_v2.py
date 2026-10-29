"""Recover payloads from a header-field preparation failure; no model access."""
import multiprocessing as mp
import numpy as np
from scripts.race_inference_test270_20261029.common import *
from scripts.segresnet_w32_test270_20261028 import cache as historical
from scripts.race_seg_20261028.prepare import fit_plane


def image_geometry(slot,row,g=None):
    import nibabel as nib
    image=np.memmap(CACHE/'images.f32',np.float32,'r',offset=slot*int(np.prod(CANVAS))*4,shape=CANVAS)
    header=nib.as_closest_canonical(nib.load(row['modalities']['t1c']))
    directions=header.affine[:3,:3]/np.linalg.norm(header.affine[:3,:3],axis=0)
    assert nib.aff2axcodes(header.affine)==('R','A','S') and np.allclose(directions,np.eye(3),atol=1e-4)
    if g is None:
        from scripts.build_segresnet_native_res_test_cache import residual_tumor_target
        reference=read(CANON_INDEX)['crop_geometry'][row['case_id']]
        target=np.memmap(CACHE/'labels.u8',np.uint8,'r',offset=slot*int(np.prod(CANVAS)),shape=CANVAS)
        label=nib.as_closest_canonical(nib.load(row['seg_path']))
        full=residual_tumor_target(np.asarray(label.dataobj))
        g=dict(reference,spacing_mm=nib.affines.voxel_sizes(header.affine).tolist(),positive_voxels=int(target.sum()),
               positives_outside_crop=int(full.sum())-int(target.sum()),
               normalization=dict(provenance='reused historical normalized payload; original raw mean/std not reconstructed'))
    g['canvas_offset']=[(c-e)//2 for c,e in zip(CANVAS,g['extent'])]
    center,score=fit_plane(image)
    matrix=np.eye(4,dtype=np.float32);matrix[0,0]=-1;matrix[0,3]=2*center
    g.update(reflection=matrix.tolist(),reference_enabled=bool(score>=.8),image_support_symmetry_dice=score,
             canonical_geometry_exact=True)
    return g


def worker(task):
    cid,g,error=historical.worker(task)
    if error:return cid,None,error
    slot,row,_=task
    return cid,image_geometry(slot,row,g),None


def main():
    verify('PREPARATION_FREEZE_V2.json');manifests()
    cases=rows();canonical=read(CANON_INDEX);geometry={};remaining=[];started=time.monotonic()
    arrays={key:np.memmap(directory/name,dtype=dtype,mode='r',shape=(270,*shape))
            for directory,shape,prefix in [(CACHE,CANVAS,'native'),(CANON,(128,128,128),'canon')]
            for name,dtype,key in [('images.f32',np.float32,prefix+'_image'),('labels.u8',np.uint8,prefix+'_label')]}
    for slot,row in enumerate(cases):
        # All test crops have nonempty WT in both historical grids. Final whole-
        # payload SHA validation also detects any incomplete writes selected here.
        complete=all(bool(np.any(array[slot])) for array in arrays.values())
        if complete:geometry[row['case_id']]=image_geometry(slot,row)
        else:remaining.append((slot,row,canonical['crop_geometry'][row['case_id']]))
    write(R/'CACHE_RECOVERY.json',dict(at=now(),reused_complete_payload_cases=len(geometry),remaining_cases=len(remaining),
          prior_failure='CACHE_FAILURE.json',model_forwards=0,
          N4_accounting='one N4 for each scheduled v2 case; v1 cancelled in-flight N4 attempts not precisely counted; no total-once claim'))
    config=dict(native_image=str(CACHE/'images.f32'),native_label=str(CACHE/'labels.u8'),
                canon_image=str(CANON/'images.f32'),canon_label=str(CANON/'labels.u8'))
    with mp.get_context('spawn').Pool(23,initializer=historical.init,initargs=(config,)) as pool:
        for cid,g,error in pool.imap_unordered(worker,remaining,chunksize=1):
            guard()
            if error:raise RuntimeError(error)
            geometry[cid]=g
            write(R/'CACHE_HEARTBEAT.json',dict(at=now(),state='building_cache_v2',completed=len(geometry),total=270,seconds=time.monotonic()-started))
    actual={k:sha(v) for k,v in config.items()}
    assert actual==read(HISTORICAL_CACHE)['payload_sha256'],'historical four payloads must match exactly'
    assert sum(g['positives_outside_crop'] for g in geometry.values())==4
    index=dict(schema=SCHEMA+'.test',split='test',test_access=True,canvas=list(CANVAS),count=270,
          case_ids=[d['case_id'] for d in cases],geometry=geometry,
          preprocessing=dict(uses_gt_for_localization=False,n4_shrink_factor=2,normalization='historical native zscore',padding='centered zero'),
          images_sha256=actual['native_image'],labels_sha256=actual['native_label'])
    write(CACHE/'index.json',index)
    write(R/'CACHE_READY.json',dict(at=now(),status='passed',all_four_payloads_exact=True,payload_sha256=actual,
          index_sha256=sha(CACHE/'index.json'),derived_reflection_metadata_added=True,
          disabled_references=sum(not g['reference_enabled'] for g in geometry.values()),
          seconds=time.monotonic()-started,processes=23,scans=270,subjects=119,recovery='CACHE_RECOVERY.json'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(R/'CACHE_FAILURE_V2.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc()));raise
