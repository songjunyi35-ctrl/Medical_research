"""Rebuild exact historical four payloads; reflection fitted from image only."""
import multiprocessing as mp
import traceback
import numpy as np
from scripts.race_inference_test270_20261029.common import *
from scripts.segresnet_w32_test270_20261028 import cache as historical
from scripts.race_seg_20261028.prepare import fit_plane


def worker(task):
    cid,g,error=historical.worker(task)
    if error:return cid,None,error
    try:
        import nibabel as nib
        slot,row,_=task
        image=np.memmap(CACHE/'images.f32',np.float32,'r',offset=slot*int(np.prod(CANVAS))*4,shape=CANVAS)
        center,score=fit_plane(image)
        header=nib.as_closest_canonical(nib.load(row['image_path']))
        directions=header.affine[:3,:3]/np.linalg.norm(header.affine[:3,:3],axis=0)
        assert nib.aff2axcodes(header.affine)==('R','A','S') and np.allclose(directions,np.eye(3),atol=1e-4)
        matrix=np.eye(4,dtype=np.float32);matrix[0,0]=-1;matrix[0,3]=2*center
        g.update(reflection=matrix.tolist(),reference_enabled=bool(score>=.8),image_support_symmetry_dice=score)
        return cid,g,None
    except BaseException as error:return cid,None,repr(error)+'\n'+traceback.format_exc()


def main():
    verify('PREPARATION_FREEZE.json');manifests();TMP.mkdir(parents=True,exist_ok=True)
    assert not CACHE.exists() and not CANON.exists(),'refuse overwriting prior cache'
    exclusive(R/'CACHE_PIXEL_PREACCESS.json',dict(at=now(),new_test_pixels=True,patient_model_forwards=0,
              one_N4_per_case=True,reflection_from_images_only=True))
    config={}
    for directory,shape,prefix in [(CACHE,CANVAS,'native'),(CANON,(128,128,128),'canon')]:
        directory.mkdir()
        for name,width,kind in [('images.f32',4,'image'),('labels.u8',1,'label')]:
            path=directory/name
            with path.open('xb') as f:f.truncate(270*int(np.prod(shape))*width)
            config[prefix+'_'+kind]=str(path)
    original=read(CANON_INDEX);cases=rows();geometry={};started=time.monotonic()
    tasks=[(i,row,original['crop_geometry'][row['case_id']]) for i,row in enumerate(cases)]
    with mp.get_context('spawn').Pool(23,initializer=historical.init,initargs=(config,)) as pool:
        for done,(cid,g,error) in enumerate(pool.imap_unordered(worker,tasks,chunksize=1),1):
            guard()
            if error:raise RuntimeError(error)
            geometry[cid]=g
            if done%15==0 or done==270:
                write(R/'CACHE_HEARTBEAT.json',dict(at=now(),completed=done,total=270,seconds=time.monotonic()-started))
    actual={k:sha(v) for k,v in config.items()}
    assert actual==read(HISTORICAL_CACHE)['payload_sha256'],'historical payload mismatch'
    assert sum(g['positives_outside_crop'] for g in geometry.values())==4
    index=dict(schema=SCHEMA+'.test',split='test',test_access=True,canvas=list(CANVAS),count=270,
          case_ids=[d['case_id'] for d in cases],geometry=geometry,
          preprocessing=dict(uses_gt_for_localization=False,n4_shrink_factor=2,normalization='historical native zscore',padding='centered zero'),
          images_sha256=actual['native_image'],labels_sha256=actual['native_label'])
    write(CACHE/'index.json',index)
    write(R/'CACHE_READY.json',dict(at=now(),status='passed',all_four_payloads_exact=True,payload_sha256=actual,
          index_sha256=sha(CACHE/'index.json'),derived_reflection_metadata_added=True,
          disabled_references=sum(not g['reference_enabled'] for g in geometry.values()),
          one_N4_per_case=True,seconds=time.monotonic()-started,processes=23,scans=270,subjects=119))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(R/'CACHE_FAILURE.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc()));raise
