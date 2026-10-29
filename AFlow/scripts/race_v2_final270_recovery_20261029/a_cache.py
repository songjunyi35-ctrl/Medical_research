"""Single-N4 rebuild of four historical payloads, on the newly assigned node."""
import multiprocessing as mp,time,traceback
import numpy as np
from scripts.race_v2_final270_recovery_20261029.a_common import *
from scripts.segresnet_w32_test270_20261028 import cache as historical
from scripts.race_seg_20261028.prepare import fit_plane

def worker(task):
    cid,g,error=historical.worker(task)
    if error:return cid,None,error
    try:
        import nibabel as nib
        slot,row,_=task
        image=np.memmap(CACHE/'images.f32',np.float32,'r',offset=slot*160*208*168*4,shape=(160,208,168))
        header=nib.as_closest_canonical(nib.load(row['modalities']['t1c']))
        directions=header.affine[:3,:3]/np.linalg.norm(header.affine[:3,:3],axis=0)
        assert nib.aff2axcodes(header.affine)==('R','A','S') and np.allclose(directions,np.eye(3),atol=1e-4)
        assert np.allclose(nib.affines.voxel_sizes(header.affine),[1,1,1],atol=1e-5)
        assert g['canvas_offset']==[(c-e)//2 for c,e in zip((160,208,168),g['extent'])]
        center,score=fit_plane(image);matrix=np.eye(4,dtype=np.float32);matrix[0,0]=-1;matrix[0,3]=2*center
        g.update(reflection=matrix.tolist(),reference_enabled=bool(score>=.8),image_support_symmetry_dice=score,canonical_geometry_exact=True)
        return cid,g,None
    except BaseException as error:return cid,None,repr(error)+'\n'+traceback.format_exc()

def main():
    verify();R.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    assert not CACHE.exists() and not CANON.exists(),'no implicit payload overwrite or retry'
    cases=cohort();reference=read(CANON_INDEX);assert reference['case_ids']==[r['case_id'] for r in cases]
    write(R/'CACHE_PIXEL_PREACCESS.json',dict(at=now(),worker=1154259,selection_sha256=SELECTION_SHA,
        original_worker=1154121,original_worker_lost=True,bounded_reconstruction=True,
        original_payload_reuse=False,N4_calls_planned=270,patient_network_forwards=0,
        authorization_sha256=sha(BASE/'AUTHORIZATION.json'),source_freeze_sha256=sha(R/'EXECUTION_FREEZE.json')))
    config={}
    for folder,shape,prefix in [(CACHE,(160,208,168),'native'),(CANON,(128,128,128),'canon')]:
        folder.mkdir(parents=True)
        for name,width,kind in [('images.f32',4,'image'),('labels.u8',1,'label')]:
            path=folder/name
            with path.open('xb') as out:out.truncate(270*int(np.prod(shape))*width)
            config[prefix+'_'+kind]=str(path)
    tasks=[(slot,row,reference['crop_geometry'][row['case_id']]) for slot,row in enumerate(cases)]
    started=time.monotonic();geometry={}
    with mp.get_context('spawn').Pool(23,initializer=historical.init,initargs=(config,)) as pool:
        for done,(cid,g,error) in enumerate(pool.imap_unordered(worker,tasks,chunksize=1),1):
            guard()
            if error:raise RuntimeError(error)
            geometry[cid]=g
            write(R/'CACHE_HEARTBEAT.json',dict(at=now(),completed=done,total=270,processes=23,seconds=time.monotonic()-started))
    actual={key:sha(path) for key,path in config.items()};expected=read(OLD/'CACHE_READY.json')['payload_sha256']
    write(R/'PAYLOAD_SHA_COMPARISON.json',dict(at=now(),expected=expected,actual=actual,all_exact=actual==expected))
    assert actual==expected,'four payload SHA mismatch; stop without model access'
    assert sum(g['positives_outside_crop'] for g in geometry.values())==4
    for row in cases:
        g=geometry[row['case_id']];old=reference['crop_geometry'][row['case_id']]
        assert all(g[key]==old[key] for key in ('origin','extent','native_shape'))
    # Deterministic order is intentional. The old index mixed1 reused metadata
    # row and269 fresh rows in asynchronous completion order; do not fake its SHA.
    index=dict(schema='aflow.final270.recovery.native.v1',split='test',test_access=True,count=270,canvas=[160,208,168],
        case_ids=[r['case_id'] for r in cases],geometry={r['case_id']:geometry[r['case_id']] for r in cases},
        preprocessing=dict(uses_gt_for_localization=False,n4_shrink_factor=2,normalization='historical native zscore',padding='centered zero'),
        images_sha256=actual['native_image'],labels_sha256=actual['native_label'],historical_payload_exact=True)
    write(CACHE/'index.json',index)
    write(CANON/'index.json',dict(schema='recovery.canonical128.payload_receipt',case_ids=index['case_ids'],count=270,
        images_sha256=actual['canon_image'],labels_sha256=actual['canon_label'],source_geometry_index_path=str(CANON_INDEX),source_geometry_index_sha256=sha(CANON_INDEX)))
    for folder in (CACHE,CANON):
        for p in folder.iterdir():p.chmod(0o444)
        folder.chmod(0o555)
    write(R/'CACHE_READY.json',dict(at=now(),status='passed',worker=1154259,cache_path=str(CACHE),canonical_cache_path=str(CANON),
        index_sha256=sha(CACHE/'index.json'),payload_sha256=actual,all_four_payloads_exact=True,
        historical_index_sha256=read(OLD/'CACHE_READY.json')['index_sha256'],index_byte_sha_reuse_claim=False,
        geometry_against_canonical_exact=True,reflection_recomputed_by_exact_image_only_fit_plane=True,
        disabled_references=sum(not g['reference_enabled'] for g in geometry.values()),canvas=[160,208,168],
        manifest_sha256=sha(MANIFEST),scans=270,subjects=119,N4_calls=270,seconds=time.monotonic()-started,
        selection_sha256=SELECTION_SHA,source_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),
        bounded_reconstruction_after_worker_loss=True,original_tmp_cache_available=False,read_only_for_B=True))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(R/f'CACHE_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
