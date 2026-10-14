"""Build the native-resolution (1mm, no resize) test270 canvas cache.  THIS OPENS TEST PIXELS (user-authorized 2026-09-20).

Per case: canonical NIfTI -> N4 shrink2 -> image-only nonzero brain crop -> nonzero z-score -> zero-pad centred
into TEST_CANVAS.  Each case is checked against the canonical 128^3 test cache: crop geometry exact, resized
label bit-exact, resized image within 1e-3.  PREACCESS.json is written before the first voxel is read.
"""
import sys,os,json,time,shutil,traceback,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from scripts.segresnet_native_res_test_common import *
from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays,_normalize_nonzero_t1ce,residual_tumor_target
from sammed3d.t1ce_wt_training import _load_canonical_nifti
from sammed3d.t1ce_wt_cached import MODEL_SHAPE
from scripts.build_t1ce_wt_cache import n4_correct
CFG={}


def init(config):
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    CFG.update(config)
    CFG['image_fd']=os.open(config['images'],os.O_WRONLY);CFG['label_fd']=os.open(config['labels'],os.O_WRONLY)
    CFG['canon_images']=np.memmap(config['canon_dir']+'/images.f32',dtype=np.float32,mode='r',shape=(TEST_SCANS,*MODEL_SHAPE))
    CFG['canon_labels']=np.memmap(config['canon_dir']+'/labels.u8',dtype=np.uint8,mode='r',shape=(TEST_SCANS,*MODEL_SHAPE))


def native_arrays(row):
    if row['split']!='test':raise ValueError('test builder only accepts test rows')
    img=_load_canonical_nifti(Path(row['modalities']['t1c']),'test T1ce');lab=_load_canonical_nifti(Path(row['seg_path']),'test label')
    if img.shape!=lab.shape or not np.allclose(img.affine,lab.affine,rtol=0,atol=1e-5):raise ValueError('raw geometry mismatch')
    spacing=[float(v) for v in nib.affines.voxel_sizes(img.affine)]
    image=np.ascontiguousarray(np.asarray(img.dataobj,dtype=np.float32));label=np.ascontiguousarray(np.asarray(lab.dataobj))
    if not np.isfinite(image).all():raise ValueError('nonfinite raw image')
    source=n4_correct(image,2);idx=np.nonzero(source>0)
    if any(x.size==0 for x in idx):raise ValueError('empty positive brain support')
    origin=[int(x.min()) for x in idx];stop=[int(x.max())+1 for x in idx];crop=tuple(slice(a,b) for a,b in zip(origin,stop))
    normalized,_,stats=_normalize_nonzero_t1ce(source[crop])
    full_target=residual_tumor_target(label);crop_target=full_target[crop];outside=int(full_target.sum())-int(crop_target.sum())
    view=prepare_t1ce_supervised_arrays(row['case_id'],source[crop],np.ascontiguousarray(label[crop]),spacing,output_shape=MODEL_SHAPE)
    canon_image=np.ascontiguousarray(view.image.numpy()[0],dtype=np.float32);canon_label=np.ascontiguousarray(view.target.numpy()[0],dtype=np.uint8)
    native_image=np.ascontiguousarray(normalized.numpy(),dtype=np.float32);native_label=np.ascontiguousarray(crop_target.numpy(),dtype=np.uint8)
    if not np.isfinite(native_image).all():raise ValueError('nonfinite native image')
    geometry={'origin':origin,'extent':[b-a for a,b in zip(origin,stop)],'native_shape':[int(v) for v in image.shape],'spacing_mm':spacing,'positive_voxels':int(native_label.sum()),'positives_outside_crop':outside,'normalization':{'nonzero_voxels':int(stats.nonzero_voxels),'mean':float(stats.mean),'std':float(stats.std)}}
    return native_image,native_label,canon_image,canon_label,geometry


def worker(task):
    slot,row,canon_geometry=task
    try:
        native_image,native_label,canon_image,canon_label,geometry=native_arrays(row)
        image_diff=float(np.abs(canon_image-np.asarray(CFG['canon_images'][slot])).max());label_exact=bool(np.array_equal(canon_label,CFG['canon_labels'][slot]))
        geometry_exact=(geometry['origin']==list(canon_geometry['origin']) and geometry['extent']==list(canon_geometry['extent']) and geometry['native_shape']==list(canon_geometry['native_shape']))
        padded_image,off=pad_to_canvas(native_image,0.);padded_label,off2=pad_to_canvas(native_label,0);assert off==off2
        n=int(np.prod(TEST_CANVAS))
        assert os.pwrite(CFG['image_fd'],padded_image.tobytes(),slot*n*4)==n*4
        assert os.pwrite(CFG['label_fd'],padded_label.tobytes(),slot*n)==n
        geometry.update(canvas_offset=off,canonical_image_within_tolerance=image_diff<=IMAGE_TOLERANCE,canonical_image_max_abs_diff=image_diff,canonical_label_exact=label_exact,canonical_geometry_exact=geometry_exact)
        return slot,row['case_id'],geometry,None
    except BaseException as exc:
        return slot,row['case_id'],None,repr(exc)+'\n'+traceback.format_exc()


def main():
    verify()
    if (R/'MODEL_PREACCESS.json').exists():raise RuntimeError('model already scored the test; no automatic repeat')
    rebuild=(R/'PREACCESS.json').exists()
    if rebuild:
        # Deterministic rebuild on another worker (scratch /tmp is per node). Payload hashes must match the first build.
        prior=json.loads((R/'CACHE_RECEIPT.json').read_text());import shutil as _sh;_sh.copy(R/'CACHE_RECEIPT.json',R/f"CACHE_RECEIPT_prior_{prior['utc'][:19].replace(':','')}.json")
    rows=read_rows(MANIFEST['test']);canon=json.loads((CANONICAL_TEST/'index.json').read_text())
    assert canon['manifest_sha256']==sha(MANIFEST['test']) and [str(x) for x in canon['case_ids']]==[r['case_id'] for r in rows] and canon['count']==TEST_SCANS
    assert len(rows)==TEST_SCANS and len({r['subject_id'] for r in rows})==TEST_SUBJECTS
    train_ids={r['subject_id'] for r in read_rows(MANIFEST['train'])};val_ids={r['subject_id'] for r in read_rows(MANIFEST['validation'])};test_ids={r['subject_id'] for r in rows}
    assert not (train_ids&test_ids) and not (val_ids&test_ids)
    TEST_CACHE.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(TEST_CACHE).free<TEST_SCANS*int(np.prod(TEST_CANVAS))*5+(4<<30):raise RuntimeError('worker scratch disk below required space')
    if not rebuild:write(R/'PREACCESS.json',{'utc':now(),'authorization':'user explicitly requested test270 evaluation of the native-resolution group (2026-09-20)','test_pixels_access':True,'candidate':'native-resolution pure SegResNet epoch173 (validation-selected, frozen 2026-09-18)','checkpoint_sha256':sha(CANDIDATE_CKPT),'baseline_checkpoint_sha256':BASELINE_CKPT_SHA,'freeze_sha256':sha(R/'FREEZE.json'),'selection_receipt_sha256':sha(R/'SELECTION_RECEIPT.json'),'cohort':{'scans':TEST_SCANS,'subjects':TEST_SUBJECTS},'test_canvas':list(TEST_CANVAS),'current_test_used_for_selection':False,'prior_test_informed_development':True,'new_outer_confirmation_access':False,'patient_isolation':True})
    n=int(np.prod(TEST_CANVAS))
    for name,per in [('images.f32',4),('labels.u8',1)]:
        p=TEST_CACHE/name
        if p.exists():p.unlink()
        with p.open('xb') as f:f.truncate(TEST_SCANS*n*per)
    config={'images':str(TEST_CACHE/'images.f32'),'labels':str(TEST_CACHE/'labels.u8'),'canon_dir':str(CANONICAL_TEST)}
    tasks=[(i,row,canon['crop_geometry'][row['case_id']]) for i,row in enumerate(rows)]
    tick=time.monotonic();geometry={};errors=[];processes=int(os.environ.get('AFLOW_BUILD_PROCESSES',60))
    with mp.get_context('spawn').Pool(processes,initializer=init,initargs=(config,)) as pool:
        for k,(slot,cid,geo,err) in enumerate(pool.imap_unordered(worker,tasks,chunksize=1),1):
            if err:errors.append({'case_id':cid,'error':err})
            else:geometry[cid]=geo
            if k%30==0 or k==TEST_SCANS:print(json.dumps({'utc':now(),'done':k,'of':TEST_SCANS,'errors':len(errors),'seconds':time.monotonic()-tick}),flush=True)
    if errors:raise RuntimeError(json.dumps(errors[:3],ensure_ascii=False))
    bad=[c for c,g in geometry.items() if not (g['canonical_image_within_tolerance'] and g['canonical_label_exact'] and g['canonical_geometry_exact'])]
    if bad:raise RuntimeError(f'{len(bad)} cases failed canonical reproduction: {bad[:5]}')
    if any(g['positive_voxels']<=0 for g in geometry.values()):raise RuntimeError('empty WT after crop')
    index={'schema':SCHEMA+'.test','schema_version':1,'split':'test','test_access':True,'canvas':list(TEST_CANVAS),'count':TEST_SCANS,'case_ids':[r['case_id'] for r in rows],'manifest':str(MANIFEST['test'].relative_to(ROOT)),'manifest_sha256':sha(MANIFEST['test']),'canonical_cache':str(CANONICAL_TEST.relative_to(ROOT)),'canonical_index_sha256':sha(CANONICAL_TEST/'index.json'),'image_dtype':'float32','label_dtype':'uint8',
           'preprocessing':{'spatial':'braincrop_native_canvas','spatial_description':'crop to the bounding box of nonzero voxels (image only, no GT), keep the native 1mm grid, zero-pad centred into a fixed 160x208x168 canvas (training canvas 160x208x160 extended by 8 on the last axis to fit the one 161-wide test crop); NO resize','image_interpolation':'none','label_interpolation':'none','normalization':'nonzero-voxel population z-score on the crop (identical to canonical)','uses_gt_for_localization':False,'n4_bias_field_correction':True,'n4_shrink_factor':2,'n4_implementation':'SimpleITK N4BiasFieldCorrectionImageFilter, 4x50 iterations, nonzero mask (identical to canonical)'},
           'geometry':geometry,'max_extent':[int(max(g['extent'][a] for g in geometry.values())) for a in range(3)],'positives_outside_crop_total':int(sum(g['positives_outside_crop'] for g in geometry.values())),'image_tolerance':IMAGE_TOLERANCE,'image_max_abs_diff_vs_canonical':float(max(g['canonical_image_max_abs_diff'] for g in geometry.values())),'all_canonical_labels_geometry_exact':True,'build_seconds':time.monotonic()-tick,'processes':processes,'builder':'scripts/build_segresnet_native_res_test_cache.py','builder_sha256':sha(Path(__file__))}
    index['images_sha256']=sha(TEST_CACHE/'images.f32');index['labels_sha256']=sha(TEST_CACHE/'labels.u8');write(TEST_CACHE/'index.json',index)
    if rebuild and (index['images_sha256']!=prior['images_sha256'] or index['labels_sha256']!=prior['labels_sha256']):raise RuntimeError('rebuilt cache payload differs from the first build')
    write(R/'CACHE_RECEIPT.json',{'utc':now(),'rebuild_of_prior_receipt':rebuild,'worker':os.environ.get('AFLOW_WORKER','unknown'),'cache':str(TEST_CACHE),'index_sha256':sha(TEST_CACHE/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'count':TEST_SCANS,'canvas':list(TEST_CANVAS),'max_extent':index['max_extent'],'positives_outside_crop_total':index['positives_outside_crop_total'],'image_max_abs_diff_vs_canonical':index['image_max_abs_diff_vs_canonical'],'all_canonical_labels_geometry_exact':True,'new_cache_bytes':(TEST_CACHE/'images.f32').stat().st_size+(TEST_CACHE/'labels.u8').stat().st_size,'build_seconds':index['build_seconds'],'canonical_test_cache_reused_for_baseline':True})
    print(json.dumps({k:v for k,v in index.items() if k!='geometry'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:write(R/'FAILURE_build.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc(),'test_access_attempted':(R/'PREACCESS.json').exists()});raise
