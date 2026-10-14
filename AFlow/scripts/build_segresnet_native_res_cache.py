"""Build native-resolution (1mm, no resize) fixed-canvas train/validation caches on the worker.

Per case: canonical NIfTI -> N4 (shrink2, identical to the canonical builder) -> image-only
nonzero brain crop -> nonzero-population z-score -> zero-pad centred into CANVAS.
Every case is checked against the canonical 128^3 cache: same crop geometry, and re-running the
canonical resize on our native crop reproduces the canonical label bit-exactly and the image within 1e-3.
Never opens test pixels.
"""
import sys,os,json,time,shutil,traceback,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from scripts.segresnet_native_res_common import *
from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays,_normalize_nonzero_t1ce,residual_tumor_target
from sammed3d.t1ce_wt_training import _load_canonical_nifti
from sammed3d.t1ce_wt_cached import MODEL_SHAPE
from scripts.build_t1ce_wt_cache import n4_correct
CFG={}
IMAGE_TOLERANCE=1e-3   # max abs z-score difference vs the canonical 128^3 image (N4 float noise across environments)


def init(config):
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    CFG.update(config)
    CFG['image_fd']=os.open(config['images'],os.O_WRONLY);CFG['label_fd']=os.open(config['labels'],os.O_WRONLY)
    CFG['canon_images']=np.memmap(config['canon_dir']+'/images.f32',dtype=np.float32,mode='r',shape=(config['canon_count'],*MODEL_SHAPE))
    CFG['canon_labels']=np.memmap(config['canon_dir']+'/labels.u8',dtype=np.uint8,mode='r',shape=(config['canon_count'],*MODEL_SHAPE))


def native_arrays(row):
    if row['split'] not in ('train','validation'):raise ValueError('only train/validation rows are allowed')
    if row.get('target')!='WT' or row.get('modality')!='t1ce' or sorted(row.get('foreground_labels',[]))!=[1,2,3] or row.get('excluded_labels')!=[4]:raise ValueError('unexpected manifest row semantics')
    img=_load_canonical_nifti(Path(row['image_path']),row['split']+' T1ce');lab=_load_canonical_nifti(Path(row['label_path']),row['split']+' label')
    if img.shape!=lab.shape or not np.allclose(img.affine,lab.affine,rtol=0,atol=1e-5):raise ValueError('raw geometry mismatch')
    spacing=[float(v) for v in nib.affines.voxel_sizes(img.affine)]
    image=np.ascontiguousarray(np.asarray(img.dataobj,dtype=np.float32));label=np.ascontiguousarray(np.asarray(lab.dataobj))
    if not np.isfinite(image).all():raise ValueError('nonfinite raw image')
    source=n4_correct(np.ascontiguousarray(image),2)
    idx=np.nonzero(source>0)
    if any(x.size==0 for x in idx):raise ValueError('empty positive brain support')
    origin=[int(x.min()) for x in idx];stop=[int(x.max())+1 for x in idx];crop=tuple(slice(a,b) for a,b in zip(origin,stop))
    normalized,_,stats=_normalize_nonzero_t1ce(source[crop])
    full_target=residual_tumor_target(label);crop_target=full_target[crop]
    outside=int(full_target.sum())-int(crop_target.sum())
    # Reproduce the canonical 128^3 view from exactly this crop (same function the canonical builder used).
    view=prepare_t1ce_supervised_arrays(row['case_id'],source[crop],np.ascontiguousarray(label[crop]),spacing,output_shape=MODEL_SHAPE)
    canon_image=np.ascontiguousarray(view.image.numpy()[0],dtype=np.float32);canon_label=np.ascontiguousarray(view.target.numpy()[0],dtype=np.uint8)
    native_image=np.ascontiguousarray(normalized.numpy(),dtype=np.float32);native_label=np.ascontiguousarray(crop_target.numpy(),dtype=np.uint8)
    if not np.isfinite(native_image).all():raise ValueError('nonfinite native image')
    geometry={'origin':origin,'extent':[b-a for a,b in zip(origin,stop)],'native_shape':[int(v) for v in image.shape],'spacing_mm':spacing,'positive_voxels':int(native_label.sum()),'positives_outside_crop':outside,'normalization':{'nonzero_voxels':int(stats.nonzero_voxels),'mean':float(stats.mean),'std':float(stats.std)}}
    return native_image,native_label,canon_image,canon_label,geometry


def worker(task):
    slot,row,canon_slot,canon_geometry=task
    try:
        native_image,native_label,canon_image,canon_label,geometry=native_arrays(row)
        image_diff=float(np.abs(canon_image-np.asarray(CFG['canon_images'][canon_slot])).max());image_exact=image_diff<=IMAGE_TOLERANCE;label_exact=bool(np.array_equal(canon_label,CFG['canon_labels'][canon_slot]))
        geometry_exact=(geometry['origin']==list(canon_geometry['origin']) and geometry['extent']==list(canon_geometry['extent']) and geometry['native_shape']==list(canon_geometry['native_shape']))
        padded_image,off=pad_to_canvas(native_image,0.);padded_label,off2=pad_to_canvas(native_label,0);assert off==off2
        n=int(np.prod(CANVAS))
        assert os.pwrite(CFG['image_fd'],padded_image.tobytes(),slot*n*4)==n*4
        assert os.pwrite(CFG['label_fd'],padded_label.tobytes(),slot*n)==n
        geometry.update(canvas_offset=off,canonical_image_within_tolerance=image_exact,canonical_image_max_abs_diff=image_diff,canonical_label_exact=label_exact,canonical_geometry_exact=geometry_exact)
        return slot,row['case_id'],geometry,None
    except BaseException as exc:
        return slot,row['case_id'],None,repr(exc)+'\n'+traceback.format_exc()


def build(split,processes):
    rows=read_rows(MANIFEST[split]);target=CACHE_ROOT/split;target.mkdir(parents=True,exist_ok=True)
    canon=json.loads((CANONICAL[split]/'index.json').read_text());canon_ids=[str(x) for x in canon['case_ids']]
    if sorted(canon_ids)!=sorted(r['case_id'] for r in rows):raise RuntimeError('canonical cache and manifest disagree')
    canon_slot={c:i for i,c in enumerate(canon_ids)}
    n=int(np.prod(CANVAS))
    for name,per in [('images.f32',4),('labels.u8',1)]:
        p=target/name
        if p.exists():p.unlink()
        with p.open('xb') as f:f.truncate(len(rows)*n*per)
    config={'images':str(target/'images.f32'),'labels':str(target/'labels.u8'),'canon_dir':str(CANONICAL[split]),'canon_count':int(canon['count'])}
    tasks=[(i,row,canon_slot[row['case_id']],canon['crop_geometry'][row['case_id']]) for i,row in enumerate(rows)]
    tick=time.monotonic();geometry={};errors=[]
    with mp.get_context('spawn').Pool(processes,initializer=init,initargs=(config,)) as pool:
        for k,(slot,cid,geo,err) in enumerate(pool.imap_unordered(worker,tasks,chunksize=1),1):
            if err:errors.append({'case_id':cid,'error':err})
            else:geometry[cid]=geo
            if k%50==0 or k==len(rows):print(json.dumps({'utc':now(),'split':split,'done':k,'of':len(rows),'errors':len(errors),'seconds':time.monotonic()-tick}),flush=True)
    if errors:raise RuntimeError(json.dumps(errors[:3],ensure_ascii=False))
    bad=[c for c,g in geometry.items() if not (g['canonical_image_within_tolerance'] and g['canonical_label_exact'] and g['canonical_geometry_exact'])]
    if bad:raise RuntimeError(f'{len(bad)} cases failed canonical reproduction: {bad[:5]}')
    if any(g['positive_voxels']<=0 for g in geometry.values()):raise RuntimeError('empty WT after crop')
    index={'schema':SCHEMA,'schema_version':1,'split':split,'test_access':False,'canvas':list(CANVAS),'count':len(rows),'case_ids':[r['case_id'] for r in rows],'manifest':str(MANIFEST[split].relative_to(ROOT)),'manifest_sha256':sha(MANIFEST[split]),
           'canonical_cache':str(CANONICAL[split].relative_to(ROOT)),'canonical_index_sha256':sha(CANONICAL[split]/'index.json'),'image_dtype':'float32','label_dtype':'uint8',
           'preprocessing':{'spatial':'braincrop_native_canvas','spatial_description':'crop to the bounding box of nonzero voxels (image only, no GT), keep the native 1mm grid, zero-pad centred into a fixed 160x208x160 canvas; NO resize','image_interpolation':'none','label_interpolation':'none','normalization':'nonzero-voxel population z-score on the crop (identical to canonical)','uses_gt_for_localization':False,'n4_bias_field_correction':True,'n4_shrink_factor':2,'n4_implementation':'SimpleITK N4BiasFieldCorrectionImageFilter, 4x50 iterations, nonzero mask (identical to canonical)','augmentation':'NOT cached; applied per-batch on GPU'},
           'geometry':geometry,'max_extent':[int(max(g['extent'][a] for g in geometry.values())) for a in range(3)],'positives_outside_crop_total':int(sum(g['positives_outside_crop'] for g in geometry.values())),
           'all_canonical_labels_geometry_exact':True,'image_tolerance':IMAGE_TOLERANCE,'image_max_abs_diff_vs_canonical':float(max(g['canonical_image_max_abs_diff'] for g in geometry.values())),'image_tolerance_note':'canonical N4 was built in another environment; residual float noise (~1e-4 in z-score units) is accepted, labels and crop geometry must be bit-exact','build_seconds':time.monotonic()-tick,'processes':processes,'builder':'scripts/build_segresnet_native_res_cache.py','builder_sha256':sha(Path(__file__))}
    index['images_sha256']=sha(target/'images.f32');index['labels_sha256']=sha(target/'labels.u8')
    write(target/'index.json',index)
    return {'role':split,'path':str(target),'count':len(rows),'index_sha256':sha(target/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'bytes':(target/'images.f32').stat().st_size+(target/'labels.u8').stat().st_size,'all_canonical_labels_geometry_exact':True,'image_max_abs_diff_vs_canonical':index['image_max_abs_diff_vs_canonical'],'max_extent':index['max_extent'],'positives_outside_crop_total':index['positives_outside_crop_total'],'build_seconds':index['build_seconds']}


def main():
    verify()
    CACHE_ROOT.mkdir(parents=True,exist_ok=True)
    need=1080*int(np.prod(CANVAS))*5+(8<<30)
    if shutil.disk_usage(CACHE_ROOT).free<need:raise RuntimeError('worker scratch disk below required space')
    processes=int(os.environ.get('AFLOW_BUILD_PROCESSES',48))
    write(R/'DATA_PREACCESS.json',{'utc':now(),'roles_opened':['train','validation'],'test_pixels_access':False,'manifest_sha256':{k:sha(v) for k,v in MANIFEST.items()},'freeze_sha256':sha(R/'FREEZE.json')})
    train=read_rows(MANIFEST['train']);val=read_rows(MANIFEST['validation']);test=read_rows(MANIFEST['test'])
    ts={r['subject_id'] for r in train};vs={r['subject_id'] for r in val};xs={r['subject_id'] for r in test}
    if ts&vs or ts&xs or vs&xs:raise RuntimeError('patient isolation violated')
    if len(train)!=972 or len(ts)!=444 or len(val)!=108 or len(vs)!=50 or len(test)!=270 or len(xs)!=119:raise RuntimeError('unexpected split sizes')
    caches=[build('train',processes),build('validation',processes)]
    write(R/'DATA_AUDIT.json',{'utc':now(),'status':'passed','train_case_ids':[r['case_id'] for r in train],'validation_case_ids':[r['case_id'] for r in val],'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'test_metadata_only':{'scans':270,'subjects':119,'sha256':sha(MANIFEST['test'])},'patient_isolation':True,'canvas':list(CANVAS),'caches':caches,'all_canonical_labels_geometry_exact':True,'image_tolerance':IMAGE_TOLERANCE,'new_test_pixels_access':False})
    print(json.dumps(caches),flush=True)


if __name__=='__main__':main()
