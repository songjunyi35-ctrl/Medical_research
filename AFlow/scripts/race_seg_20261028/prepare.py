"""One N4/case yields both historical views; new RC labels only on train.

No validation class4 target or test pixels are loaded. Plane is image-only.
"""
import concurrent.futures
import multiprocessing as mp
import time
import numpy as np
from scripts.race_seg_20261028.common import *


def fit_plane(image):
    support=image[::2,::2,::2]!=0
    positions=np.flatnonzero(support.any((1,2)))
    if len(positions)<8: raise ValueError('insufficient image support')
    middle=(int(positions[0])+int(positions[-1]))/2
    best=None
    for doubled in range(round(2*middle)-6,round(2*middle)+7):
        indices=doubled-np.arange(len(support))
        inside=(indices>=0)&(indices<len(support))
        reflected=np.zeros_like(support);reflected[inside]=support[indices[inside]]
        overlap=2*np.count_nonzero(support&reflected)/max(1,np.count_nonzero(support)+np.count_nonzero(reflected))
        candidate=(overlap,-abs(doubled/2-middle),-doubled)
        if best is None or candidate>best[0]: best=(candidate,float(doubled),overlap)
    # Downsample voxel j corresponds to input2*j, so c_input=doubled.
    return best[1],best[2]


def process_case(task):
    split,slot,row,target=task
    import torch
    import nibabel as nib
    import SimpleITK as sitk
    torch.set_num_threads(1);sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    from scripts.nnunet_local_full_20261022.native_builder_v2 import native_arrays
    x,y,cx,cy,geometry=native_arrays(row)
    reference=read(ROOT/('workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json'
                        if split=='train' else 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json'))
    if geometry['origin']!=reference['crop_geometry'][row['case_id']]['origin'] or geometry['extent']!=reference['crop_geometry'][row['case_id']]['extent']:
        raise RuntimeError('historical crop geometry mismatch')
    # Header orientation verifies which voxel dimension is anatomical left/right.
    nii=nib.as_closest_canonical(nib.load(row['image_path']))
    directions=nii.affine[:3,:3]/np.linalg.norm(nii.affine[:3,:3],axis=0)
    if nib.aff2axcodes(nii.affine)!=('R','A','S') or not np.allclose(directions,np.eye(3),atol=1e-4):
        raise ValueError('requires axis-aligned canonical RAS geometry')
    offsets=[(c-e)//2 for c,e in zip(CANVAS,x.shape)]
    if any(v<0 for v in offsets): raise ValueError('native crop exceeds canvas')
    slices=tuple(slice(o,o+e) for o,e in zip(offsets,x.shape))
    px=np.zeros(CANVAS,np.float32);py=np.zeros(CANVAS,np.uint8)
    px[slices]=x;py[slices]=y
    center,score=fit_plane(px)
    reflection=np.eye(4,dtype=np.float32);reflection[0,0]=-1;reflection[0,3]=2*center
    for name,value in [('images.f32',px),('labels.u8',py),('canonical_images.f32',cx),('canonical_labels.u8',cy)]:
        fd=os.open(str(Path(target)/name),os.O_WRONLY)
        try:
            body=np.ascontiguousarray(value).tobytes()
            if os.pwrite(fd,body,slot*len(body))!=len(body): raise IOError('short cache write')
        finally: os.close(fd)
    if split=='train':
        label=nib.as_closest_canonical(nib.load(row['label_path']))
        if label.shape!=nii.shape or not np.allclose(label.affine,nii.affine,atol=1e-5,rtol=0): raise ValueError('label geometry mismatch')
        raw=np.asarray(label.dataobj)
        if not np.isfinite(raw).all() or not np.allclose(raw,np.rint(raw),rtol=0,atol=1e-5) or not set(np.unique(raw)).issubset({0,1,2,3,4}): raise ValueError('class label domain')
        crop=tuple(slice(o,o+e) for o,e in zip(geometry['origin'],geometry['extent']))
        typed=np.zeros(CANVAS,np.uint8);typed[slices]=np.where(np.isin(raw[crop],[1,2,3]),1,np.where(raw[crop]==4,2,0)).astype(np.uint8)
        if not np.array_equal(typed==1,py>0): raise ValueError('typed WT differs from historical binary target')
        fd=os.open(str(Path(target)/'classes.u8'),os.O_WRONLY)
        try: os.pwrite(fd,typed.tobytes(),slot*typed.size)
        finally: os.close(fd)
    geometry['canvas_offset']=offsets
    geometry['reflection']=reflection.tolist()
    geometry['image_support_symmetry_dice']=score
    geometry['reference_enabled']=bool(score>=.8)
    return row['case_id'],geometry


def main():
    verify_sources('PREPARATION_FREEZE.json');disk();deadline();validate_manifests()
    R.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    with (R/'DATA_PREACCESS.json').open('x') as f:
        json.dump({'at':now(),'train':972,'validation':108,'training_class4_only':True,'new_test_pixels':False,'one_n4_per_case':True},f)
    if shutil.disk_usage(TMP).free<100*2**30: raise RuntimeError('scratch below100GiB')
    historical={x['role']:x for x in read(ROOT/'archive/research_records/2026-10-26_segresnet_teacher_distill/NATIVE_DATA_AUDIT.json')['caches']}
    for split in ('train','validation'):
        deadline();records=rows(split);folder=CACHE/split;folder.mkdir(parents=True,exist_ok=True)
        count=len(records);vol=int(np.prod(CANVAS))
        sizes={'images.f32':vol*4,'labels.u8':vol,'canonical_images.f32':128**3*4,'canonical_labels.u8':128**3}
        if split=='train': sizes['classes.u8']=vol
        for name,size in sizes.items():
            with (folder/name).open('xb') as f:f.truncate(count*size)
        geometries={};started=time.monotonic()
        tasks=[(split,i,row,str(folder)) for i,row in enumerate(records)]
        with concurrent.futures.ProcessPoolExecutor(16,mp_context=mp.get_context('spawn')) as pool:
            for cid,geometry in pool.map(process_case,tasks):
                geometries[cid]=geometry
                if len(geometries)%25==0 or len(geometries)==count:
                    write(R/'PREPARE_HEARTBEAT.json',{'at':now(),'split':split,'done':len(geometries),'count':count,'seconds':time.monotonic()-started})
        digests={name:sha(folder/name) for name in sizes}
        for name,key in [('images.f32','images_sha256'),('labels.u8','labels_sha256')]:
            if digests[name]!=historical[split][key]: raise RuntimeError('historical native SHA mismatch '+split+' '+name)
        canonical_path=ROOT/('workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json' if split=='train' else 'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108/index.json')
        old=read(canonical_path)
        if digests['canonical_labels.u8']!=old['labels_sha256'] or digests['canonical_images.f32']!=old['images_sha256']:
            raise RuntimeError('historical canonical128 SHA mismatch '+split)
        disabled=sum(not g['reference_enabled'] for g in geometries.values())
        if disabled/count>.05: raise RuntimeError('more than5percent image-reflection failures')
        index={'schema':'aflow.race.native','split':split,'count':count,'canvas':list(CANVAS),'case_ids':[r['case_id'] for r in records],'geometry':geometries,'digests':digests,'test_access':False,'reference_disabled':disabled,'historical_native_sha_exact':True,'historical128_sha_exact':True}
        write(folder/'index.json',index);write(R/f'DATA_{split}.json',index)
    write(R/'DATA_READY.json',{'at':now(),'status':'passed','historical_payloads_exact':True,'train':972,'validation':108,'test_pixels':False})

if __name__=='__main__':
    try: main()
    except BaseException as error:
        write(R/'PREPARE_FAILURE.json',{'at':now(),'error':repr(error)});raise
