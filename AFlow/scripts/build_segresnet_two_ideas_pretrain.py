"""Build the 128^3 pretraining cache for the 334 OpenNeuro cases with EXACTLY the canonical project preprocessing:
canonical RAS -> N4 (shrink 2, 4x50, nonzero mask) -> image-only nonzero brain crop -> nonzero z-score ->
trilinear image / nearest-exact label resize to 128^3; WT = labels {1,2,3} (ds007045 has no label 4).
The index is written in the canonical cache schema so the frozen BrainCropDataset reads it unchanged.
"""
import sys,os,json,time,shutil,traceback,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from scripts.segresnet_two_ideas_common import *
from sammed3d.t1ce_binary import prepare_t1ce_supervised_arrays
from sammed3d.t1ce_wt_training import _load_canonical_nifti
from scripts.build_t1ce_wt_cache import n4_correct
CFG={}


def init(config):
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    CFG.update(config);CFG['image_fd']=os.open(config['images'],os.O_WRONLY);CFG['label_fd']=os.open(config['labels'],os.O_WRONLY)


def arrays(row):
    if sha(row['image_path'])!=row['image_sha256'] or sha(row['label_path'])!=row['label_sha256']:raise ValueError('source file hash changed since download')
    img=_load_canonical_nifti(Path(row['image_path']),'pretrain T1ce');lab=_load_canonical_nifti(Path(row['label_path']),'pretrain label')
    if img.shape!=lab.shape or not np.allclose(img.affine,lab.affine,rtol=0,atol=1e-5):raise ValueError('raw geometry mismatch')
    spacing=[float(v) for v in nib.affines.voxel_sizes(img.affine)]
    image=np.ascontiguousarray(np.asarray(img.dataobj,dtype=np.float32));label=np.ascontiguousarray(np.asarray(lab.dataobj))
    if not np.isfinite(image).all():raise ValueError('nonfinite raw image')
    labels=set(np.unique(label).tolist())
    if not labels<={0,1,2,3}:raise ValueError(f'unexpected labels {sorted(labels)}')
    source=n4_correct(np.where(image>0,image,0).astype(np.float32),2);idx=np.nonzero(source>0)
    if any(x.size==0 for x in idx):raise ValueError('empty positive brain support')
    origin=[int(x.min()) for x in idx];stop=[int(x.max())+1 for x in idx];crop=tuple(slice(a,b) for a,b in zip(origin,stop))
    wt_total=int((label>0).sum());wt_in=int((label[crop]>0).sum())
    if wt_total<=0 or (wt_total-wt_in)/wt_total>1e-4:raise ValueError('frozen per-case crop-loss gate exceeded')
    view=prepare_t1ce_supervised_arrays(row['case_id'],source[crop],np.ascontiguousarray(label[crop]),spacing,output_shape=MODEL_SHAPE)
    out_image=np.ascontiguousarray(view.image.numpy()[0],dtype=np.float32);out_label=np.ascontiguousarray(view.target.numpy()[0],dtype=np.uint8)
    if not np.isfinite(out_image).all():raise ValueError('nonfinite cache output')
    if int(out_label.sum())<=0:raise ValueError('empty WT after resize')
    return out_image,out_label,{'origin':origin,'extent':[b-a for a,b in zip(origin,stop)],'native_shape':[int(v) for v in image.shape],'spacing_mm':spacing,'raw_labels':sorted(labels),'wt_voxels_native':wt_total,'wt_voxels_outside_crop':wt_total-wt_in,'wt_voxels_128':int(out_label.sum())}


def worker(task):
    slot,row=task
    try:
        image,label,geo=arrays(row);n=128**3
        assert os.pwrite(CFG['image_fd'],image.tobytes(),slot*n*4)==n*4 and os.pwrite(CFG['label_fd'],label.tobytes(),slot*n)==n
        return slot,row['case_id'],geo,None
    except BaseException as exc:return slot,row['case_id'],None,repr(exc)+'\n'+traceback.format_exc()


def main():
    verify();rows=read_rows(R/'PRETRAIN_MANIFEST.jsonl');assert len(rows)==334
    receipt=json.loads((R/'DOWNLOAD_RECEIPT.json').read_text());assert receipt['status']=='passed' and sha(R/'PRETRAIN_MANIFEST.jsonl')==receipt['manifest_sha256']
    PRETRAIN_CACHE.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(PRETRAIN_CACHE).free<334*128**3*5+(4<<30):raise RuntimeError('scratch disk below required space')
    for name,per in [('images.f32',4),('labels.u8',1)]:
        p=PRETRAIN_CACHE/name
        if p.exists():raise FileExistsError('refuse to overwrite pretraining cache')
        with p.open('xb') as f:f.truncate(len(rows)*128**3*per)
    canonical_pre=json.loads((TRAIN_CACHE/'index.json').read_text())['preprocessing']
    processes=int(os.environ.get('AFLOW_BUILD_PROCESSES',24));tick=time.monotonic();geometry={};errors=[]
    with mp.get_context('spawn').Pool(processes,initializer=init,initargs=({'images':str(PRETRAIN_CACHE/'images.f32'),'labels':str(PRETRAIN_CACHE/'labels.u8')},)) as pool:
        for k,(slot,cid,geo,err) in enumerate(pool.imap_unordered(worker,list(enumerate(rows)),chunksize=1),1):
            if err:errors.append({'case_id':cid,'error':err})
            else:geometry[cid]=geo
            if k%50==0 or k==len(rows):print(json.dumps({'utc':now(),'done':k,'of':len(rows),'errors':len(errors),'seconds':time.monotonic()-tick}),flush=True)
    if errors:raise RuntimeError(json.dumps(errors[:3],ensure_ascii=False))
    index={'schema':CACHE_SCHEMA,'schema_version':1,'manifest':str((R/'PRETRAIN_MANIFEST.jsonl').relative_to(ROOT)),'manifest_sha256':sha(R/'PRETRAIN_MANIFEST.jsonl'),'count':len(rows),'model_shape':list(MODEL_SHAPE),'image_dtype':'float32','label_dtype':'uint8',
           'preprocessing':{**canonical_pre,'skull_stripping':'not applied; ds007045 processing derivatives are MNI152 brain-extracted','source':'OpenNeuro ds007045:1.0.1 derivatives/processing *_t1ce_MNI152_brain + derivatives/segmentation *_MNI152_tumor-mask; WT = labels {1,2,3}; no label 4 in this dataset'},
           'case_ids':[r['case_id'] for r in rows],'per_case_positive_voxels':{c:g['wt_voxels_128'] for c,g in geometry.items()},'crop_geometry':{c:{k:g[k] for k in ('origin','extent','native_shape')} for c,g in geometry.items()},'extra_geometry':geometry,
           'wt_voxels_outside_crop_total':int(sum(g['wt_voxels_outside_crop'] for g in geometry.values())),'cases_with_wt_outside_crop':[c for c,g in geometry.items() if g['wt_voxels_outside_crop']>0],
           'build_seconds':time.monotonic()-tick,'split':PRETRAIN_SPLIT,'test_access':False,'role':'pretraining only; institution-disjoint public pre-operative GBM; never evaluated as an external cohort','builder':'scripts/build_segresnet_two_ideas_pretrain.py','builder_sha256':sha(Path(__file__))}
    index['images_sha256']=sha(PRETRAIN_CACHE/'images.f32');index['labels_sha256']=sha(PRETRAIN_CACHE/'labels.u8');write(PRETRAIN_CACHE/'index.json',index)
    ds=BrainCropDataset(PRETRAIN_CACHE,expected_split=PRETRAIN_SPLIT);assert len(ds)==334 and ds[0]['image'].shape==(1,128,128,128)
    write(R/'PRETRAIN_CACHE_RECEIPT.json',{'utc':now(),'status':'passed','cache':str(PRETRAIN_CACHE),'index_sha256':sha(PRETRAIN_CACHE/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':index['labels_sha256'],'count':334,'subjects':334,'sites':receipt['site_counts'],'wt_voxels_outside_crop_total':index['wt_voxels_outside_crop_total'],'cases_with_wt_outside_crop':index['cases_with_wt_outside_crop'],'extent_max':[int(max(g['extent'][a] for g in geometry.values())) for a in range(3)],'build_seconds':index['build_seconds'],'preprocessing_identical_to_canonical_except_source_note':True,'test_access':False})
    print(json.dumps({k:v for k,v in index.items() if k not in ('case_ids','per_case_positive_voxels','crop_geometry','extra_geometry')}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:write(R/'FAILURE_build_cache.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});raise
