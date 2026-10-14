"""Fit-only soft-target cache; original images/labels/caches are never modified."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sammed3d.fractional_occupancy import fractional_occupancy
from scripts import fractional_occupancy_protocol as p
from scripts.train_t1ce_wt_capg3d import write_json_atomic


def main():
    pre=p.RECORD/'CACHE_PREACCESS.json'
    if pre.exists() or p.SOFT_CACHE.exists():raise FileExistsError('cache attempt already exists; preserve prior evidence')
    p.source.check_freeze();fit,_=p.check_roles();p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    aux_path=p.source.RECORD/'AUX_CACHE_RECEIPT.json';aux=json.loads(aux_path.read_text())
    assert aux['all_wt_byte_exact'] and len(aux['details'])==684
    p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    expected_labels={r['case_id']:r['source_label_sha256'] for r in aux['details']}
    assert set(expected_labels)=={r['case_id'] for r in fit}
    needed=684*128**3*2
    if shutil.disk_usage(p.SOFT_CACHE.parent).free < needed+1024**3:raise RuntimeError('soft cache disk reserve')
    files=['archive/research_records/2026-09-10_fractional_occupancy/IDEA.md','archive/research_records/2026-09-10_fractional_occupancy/PREREG.md','sammed3d/fractional_occupancy.py','tests/test_fractional_occupancy.py','scripts/build_fractional_occupancy_cache.py','scripts/fractional_occupancy_protocol.py']
    p.write_new(pre,{'status':'frozen_before_raw_fit_label_access_and_cache_build','created_utc':datetime.now(timezone.utc).isoformat(),'files':{name:p.sha(p.ROOT/name) for name in files},'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'source_index_sha256':p.INDEX_SHA,'source_hard_labels_sha256':aux['source_labels_sha256'],'source_auxiliary_receipt_sha256':p.sha(aux_path),'source_label_sha256':expected_labels,'output':str(p.SOFT_CACHE/'targets.f16'),'bytes':needed,'dtype':'float16','compute_dtype':'float64','old_grid_hard_label_verification':'nearest-exact byte parity for every fit case','development_pixels_accessed':False,'model_forwards':0})
    p.SOFT_CACHE.mkdir(parents=True,exist_ok=False)
    target_path=p.SOFT_CACHE/'targets.f16'
    destination=os.open(target_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.ftruncate(destination,needed)
    source_fd=os.open(p.CACHE/'labels.u8',os.O_RDONLY)
    index=json.loads((p.CACHE/'index.json').read_text());slots={c:i for i,c in enumerate(index['case_ids'])}
    details=[];start=time.monotonic();torch.set_num_threads(2)
    current=None
    try:
        with (p.RECORD/'CACHE_ACCESS.jsonl').open('x') as access:
            for ordinal,row in enumerate(fit):
                current=row['case_id'];label_path=Path(row['label_path'])
                access.write(json.dumps({'case_id':current,'ordinal':ordinal,'event':'source_label_attempt'})+'\n');access.flush()
                p.checked(label_path,expected_labels[current])
                image=nib.as_closest_canonical(nib.load(label_path))
                raw=np.asanyarray(image.dataobj)
                if not np.isin(raw,[0,1,2,3,4]).all():raise ValueError('native label domain')
                geometry=index['crop_geometry'][current]
                assert list(raw.shape)==geometry['native_shape']
                wt=np.isin(raw,[1,2,3]);full_mass=int(wt.sum())
                box=tuple(slice(o,o+n) for o,n in zip(geometry['origin'],geometry['extent'],strict=True))
                crop=np.ascontiguousarray(wt[box],dtype=np.uint8);mass=int(crop.sum());assert mass>0
                original_bytes=os.pread(source_fd,128**3,slots[current]*128**3);assert len(original_bytes)==128**3
                original=np.frombuffer(original_bytes,dtype=np.uint8).reshape(128,128,128)
                hard=F.interpolate(torch.from_numpy(crop.astype(np.float32))[None,None],size=(128,128,128),mode='nearest-exact')[0,0].byte().numpy()
                if not np.array_equal(hard,original):raise RuntimeError(f'{current}: source hard WT no longer matches cache')
                q=fractional_occupancy(crop);stored=np.ascontiguousarray(q,dtype=np.float16)
                expected=mass*128**3/np.prod(crop.shape)
                error64=abs(float(q.sum(dtype=np.float64))-expected)/expected
                error16=abs(float(stored.sum(dtype=np.float64))-expected)/expected
                assert np.isfinite(stored).all() and stored.min()>=0 and stored.max()<=1 and stored.sum(dtype=np.float64)>0
                assert error64<=1e-10 and error16<=.001
                body=stored.tobytes();written=os.pwrite(destination,body,ordinal*128**3*2);assert written==len(body)
                hardened=stored>.5;hard_n=int(original.sum());new_hard_n=int(hardened.sum())
                details.append({'case_id':current,'subject_id':row['subject_id'],'source_label_sha256':expected_labels[current],'hard_cache_byte_exact':True,'native_wt_voxels':full_mass,'crop_native_wt_voxels':mass,'outside_common_crop_wt_voxels':full_mass-mass,'crop_shape':list(crop.shape),'hard_cache_voxels':hard_n,'expected_occupancy_sum':float(expected),'float64_sum':float(q.sum(dtype=np.float64)),'stored_float16_sum':float(stored.sum(dtype=np.float64)),'relative_volume_error_float64':float(error64),'relative_volume_error_float16':float(error16),'fractional_voxels':int(((stored>0)&(stored<1)).sum()),'soft_positive_voxels':int((stored>0).sum()),'soft_max':float(stored.max()),'hardened_half_voxels':new_hard_n,'hardened_half_vs_old_hard_dice_not_model':2*int((hardened&(original>0)).sum())/(new_hard_n+hard_n),'target_array_sha256':hashlib.sha256(body).hexdigest()})
                access.write(json.dumps({'case_id':current,'ordinal':ordinal,'event':'target_written_verified'})+'\n');access.flush()
                if len(details)%32==0:
                    write_json_atomic(p.RECORD/'CACHE_PROGRESS.json',{'completed':len(details),'scans':684,'seconds':time.monotonic()-start,'new_model_forwards':0});print(f'{len(details)}/684',flush=True)
        os.fsync(destination)
    except BaseException as error:
        p.write_new(p.RECORD/'CACHE_FAILURE.json',{'status':'failed','created_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error),'attempted_case':current,'completed_rows':details,'partial_cache_not_admitted':str(target_path),'development_access':False,'new_model_forwards':0});raise
    finally:os.close(destination);os.close(source_fd)
    assert len(details)==684 and target_path.stat().st_size==needed
    cached_index={'schema':'aflow.fractional_occupancy.fit_only_cache','created_utc':datetime.now(timezone.utc).isoformat(),'case_ids':[r['case_id'] for r in fit],'scans':684,'subjects':313,'shape':[128,128,128],'dtype':'float16','targets_sha256':p.sha(target_path),'targets_bytes':needed,'fit_sha256':p.FIT_SHA,'cache_preaccess_sha256':p.sha(pre),'source_index_sha256':p.INDEX_SHA,'all_hard_cache_byte_exact':True,'compute':'exact_separable_cell_overlap_float64','postprocessing_binarization':False,'model_forwards':0,'development_access':False}
    p.write_new(p.SOFT_CACHE/'index.json',cached_index);p.write_new(p.RECORD/'CACHE_INDEX.json',cached_index)
    receipt={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'target_path':str(target_path),'target_sha256':p.sha(target_path),'index_sha256':p.sha(p.SOFT_CACHE/'index.json'),'bytes':needed,'scans':684,'subjects':313,'fit_sha256':p.FIT_SHA,'cache_preaccess_sha256':p.sha(pre),'source_auxiliary_receipt_sha256':p.sha(aux_path),'source_images_sha256':aux['source_images_sha256'],'source_hard_labels_sha256':aux['source_labels_sha256'],'all_hard_cache_byte_exact':True,'max_relative_volume_error_float64':max(r['relative_volume_error_float64'] for r in details),'max_relative_volume_error_float16':max(r['relative_volume_error_float16'] for r in details),'hardened_half_empty_cases':[r['case_id'] for r in details if r['hardened_half_voxels']==0],'details':details,'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'access':{'fit_raw_labels':684,'fit_image_pixels':False,'development_pixels':False,'test':False,'confirmation':False,'outer':False,'external':False},'model_forwards':0,'training_admitted':False}
    p.write_new(p.RECORD/'CACHE_RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='details'},indent=2),flush=True)


if __name__=='__main__':main()
