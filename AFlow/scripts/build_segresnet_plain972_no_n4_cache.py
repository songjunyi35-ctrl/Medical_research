"""Build full original train/validation no-extra-N4 caches; never open test pixels."""
import sys,json,time,os,multiprocessing as mp,shutil,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import nibabel as nib
import torch
from scripts.segresnet_plain972_no_n4_common import ROOT,R,TRAIN_CACHE,VAL_CACHE,sha,write,now,verify
from scripts.sammed3d_full972_protocol import check_roles,FIT,DEV,TEST
from scripts.build_no_n4_clean_cache import arrays
from sammed3d.t1ce_wt_training import validate_row,_load_canonical_nifti
CONTROL_TRAIN=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train'
CONTROL_VAL=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_ema_20260819/validation108'
CONFIG={}

def init(path):
    torch.set_num_threads(1)
    CONFIG['images']=os.open(str(Path(path)/'images.f32'),os.O_WRONLY)
    CONFIG['labels']=os.open(str(Path(path)/'labels.u8'),os.O_WRONLY)

def prepare(row):
    validate_row(row);assert row['split'] in ['train','validation']
    image=_load_canonical_nifti(Path(row['image_path']),'T1ce')
    label=_load_canonical_nifti(Path(row['label_path']),'WT label')
    assert image.shape==label.shape and np.allclose(image.affine,label.affine,atol=1e-5,rtol=0)
    return arrays(row['case_id'],np.asanyarray(image.dataobj),np.asanyarray(label.dataobj),nib.affines.voxel_sizes(image.affine),False)

def worker(task):
    i,row=task;x,y,geo=prepare(row);assert np.isfinite(x).all() and int(y.sum())>0
    assert os.pwrite(CONFIG['images'],x.tobytes(),i*128**3*4)==128**3*4
    assert os.pwrite(CONFIG['labels'],y.tobytes(),i*128**3)==128**3
    return i,row['case_id'],geo,int(y.sum())

def build(rows,manifest,target,control,split):
    if target.exists():raise RuntimeError('refuse existing new cache; no automatic rebuild')
    target.mkdir(parents=True)
    for name,n in [('images.f32',4),('labels.u8',1)]:
        with (target/name).open('xb') as f:f.truncate(len(rows)*128**3*n)
    ref=json.loads((control/'index.json').read_text())
    assert ref['case_ids']==[r['case_id'] for r in rows]
    tick=time.monotonic();results=[]
    with mp.get_context('spawn').Pool(8,initializer=init,initargs=(str(target),)) as pool:
        for result in pool.imap_unordered(worker,enumerate(rows),chunksize=1):
            results.append(result)
            if len(results)%50==0 or len(results)==len(rows):
                print(json.dumps({'utc':now(),'role':split,'built':len(results),'total':len(rows)}),flush=True)
                write(R/'CACHE_PROGRESS.json',{'utc':now(),'status':'building','role':split,'built':len(results),'total':len(rows),'test_access':False})
    results.sort();geo={cid:g for _,cid,g,_ in results};assert geo==ref['crop_geometry']
    labelsha=sha(target/'labels.u8');assert labelsha==ref['labels_sha256']==sha(control/'labels.u8')
    index={**ref,'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':sha(manifest),'preprocessing':{**ref['preprocessing'],'n4_bias_field_correction':False,'n4_shrink_factor':None,'n4_implementation':None,'nonpositive_input_values':'zero, same support rule as historical N4'},'images_sha256':sha(target/'images.f32'),'labels_sha256':labelsha,'builder':'scripts/build_segresnet_plain972_no_n4_cache.py','builder_sha256':sha(Path(__file__)),'split':split,'test_access':False,'build_seconds':time.monotonic()-tick,'crop_geometry':geo,'per_case_positive_voxels':{cid:n for _,cid,_,n in results}}
    write(target/'index.json',index);write(R/f'CACHE_INDEX_{split.upper()}.json',index)
    x=np.memmap(target/'images.f32',mode='r',dtype=np.float32,shape=(len(rows),128,128,128));y=np.memmap(target/'labels.u8',mode='r',dtype=np.uint8,shape=(len(rows),128,128,128))
    torch.set_num_threads(1);checks=[]
    for slot in [0,len(rows)//2,len(rows)-1]:
        fresh,lab,g=prepare(rows[slot]);checks.append({'case_id':rows[slot]['case_id'],'image_exact':bool(np.array_equal(fresh,x[slot])),'label_exact':bool(np.array_equal(lab,y[slot])),'geometry_exact':g==geo[rows[slot]['case_id']]})
    assert all(c['image_exact'] and c['label_exact'] and c['geometry_exact'] for c in checks)
    return {'role':split,'path':str(target),'count':len(rows),'index_sha256':sha(target/'index.json'),'images_sha256':index['images_sha256'],'labels_sha256':labelsha,'all_labels_and_geometry_match_N4':True,'independent_rebuilds':checks,'build_seconds':index['build_seconds']}

def main():
    verify();fit,dev=check_roles();assert sha(TEST)=='a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9'
    if (R/'CACHE_PREACCESS.json').exists():raise RuntimeError('cache already attempted')
    required=1080*128**3*5
    assert shutil.disk_usage('/tmp').free>required+10*2**30
    write(R/'CACHE_PREACCESS.json',{'utc':now(),'train_pixels':True,'validation_pixels':True,'test_pixels':False,'manifest_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [FIT,DEV,TEST]},'freeze_sha256':sha(R/'FREEZE.json'),'expected_cache_bytes':required,'tmp_free_bytes':shutil.disk_usage('/tmp').free,'prior_test_informed_development':True})
    caches=[build(fit,FIT,TRAIN_CACHE,CONTROL_TRAIN,'train'),build(dev,DEV,VAL_CACHE,CONTROL_VAL,'validation')]
    write(R/'DATA_AUDIT.json',{'utc':now(),'status':'passed','train_case_ids':[r['case_id'] for r in fit],'validation_case_ids':[r['case_id'] for r in dev],'train_scans':972,'train_subjects':444,'validation_scans':108,'validation_subjects':50,'test_metadata_only':{'scans':270,'subjects':119,'sha256':sha(TEST)},'patient_isolation':True,'caches':caches,'all_labels_and_geometry_match_N4':True,'new_test_pixels_access':False})
    write(R/'CACHE_PROGRESS.json',{'utc':now(),'status':'complete','train':972,'validation':108,'test_access':False});print('CACHE PASSED',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:write(R/'CACHE_FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'test_access':False});raise
