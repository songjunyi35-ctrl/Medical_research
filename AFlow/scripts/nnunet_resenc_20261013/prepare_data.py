"""Train-only nnU-Net planning, then fixed-split train+validation preprocessing."""
import concurrent.futures
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.nnunet_resenc_20261013.common import *

def native_build():
    from scripts import build_segresnet_pretrained_native_cache as builder
    builder.CACHE_ROOT=NATIVE
    references={d['role']:d for d in read(ROOT/'archive/research_records/2026-09-24_segresnet_native_multiscale_supervision/DATA_AUDIT.json')['caches']}
    done=[]
    for split in ['train','validation']:
        write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'building_native_cache','split':split,'training_started':False,'test_pixels_access':False})
        d=builder.build(split,processes=24)
        if any(d[k]!=references[split][k] for k in ['images_sha256','labels_sha256']):
            raise RuntimeError('native payload mismatch: '+split)
        done.append(d)
    write(R/'NATIVE_DATA_AUDIT.json',{'at':now(),'status':'passed','caches':done,'historical_payload_exact':True,'test_pixels_access':False})

def export_case(task):
    import numpy as np
    import nibabel as nib
    split,slot,cid,geo,count=task
    canvas=(160,208,160)
    image=np.memmap(NATIVE/split/'images.f32',dtype=np.float32,mode='r',shape=(count,*canvas))
    label=np.memmap(NATIVE/split/'labels.u8',dtype=np.uint8,mode='r',shape=(count,*canvas))
    sl=tuple(slice(o,o+e) for o,e in zip(geo['canvas_offset'],geo['extent']))
    x=np.array(image[(slot,*sl)]);y=np.array(label[(slot,*sl)])
    affine=np.diag([*geo['spacing_mm'],1.])
    ip=RAW/'imagesTr'/f'{cid}_0000.nii.gz';lp=RAW/'labelsTr'/f'{cid}.nii.gz'
    nib.save(nib.Nifti1Image(x,affine),ip);nib.save(nib.Nifti1Image(y,affine),lp)
    if not np.array_equal(np.asarray(nib.load(ip).dataobj),x) or not np.array_equal(np.asarray(nib.load(lp).dataobj),y):
        raise RuntimeError('export pixel mismatch '+cid)
    return {'case_id':cid,'split':split,'image_sha256':sha(ip),'label_sha256':sha(lp),'shape':list(x.shape)}

def export(split):
    for d in ['imagesTr','labelsTr']:(RAW/d).mkdir(parents=True,exist_ok=True)
    idx=read(NATIVE/split/'index.json')
    tasks=[(split,i,c,idx['geometry'][c],idx['count']) for i,c in enumerate(idx['case_ids'])]
    with concurrent.futures.ProcessPoolExecutor(12,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(export_case,tasks))
    write(R/f'EXPORT_{split}.json',{'at':now(),'roundtrip_exact':True,'cases':results})

def main():
    os.environ.update(env())
    verify_source_freeze('DATA_FREEZE.json');disk();summary=verify_manifests()
    if (R/'DATA_PREACCESS.json').exists():raise RuntimeError('data attempt already exists; inspect before recovery')
    if shutil.disk_usage('/tmp').free<150*2**30:raise RuntimeError('scratch below150GiB')
    write(R/'DATA_PREACCESS.json',{'at':now(),'roles':['train','validation'],'test_metadata_only':True,'test_pixels_access':False,'manifest':summary})
    native_build();export('train')
    dataset={'channel_names':{'0':'T1ce'},'labels':{'background':0,'WT':1},'numTraining':972,
             'file_ending':'.nii.gz','overwrite_image_reader_writer':'NibabelIO'}
    write(RAW/'dataset.json',dataset)
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset,plan_experiment_dataset,preprocess_dataset
    from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncL
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'train972_fingerprint_and_plan','training_started':False,'test_pixels_access':False})
    extract_fingerprint_dataset(813,num_processes=12,check_dataset_integrity=True,verbose=False)
    plan,identifier=plan_experiment_dataset(813,experiment_planner_class=nnUNetPlannerResEncL,overwrite_target_spacing=(1,1,1))
    if identifier!=PLANS:raise RuntimeError('unexpected plans identifier '+identifier)
    shutil.copy2(PRE/'dataset_fingerprint.json',R/'TRAIN972_FINGERPRINT.json')
    write(R/'TRAIN972_DATASET.json',dataset)
    shutil.copy2(PRE/(PLANS+'.json'),R/'TRAIN972_PLANS.json')
    # Adding validation here only makes files available to the frozen preprocessor;
    # its IDs are excluded from fitting by the single explicit split and trainer assertions.
    export('validation');dataset['numTraining']=1080;write(RAW/'dataset.json',dataset)
    write(PRE/'dataset.json',dataset)
    split={'train':[x['case_id'] for x in rows('train')],'val':[x['case_id'] for x in rows('validation')]}
    write(PRE/'splits_final.json',[split]);write(R/'SPLITS.json',[split])
    write(R/'PIPELINE_STATUS.json',{'at':now(),'state':'fixed_plan_preprocessing1080','training_started':False,'test_pixels_access':False})
    preprocess_dataset(813,PLANS,configurations=['3d_fullres'],num_processes=[12],verbose=False)
    assert sha(PRE/(PLANS+'.json'))==sha(R/'TRAIN972_PLANS.json')
    # Label restoration checks all1080; preprocessing may crop but never resample off1mm.
    import numpy as np
    import nibabel as nib
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    folder=PRE/plan['configurations']['3d_fullres']['data_identifier']
    ds=infer_dataset_class(str(folder))(str(folder))
    assert set(ds.identifiers)==set(split['train']+split['val'])
    for cid in ds.identifiers:
        data,seg,_,props=ds.load_case(cid);seg=np.asarray(seg[:])[0]
        rebuilt=np.zeros(props['shape_before_cropping'],dtype=np.uint8)
        box=tuple(slice(a,b) for a,b in props['bbox_used_for_cropping'])
        rebuilt[box]=(seg>0).astype(np.uint8)
        rebuilt=rebuilt.transpose(plan['transpose_backward']).transpose(2,1,0)
        original=np.asarray(nib.load(RAW/'labelsTr'/f'{cid}.nii.gz').dataobj)
        if not np.array_equal(rebuilt,original):raise RuntimeError('label geometry roundtrip mismatch '+cid)
    write(R/'DATA_READY.json',{'at':now(),'status':'passed','patient_isolation':True,'planning_train_scans':972,
       'validation_not_used_for_planning':True,'preprocessed_scans':1080,'all_label_native_roundtrips_exact':True,
       'plan_sha256':sha(PRE/(PLANS+'.json')),'split_sha256':sha(PRE/'splits_final.json'),'test_pixels_access':False})

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(R/'DATA_FAILURE.json',{'at':now(),'error':repr(e),'training_started':False,'test_pixels_access':False});raise
