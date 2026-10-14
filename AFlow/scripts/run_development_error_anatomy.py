"""Fixed four-old-model descriptive readout, never loads a new candidate."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time
import traceback
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sammed3d.error_anatomy import count_errors
from sammed3d.packed_prediction_masks import PackedPredictionWriter
from sammed3d.spatial_roundtrip_audit import resize
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts import structured_background_protocol as p

REC=p.ROOT/'archive/research_records/2026-09-10_development_error_anatomy'
TEMP=Path('/tmp/aflow-error-anatomy-20260910')
CODE=['scripts/run_development_error_anatomy.py','scripts/finalize_development_error_anatomy.py','sammed3d/error_anatomy.py','tests/test_error_anatomy.py','sammed3d/packed_prediction_masks.py','sammed3d/t1ce_wt_capg3d.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/spatial_roundtrip_audit.py']


def source_models():
    native=p.RECORD/'RAW_DEVELOPMENT.json';subject=p.ROOT/'archive/research_records/2026-09-07_subject_balanced_sampling/RAW_DEVELOPMENT.json'
    p.checked(native,'80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6');p.checked(subject,'96785e393fd1033684014fda091f2de1545b42a0fce6fe226f79a8a10630017d')
    a,b=json.loads(native.read_text()),json.loads(subject.read_text());models={}
    for seed in p.SEEDS:
        for policy in ('native','subject'):
            path=p.WORK/f'seed{seed}_control/TRAIN_DONE.json' if policy=='native' else p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed{seed}_subject_balanced/TRAIN_DONE.json'
            report=json.loads(path.read_text());p.checked(report['checkpoint']['path'],report['checkpoint']['sha256'])
            models[f'{seed}_{policy}']={'checkpoint':report['checkpoint'],'report':{'path':str(path),'sha256':p.sha(path)},'baseline':a['results'][f'seed{seed}_control'] if policy=='native' else b['results'][str(seed)]['candidate']}
    return models,{'native':{'path':str(native),'sha256':p.sha(native)},'subject':{'path':str(subject),'sha256':p.sha(subject)}}


def cache():
    if (REC/'CACHE_PREACCESS.json').exists():raise FileExistsError('cache attempt already started')
    p.check_freeze();_,dev=p.check_roles();torch.set_num_threads(1)
    assert shutil.disk_usage('/tmp').free>2*1024**3
    models,sources=source_models();index=json.loads((p.CACHE/'index.json').read_text());p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    hashes={r['case_id']:p.sha(Path(r['label_path'])) for r in dev}
    aux=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'labels.u8',aux['source_labels_sha256'])
    p.write_new(REC/'CACHE_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.ROOT/n) for n in CODE},'idea_sha256':p.sha(REC/'IDEA.md'),'models':{k:{n:v[n] for n in ('checkpoint','report')} for k,v in models.items()},'sources':sources,'source_label_sha256':hashes,'dev_sha256':p.DEV_SHA,'cache_index_sha256':p.INDEX_SHA,'wt_labels_sha256':aux['source_labels_sha256'],'class_arrays_decoded_at_write':0,'source_payloads_hashed_before_write':180,'new_forwards':0,'test_access':False})
    TEMP.mkdir(exist_ok=False);output=np.memmap(TEMP/'classes.u8',mode='w+',dtype=np.uint8,shape=(180,128,128,128))
    old=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128));lookup={c:i for i,c in enumerate(index['case_ids'])};rows=[];started=time.monotonic()
    try:
        for i,row in enumerate(dev):
            p.checked(Path(row['label_path']),hashes[row['case_id']]);label=nib.as_closest_canonical(nib.load(row['label_path']));array=np.asarray(label.dataobj)
            geo=index['crop_geometry'][row['case_id']]
            assert list(array.shape)==geo['native_shape'] and nib.aff2axcodes(label.affine)==('R','A','S')
            assert np.isfinite(array).all() and np.array_equal(array,np.rint(array)) and set(np.unique(array))<={0,1,2,3,4}
            box=tuple(slice(o,o+e) for o,e in zip(geo['origin'],geo['extent']))
            classes=resize(array[box],(128,128,128),'nearest-exact').astype(np.uint8)
            assert np.array_equal(np.isin(classes,[1,2,3]),old[lookup[row['case_id']]]>0)
            output[i]=classes;rows.append({'case_id':row['case_id'],'class_counts':np.bincount(classes.ravel(),minlength=5).tolist(),'class_sha256':p.sha(Path(row['label_path']))})
        output.flush();del output
        p.write_new(REC/'CLASS_CACHE.json',{'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'path':str(TEMP/'classes.u8'),'sha256':p.sha(TEMP/'classes.u8'),'shape':[180,128,128,128],'case_ids':[r['case_id'] for r in dev],'rows':rows,'wt_byte_parity_cases':180,'seconds':time.monotonic()-started,'new_forwards':0,'test_access':False})
    except BaseException as error:
        p.write_new(REC/'CACHE_FAILURE.json',{'error':repr(error),'decoded_completed':len(rows),'test_access':False});raise


def check_cache():
    pre=json.loads((REC/'CACHE_PREACCESS.json').read_text())
    for n,d in pre['files'].items():p.checked(p.ROOT/n,d)
    p.checked(REC/'IDEA.md',pre['idea_sha256'])
    c=json.loads((REC/'CLASS_CACHE.json').read_text());assert c['status']=='passed'
    p.checked(Path(c['path']),c['sha256']);return pre,c


@torch.inference_mode()
def smoke():
    if (REC/'SMOKE_RECEIPT.json').exists():raise FileExistsError('smoke already done')
    pre,c=check_cache();fit,_=p.check_roles();torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device);assert free>10*1024**3,'training keeps memory priority'
    data=BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in fit[:8]])
    x=torch.stack([data[i]['image'] for i in range(8)]).to(device)
    info=pre['models']['2026090711_native']['checkpoint'];p.checked(info['path'],info['sha256'])
    model=AutomaticPromptLocator(center_aware=False).to(device).eval();model.load_state_dict(torch.load(info['path'],map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logit=model(x).segmentation_logits.float()
    assert torch.isfinite(logit).all();peak=torch.cuda.max_memory_allocated(device);assert peak<8*1024**3
    p.write_new(REC/'SMOKE_RECEIPT.json',{'status':'passed','fit_cases':[r['case_id'] for r in fit[:8]],'forwards':8,'peak_bytes':peak,'free_bytes_before':free,'development_forwards':0,'test_access':False,'code_sha256':p.sha(__file__)})


@torch.inference_mode()
def evaluate():
    if (REC/'DEV_PREACCESS.json').exists():raise FileExistsError('readout already started')
    pre,c=check_cache();assert json.loads((REC/'SMOKE_RECEIPT.json').read_text())['status']=='passed'
    _,dev=p.check_roles();models,sources=source_models();torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    free,_=torch.cuda.mem_get_info(device);assert free>10*1024**3
    assert c['case_ids']==[r['case_id'] for r in dev]
    aux=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'images.f32',aux['source_images_sha256'])
    p.write_new(REC/'DEV_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'cache_preaccess_sha256':p.sha(REC/'CACHE_PREACCESS.json'),'class_cache_receipt_sha256':p.sha(REC/'CLASS_CACHE.json'),'smoke_sha256':p.sha(REC/'SMOKE_RECEIPT.json'),'models':pre['models'],'sources':sources,'images_sha256':aux['source_images_sha256'],'expected_forwards':720,'new_forwards':0,'development_role':'reused already revealed old models only','test_access':False})
    labels=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=(180,128,128,128))
    data=BrainCropDataset(p.CACHE,expected_split='train',case_ids=c['case_ids']);loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=0)
    writer=PackedPredictionWriter(TEMP/'masks',list(models),c['case_ids']);results={};forwards=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    current=[]
    try:
        for key,info in models.items():
            model=AutomaticPromptLocator(center_aware=False).to(device).eval();model.load_state_dict(torch.load(info['checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
            current=[]
            for batch in loader:
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(batch['image'].to(device)).segmentation_logits.float()
                assert torch.isfinite(logits).all();forwards+=len(logits)
                pred=(logits[:,0]>0).cpu().numpy()
                for j,case in enumerate(batch['case_id']):
                    slot=len(current);point=int(logits[j,0].flatten().argmax());count=count_errors(pred[j],labels[slot],point);old=info['baseline']['per_case'][slot]
                    assert case==old['case_id'] and point==old['point_flat_index']
                    assert all(count[k]==old[k] for k in ['intersection','predicted_voxels','target_voxels','point_hit'])
                    assert 2*count['intersection']/(count['predicted_voxels']+count['target_voxels'])==old['dice']
                    current.append({'case_id':case,'subject_id':batch['subject_id'][j],'point_flat_index':point,'dice':old['dice'],'counts':count})
                writer.append(key,pred,batch['case_id'])
                assert torch.cuda.max_memory_allocated(device)<8*1024**3
            results[key]=current;del model,logits;torch.cuda.empty_cache()
            print(key+' exact old prediction parity',flush=True)
        assert forwards==720;p.write_new(REC/'MASK_INDEX.json',writer.finalize())
        p.write_new(REC/'RAW_DEVELOPMENT.json',{'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'new_forwards':forwards,'seconds':time.monotonic()-started,'peak_bytes':torch.cuda.max_memory_allocated(device),'dev_preaccess_sha256':p.sha(REC/'DEV_PREACCESS.json'),'mask_index_sha256':p.sha(REC/'MASK_INDEX.json'),'test_access':False,'new_candidate_checkpoint_loaded':False})
    except BaseException as error:
        p.write_new(REC/'DEVELOPMENT_FAILURE.json',{'error':repr(error),'traceback':traceback.format_exc(),'new_forwards_lower_bound':forwards,'completed_models':results,'partial_rows':current,'test_access':False,'no_baseline_replacement_allowed':True});raise
    finally:writer.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['cache','smoke','evaluate'],required=True);args=ap.parse_args();globals()[args.stage]()
