"""User-authorized one-shot epoch50 evaluation, isolated from the ongoing trainer."""
import os,sys,json,time,hashlib,traceback,concurrent.futures,multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.sammed3d_graymean_single972 import ROOT,sha,write,now,seed_all
from scripts.build_t1ce_wt_official15_test_cache import read_and_audit_rows,TEST_MANIFEST
from scripts.build_t1ce_wt_official15_cache import preprocess_row
from scripts import train_sammed3d_t1ce_wt_upstream_literal_va as upstream
from scripts.evaluate_t1ce_wt_upstream_literal_test import summarize
from sammed3d import engine
R=ROOT/'archive/research_records/2026-09-17_sammed3d_graymean_e050_test270'
CACHE=Path('/tmp/aflow_graymean_e050_test270_20260917')
SHAPE=(270,128,128,128)


def verify():
    freeze=json.loads((R/'FREEZE.json').read_text())
    for p,h in freeze['files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('frozen input changed: '+p)
    _,rows=read_and_audit_rows(TEST_MANIFEST)
    if len({r['subject_id'] for r in rows})!=119:raise RuntimeError('subject count')
    return rows


def load():
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    seed_all(2023)
    saved=torch.load(R/'checkpoint_epoch050.pth',map_location='cpu',weights_only=False)
    assert saved['epoch']==50
    model=engine.sam_model_registry3D[engine.MODEL_TYPE](checkpoint=None)
    model.load_state_dict(saved['model_state_dict'],strict=True);del saved
    model.requires_grad_(False);model.eval();return model.cuda()


def smoke():
    verify();model=load()
    x=torch.randn(1,1,128,128,128,device='cuda');y=torch.zeros_like(x);y[:,:,40:88,40:88,40:88]=1
    with torch.no_grad(),upstream._autocast(torch.device('cuda:0')):
        curve=upstream.upstream_validation_curve(model=model,image=x,target=y)
    assert len(curve)==5 and np.isfinite(curve).all()
    write(R/'GPU_SMOKE.json',{'utc':now(),'status':'passed','gpu':torch.cuda.get_device_name(),'memory_fraction':.20,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'synthetic_curve':curve,'test_pixels_access':False})
    print('GPU synthetic smoke passed',flush=True)


def prepare_one(arg):
    index,row=arg
    torch.set_num_threads(1)
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    sources={k:sha(row[k]) for k in ['image_path','label_path']}
    image,label,qc=preprocess_row(row)
    xb=np.ascontiguousarray(image[0],dtype=np.float32).tobytes();yb=np.ascontiguousarray(label[0],dtype=np.uint8).tobytes()
    for name,data,width in [('images.f32',xb,4),('labels.u8',yb,1)]:
        fd=os.open(CACHE/name,os.O_WRONLY)
        try:os.pwrite(fd,data,index*128**3*width)
        finally:os.close(fd)
    return {'case_id':row['case_id'],'subject_id':row['subject_id'],'manifest_index':index,'source_sha256':sources,'image_tensor_sha256':hashlib.sha256(xb).hexdigest(),'label_tensor_sha256':hashlib.sha256(yb).hexdigest(),'qc':qc}


def build(rows,baseline):
    import shutil
    if CACHE.exists():raise RuntimeError('refusing existing test cache; audit before retry')
    if shutil.disk_usage('/tmp').free<6*(1<<30):raise RuntimeError('cache disk gate')
    CACHE.mkdir()
    for name,dtype in [('images.f32',np.float32),('labels.u8',np.uint8)]:np.memmap(CACHE/name,dtype=dtype,mode='w+',shape=SHAPE).flush()
    records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(prepare_one,enumerate(rows),chunksize=1):
            records.append(row)
            if len(records)%20==0:print(json.dumps({'utc':now(),'stage':'preprocess','complete':len(records),'total':270}),flush=True)
    hashes={name:sha(CACHE/name) for name in ['images.f32','labels.u8']}
    parity=hashes['images.f32']==baseline['data']['cache_images_sha256'] and hashes['labels.u8']==baseline['data']['cache_labels_sha256']
    receipt={'utc':now(),'count':len(records),'subjects':119,'path':str(CACHE),'hashes':hashes,'cases':records,'baseline_tensor_byte_parity':parity,'bytes':sum((CACHE/n).stat().st_size for n in hashes),'test_access':True,'reused_internal':True}
    write(R/'CACHE_INDEX.json',receipt)
    if not parity:raise RuntimeError('rebuilt tensors differ from baseline cache: stop before inference')
    return receipt


def evaluate():
    rows=verify()
    if (R/'PREACCESS.json').exists():raise RuntimeError('one-shot test access already attempted; do not rerun automatically')
    assert json.loads((R/'GPU_SMOKE.json').read_text())['status']=='passed'
    selection=json.loads((R/'SELECTION_RECEIPT.json').read_text());baseline=json.loads((ROOT/selection['baseline_report']).read_text())
    write(R/'PREACCESS.json',{'utc':now(),'authorization':'user explicitly requested epoch50 test270 2026-09-17','test_pixels_access':True,'test_model_evaluation_planned':True,'freeze_sha256':sha(R/'FREEZE.json'),'selection_receipt_sha256':sha(R/'SELECTION_RECEIPT.json'),'checkpoint_sha256':sha(R/'checkpoint_epoch050.pth'),'training_continues_unchanged':True,'reused_internal_test':True,'not_pristine_blind':True})
    started=time.monotonic();cache=build(rows,baseline);prep_seconds=time.monotonic()-started
    model=load();torch.cuda.reset_peak_memory_stats();infer_start=time.monotonic()
    images=np.memmap(CACHE/'images.f32',dtype=np.float32,mode='r',shape=SHAPE);labels=np.memmap(CACHE/'labels.u8',dtype=np.uint8,mode='r',shape=SHAPE)
    records=[]
    write(R/'MODEL_PREACCESS.json',{'utc':now(),'checkpoint_sha256':sha(R/'checkpoint_epoch050.pth'),'cache_index_sha256':sha(R/'CACHE_INDEX.json'),'test_access':True})
    with torch.no_grad():
        for i,row in enumerate(rows):
            np.random.seed(upstream.stable_case_seed(row['case_id']))
            x=torch.from_numpy(np.array(images[i]))[None,None].cuda();y=torch.from_numpy(np.array(labels[i]))[None,None].to(device='cuda',dtype=torch.float32)
            tick=time.monotonic()
            with upstream._autocast(torch.device('cuda:0')):curve=upstream.upstream_validation_curve(model=model,image=x,target=y)
            assert len(curve)==5 and np.isfinite(curve).all()
            record={'manifest_index':i,'case_id':row['case_id'],'subject_id':row['subject_id'],'target_voxels':int(labels[i].sum()),'click_dice':[float(v) for v in curve],'best_of_5_dice':float(max(curve)),'best_click':int(np.argmax(curve)+1),'final_click_5_dice':float(curve[-1]),'inference_seconds':time.monotonic()-tick}
            records.append(record)
            with (R/'CASES.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if (i+1)%20==0 or i==269:print(json.dumps({'utc':now(),'stage':'inference','complete':i+1,'total':270}),flush=True)
    assert [r['case_id'] for r in records]==[r['case_id'] for r in baseline['cases']]
    assert all(a['target_voxels']==b['target_voxels'] for a,b in zip(records,baseline['cases']))
    report={'status':'complete_270_case_test','created_utc':now(),'checkpoint_epoch':50,'checkpoint_sha256':sha(R/'checkpoint_epoch050.pth'),'selection':selection,'test_access':True,'reused_internal':True,'test_used_for_selection_or_adaptation':False,'scans':270,'subjects':119,'primary_metric':'mean per-case max Dice over clicks1..5','precision':'FP16 autocast','threshold_raw_logit':0,'click_threshold_raw_logit':.5,'test_time_augmentation':False,'postprocessing':False,'gt_centered_and_gt_clicks':True,'grid':'1.5mm GT-centered128^3','baseline_tensor_byte_parity':cache['baseline_tensor_byte_parity'],'cache_index_sha256':sha(R/'CACHE_INDEX.json'),'preaccess_sha256':sha(R/'PREACCESS.json'),'model_preaccess_sha256':sha(R/'MODEL_PREACCESS.json'),'freeze_sha256':sha(R/'FREEZE.json'),'runtime':{'preprocess_seconds':prep_seconds,'inference_seconds':time.monotonic()-infer_start,'total_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu_count':1,'physical_gpu':1,'training_concurrent':True},'summary':summarize(records),'cases':records}
    write(R/'RAW_TEST270.json',report)
    write(R/'DONE.json',{'utc':now(),'status':'complete','raw_report_sha256':sha(R/'RAW_TEST270.json'),'checkpoint_sha256':sha(R/'checkpoint_epoch050.pth')})
    print(json.dumps(report['summary']['primary_best_of_5']),flush=True)

if __name__=='__main__':
    try:
        if sys.argv[1]=='smoke':smoke()
        elif sys.argv[1]=='test':evaluate()
        else:raise ValueError('mode must be smoke/test')
    except Exception as exc:write(R/'FAILURE.json',{'utc':now(),'mode':sys.argv[1],'error':repr(exc),'traceback':traceback.format_exc(),'test_access':(R/'PREACCESS.json').exists()});raise
