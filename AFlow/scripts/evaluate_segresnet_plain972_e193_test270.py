"""One fixed user-authorized epoch193 test; historical rows are stats only."""
import sys,os,json,time,traceback,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.segresnet_plain972_common import ROOT,sha,write,now,model_for,seed_all,summary_ci
from scripts.sammed3d_full972_protocol import check_roles,FIT,DEV,TEST
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset
from scripts.train_t1ce_wt_capg3d import validate
R=ROOT/'archive/research_records/2026-09-17_segresnet_plain972_e193_test270'
CACHE=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'


def paired(candidate,baseline):
    left={r['case_id']:r for r in candidate};right={r['case_id']:r for r in baseline}
    assert len(left)==len(right)==270 and left.keys()==right.keys()
    assert all(left[k]['subject_id']==right[k]['subject_id'] for k in left)
    subjects=sorted({r['subject_id'] for r in candidate})
    n=np.array([sum(r['subject_id']==s for r in candidate) for s in subjects])
    sums=np.array([sum(r['locator_dsc']-right[r['case_id']]['locator_dsc'] for r in candidate if r['subject_id']==s) for s in subjects])
    ix=np.random.default_rng(20260917).integers(0,len(subjects),(10000,len(subjects)))
    delta=np.array([r['locator_dsc']-right[r['case_id']]['locator_dsc'] for r in candidate])
    return {'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),'scan_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'improved_scans':int((delta>0).sum()),'worse_scans':int((delta<0).sum()),'tied_scans':int((delta==0).sum()),'point_hits_delta':int(sum(r['automatic_point_hits_gt'] for r in candidate)-sum(r['automatic_point_hits_gt'] for r in baseline)),'scope':'paired reused internal cohort; descriptive multiple comparisons, not external confirmation'}


def main():
    if (R/'PREACCESS.json').exists():raise RuntimeError('one-shot access already attempted; no automatic repeat')
    freeze=json.loads((R/'FREEZE.json').read_text())
    for name,h in freeze['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('frozen file changed: '+name)
    check_roles();rows=[json.loads(x) for x in TEST.read_text().splitlines() if x.strip()]
    assert len(rows)==270 and len({r['case_id'] for r in rows})==270 and len({r['subject_id'] for r in rows})==119
    dataset=TestBrainCropDataset(CACHE,[r['case_id'] for r in rows]);index=dataset.index
    assert index['manifest_sha256']==sha(TEST) and index['model_shape']==[128,128,128]
    train_cache_index=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/index.json'
    assert index['preprocessing']==json.loads(train_cache_index.read_text())['preprocessing']
    write(R/'PREACCESS.json',{'utc':now(),'authorization':'user explicitly requested epoch193 best checkpoint test270','test_pixels_access':True,'selected_epoch':193,'checkpoint_sha256':sha(R/'checkpoint_epoch193.pth'),'freeze_sha256':sha(R/'FREEZE.json'),'selection_receipt_sha256':sha(R/'SELECTION_RECEIPT.json'),'cohort':{'scans':270,'subjects':119},'current_test_used_for_selection':False,'prior_test_informed_development':True,'new_outer_confirmation_access':False})
    start=time.monotonic()
    hashes={}
    for name,key,width in [('images.f32','images_sha256',4),('labels.u8','labels_sha256',1)]:
        path=CACHE/name;assert path.stat().st_size==270*128**3*width
        hashes[name]=sha(path);assert hashes[name]==index[key]
    write(R/'CACHE_RECEIPT.json',{'utc':now(),'cache':str(CACHE.relative_to(ROOT)),'index_sha256':sha(CACHE/'index.json'),'payload_sha256':hashes,'count':270,'new_cache_bytes':0,'preprocessing':index['preprocessing']})
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);seed_all(20260819)
    state=torch.load(R/'checkpoint_epoch193.pth',map_location='cpu',weights_only=False);assert state['epoch']==193
    model=model_for(device);model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval();model.requires_grad_(False)
    write(R/'MODEL_PREACCESS.json',{'utc':now(),'checkpoint_sha256':sha(R/'checkpoint_epoch193.pth'),'cache_receipt_sha256':sha(R/'CACHE_RECEIPT.json'),'batch_size':2,'precision':'bfloat16','threshold':0,'one_forward':True,'no_augmentation_or_postprocessing':True})
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic();result=validate(model,loader,device);torch.cuda.synchronize();elapsed=time.monotonic()-tick
    cases=result['per_case'];assert len(cases)==270 and result['subjects']==119 and [x['case_id'] for x in cases]==[r['case_id'] for r in rows]
    assert all(np.isfinite(x['locator_dsc']) for x in cases)
    raw={'status':'complete_test270','created_utc':now(),'epoch':193,'checkpoint_sha256':sha(R/'checkpoint_epoch193.pth'),'scans':270,'subjects':119,'summary':result,'test_access':True,'not_pristine_blind':True,'not_external':True,'prior_test_informed_development':True,'test_used_for_current_selection_or_tuning':False,'protocol':{'model':'pure SegResNet4700897','training_scans':972,'training_subjects':444,'train_epochs_completed':200,'validation_selection_epoch':193,'resolution':128,'threshold_raw_logit':0,'precision':'bfloat16','batch':2,'tta':False,'postprocessing':False,'sam':False,'uses_GT_for_localization':False,'forwards':1},'runtime':{'inference_seconds':elapsed,'total_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':torch.cuda.get_device_name()},'cache_index_sha256':sha(CACHE/'index.json'),'cache_payload_sha256':hashes,'freeze_sha256':sha(R/'FREEZE.json'),'preaccess_sha256':sha(R/'PREACCESS.json')}
    write(R/'RAW_TEST270.json',raw)
    metrics=summary_ci(cases);comparisons={}
    for item in freeze['comparators']:
        d=json.loads((ROOT/item['path']).read_text());baseline=d[item['key']]['per_case']
        comparisons[item['method']]={'baseline_scan_dice':float(np.mean([x['locator_dsc'] for x in baseline])),'baseline_subject_dice':summary_ci(baseline)['subject_dice'],**paired(cases,baseline)}
    metrics.update(status='complete_test270',created_utc=now(),epoch=193,checkpoint_sha256=raw['checkpoint_sha256'],point_hits=sum(x['automatic_point_hits_gt'] for x in cases),point_denominator=270,test_access=True,not_pristine_blind=True,not_external=True,prior_test_informed_development=True,current_test_used_for_selection_or_retuning=False,new_outer_confirmation_access=False,comparisons=comparisons,runtime=raw['runtime'],evidence={str(p.relative_to(ROOT)):sha(p) for p in [R/'RAW_TEST270.json',R/'checkpoint_epoch193.pth',R/'SELECTION_RECEIPT.json',R/'PREACCESS.json',R/'MODEL_PREACCESS.json',R/'CACHE_RECEIPT.json',R/'FREEZE.json']})
    write(R/'RESULT.json',metrics);write(R/'DONE.json',{'utc':now(),'status':'complete','result_sha256':sha(R/'RESULT.json'),'raw_sha256':sha(R/'RAW_TEST270.json')})
    print(json.dumps(metrics,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:write(R/'FAILURE.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc(),'test_access_attempted':(R/'PREACCESS.json').exists()});raise
