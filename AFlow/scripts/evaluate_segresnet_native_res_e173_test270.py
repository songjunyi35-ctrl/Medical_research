"""One fixed user-authorized test270 evaluation of the native-resolution SegResNet epoch173.

Also scores the frozen e193 baseline on the SAME native test grid (128^3 logits trilinear-upsampled, primary;
nearest mask, secondary) and re-derives its 128^3-grid Dice to prove the checkpoint/cache reproduce 0.759179.
Paired comparisons use the same patient-cluster bootstrap as the validation study (seed 20260918).
"""
import sys,os,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from scripts.segresnet_native_res_test_common import *
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset


def main():
    if (R/'MODEL_PREACCESS.json').exists():raise RuntimeError('one-shot evaluation already attempted; no automatic repeat')
    verify()
    # Deterministic numerics, matching the historical e193 test evaluation (cudnn deterministic, no autotuning, PyTorch default TF32 flags).
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    if sha(BASELINE_CKPT)!=BASELINE_CKPT_SHA:raise RuntimeError('baseline checkpoint hash changed')
    receipt=json.loads((R/'CACHE_RECEIPT.json').read_text());assert receipt['count']==TEST_SCANS
    rows=read_rows(MANIFEST['test']);ids=[r['case_id'] for r in rows]
    native=NativeTestDataset(TEST_CACHE,ids);assert sha(TEST_CACHE/'index.json')==receipt['index_sha256']
    grid=TestBrainCropDataset(CANONICAL_TEST,ids);assert grid.index['images_sha256']=='cf834d2b4e6cdaa414689c8723d532178980f6ad459324d13a903321f1064888'
    base_reported={c['case_id']:c for c in json.loads(BASELINE_TEST_RAW.read_text())['summary']['per_case']};assert len(base_reported)==TEST_SCANS
    device=torch.device('cuda:0');torch.cuda.set_device(device);seed_all(20260819)
    write(R/'MODEL_PREACCESS.json',{'utc':now(),'candidate_checkpoint_sha256':sha(CANDIDATE_CKPT),'baseline_checkpoint_sha256':BASELINE_CKPT_SHA,'cache_receipt_sha256':sha(R/'CACHE_RECEIPT.json'),'preaccess_sha256':sha(R/'PREACCESS.json'),'batch_size':2,'precision':'bfloat16','threshold':0,'one_forward':True,'no_augmentation_or_postprocessing':True,'candidate_grid':'native 1mm crop in 160x208x168 canvas, padding excluded from metrics','baseline_grids':['128^3 recompute (identity check)','native crop via trilinear logits (primary paired reference)','native crop via nearest mask (secondary)']})
    start=time.monotonic()
    # --- candidate: native-resolution e173 ---
    cand=model_for(device);state=torch.load(CANDIDATE_CKPT,map_location='cpu',weights_only=False);assert state['epoch']==173;cand.load_state_dict(state['model_state_dict'],strict=True);del state;cand.eval();cand.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic();cases=[]
    with torch.no_grad():
        for s in range(0,TEST_SCANS,2):
            items=[native[i] for i in range(s,min(s+2,TEST_SCANS))]
            image=torch.stack([it['image'] for it in items]).to(device);label=torch.stack([it['label'] for it in items]).to(device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=cand(image).segmentation_logits
            for k,it in enumerate(items):
                m=case_metrics(logits[k].float(),label[k],native.geometry[it['case_id']]);m.update(case_id=it['case_id'],subject_id=it['subject_id']);cases.append(m)
    torch.cuda.synchronize();cand_seconds=time.monotonic()-tick;cand_peak=torch.cuda.max_memory_allocated()
    assert len(cases)==TEST_SCANS and [c['case_id'] for c in cases]==ids and all(np.isfinite(c['locator_dsc']) for c in cases) and len({c['subject_id'] for c in cases})==TEST_SUBJECTS
    del cand;torch.cuda.empty_cache()
    # --- baseline e193 on the same native grid + 128^3 identity check ---
    base=model_for(device);state=torch.load(BASELINE_CKPT,map_location='cpu',weights_only=False);base.load_state_dict(state['model_state_dict'],strict=True);del state;base.eval();base.requires_grad_(False)
    tick=time.monotonic();b_tri=[];b_nn=[];b_128=[]
    with torch.no_grad():
        for s in range(0,TEST_SCANS,2):
            items=[grid[i] for i in range(s,min(s+2,TEST_SCANS))]
            image=torch.stack([it['image'] for it in items]).to(device);label128=torch.stack([it['label'] for it in items]).to(device)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=base(image).segmentation_logits.float()
            d128=dice_per_case(logits,label128)
            for k,it in enumerate(items):
                cid=it['case_id'];geo=native.geometry[cid];nat=native[s+k];assert nat['case_id']==cid
                sl=canvas_slices(geo);crop_label=nat['label'][(slice(None),)+sl].to(device);extent=tuple(int(v) for v in geo['extent'])
                up=F.interpolate(logits[k:k+1],size=extent,mode='trilinear',align_corners=False)[0]
                nn=F.interpolate((logits[k:k+1]>0).float(),size=extent,mode='nearest-exact')[0]*2-1
                flat=int(torch.argmax(up[0].flatten()));h,w=extent[1],extent[2];z=flat//(h*w);rem=flat%(h*w);y=rem//w;x=rem%w
                b_tri.append({'case_id':cid,'subject_id':it['subject_id'],'locator_dsc':float(dice_per_case(up.unsqueeze(0),crop_label.unsqueeze(0))[0]),'automatic_point':[int(z),int(y),int(x)],'automatic_point_hits_gt':bool(crop_label[0,z,y,x]>0.5),'crop_extent':list(extent)})
                b_nn.append({'case_id':cid,'subject_id':it['subject_id'],'locator_dsc':float(dice_per_case(nn.unsqueeze(0),crop_label.unsqueeze(0))[0]),'automatic_point':[int(z),int(y),int(x)],'automatic_point_hits_gt':bool(crop_label[0,z,y,x]>0.5)})
                rep=base_reported[cid];b_128.append({'case_id':cid,'subject_id':it['subject_id'],'locator_dsc':float(d128[k]),'reported_locator_dsc':float(rep['locator_dsc']),'automatic_point_hits_gt':bool(rep['automatic_point_hits_gt'])})
    torch.cuda.synchronize();base_seconds=time.monotonic()-tick
    repro_err=max(abs(x['locator_dsc']-x['reported_locator_dsc']) for x in b_128)
    if repro_err>1e-3:raise RuntimeError(f'baseline 128-grid reproduction error {repro_err}; wrong checkpoint or cache')
    raw={'status':'complete_test270','created_utc':now(),'candidate':{'method':'native_resolution_pure_segresnet_full972_epoch173','epoch':173,'checkpoint_sha256':sha(CANDIDATE_CKPT),'grid':'native 1mm image-only brain crop; canvas 160x208x168; padding excluded','summary':summarize(cases)},
         'baseline_e193':{'checkpoint_sha256':BASELINE_CKPT_SHA,'native_trilinear_logits':summarize(b_tri),'native_nearest_mask':summarize(b_nn),'grid128_recomputed':summarize([{**x,'automatic_point_hits_gt':x['automatic_point_hits_gt']} for x in b_128]),'grid128_reported_scan_dice':float(np.mean([x['reported_locator_dsc'] for x in b_128])),'max_grid128_reproduction_error':repro_err},
         'scans':TEST_SCANS,'subjects':TEST_SUBJECTS,'test_access':True,'not_pristine_blind':True,'not_external':True,'prior_test_informed_development':True,'test_used_for_current_selection_or_tuning':False,
         'protocol':{'model':'pure SegResNet4700897','training_scans':972,'training_subjects':444,'train_epochs_completed':200,'validation_selection_epoch':173,'per_gpu_batch':4,'optimizer_updates':24400,'resolution':'native_1mm_crop_canvas_160x208x168','threshold_raw_logit':0,'precision':'bfloat16','batch':2,'tta':False,'postprocessing':False,'sam':False,'uses_GT_for_localization':False,'forwards':1},
         'runtime':{'candidate_inference_seconds':cand_seconds,'candidate_peak_allocated_bytes':cand_peak,'baseline_inference_seconds':base_seconds,'total_seconds':time.monotonic()-start,'gpu':torch.cuda.get_device_name()},
         'cache_index_sha256':receipt['index_sha256'],'cache_payload_sha256':{'images.f32':receipt['images_sha256'],'labels.u8':receipt['labels_sha256']},'canonical_test_cache_index_sha256':sha(CANONICAL_TEST/'index.json'),'freeze_sha256':sha(R/'FREEZE.json'),'preaccess_sha256':sha(R/'PREACCESS.json')}
    write(R/'RAW_TEST270.json',raw)
    metrics=summary_ci(cases)
    metrics.update(status='complete_test270',created_utc=now(),method=raw['candidate']['method'],epoch=173,checkpoint_sha256=raw['candidate']['checkpoint_sha256'],grid=raw['candidate']['grid'],point_hits=sum(c['automatic_point_hits_gt'] for c in cases),point_denominator=TEST_SCANS,
        baseline_e193_same_grid={'trilinear_logits':summary_ci(b_tri),'nearest_mask':summary_ci(b_nn),'grid128_recomputed':summary_ci(b_128),'grid128_reported_scan_dice':raw['baseline_e193']['grid128_reported_scan_dice'],'max_grid128_reproduction_error':repro_err},
        comparisons={'vs_e193_native_trilinear':paired(b_tri,cases),'vs_e193_native_nearest':paired(b_nn,cases)},comparison_scope='same reused internal test270 on the same native crop grid; candidate frozen from validation before this test; baseline resampled from its 128^3 prediction; descriptive, not external confirmation; cross-grid subtraction against 0.759179 is NOT valid',
        test_access=True,not_pristine_blind=True,not_external=True,prior_test_informed_development=True,current_test_used_for_selection_or_retuning=False,new_outer_confirmation_access=False,runtime=raw['runtime'],
        evidence={str(p.relative_to(ROOT)):sha(p) for p in [R/'RAW_TEST270.json',CANDIDATE_CKPT,R/'SELECTION_RECEIPT.json',R/'PREACCESS.json',R/'MODEL_PREACCESS.json',R/'CACHE_RECEIPT.json',R/'FREEZE.json']})
    write(R/'RESULT.json',metrics);write(R/'DONE.json',{'utc':now(),'status':'complete','result_sha256':sha(R/'RESULT.json'),'raw_sha256':sha(R/'RAW_TEST270.json')})
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('evidence',)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:write(R/'FAILURE_evaluate.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc(),'model_access_attempted':(R/'MODEL_PREACCESS.json').exists()});raise
