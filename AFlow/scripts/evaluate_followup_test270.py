"""Fixed166/native and172/128 test; store candidate raw before reference/statistics."""
import sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from scripts.followup_test270_common import *
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset


def load_model(path,expected_epoch,device):
    model=model_for(device);state=torch.load(path,map_location='cpu',weights_only=False);assert state['epoch']==expected_epoch
    model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval();model.requires_grad_(False);return model


def reference_projection(native,grid,device):
    if (S/'PRETRAINED179_NATIVE_RAW.json').exists():return json.loads((S/'PRETRAINED179_NATIVE_RAW.json').read_text())
    write_new(S/'REFERENCE_MODEL_PREACCESS.json',{'utc':now(),'checkpoint_sha256':REF_SHA,'role':'fixed already-tested179 same-grid reference projection',
              'test_access':True,'not_new_candidate':True,'no_retuning':True,'freeze_sha256':sha(S/'FREEZE.json')})
    model=load_model(PRETRAINED/'checkpoint_selected.pth',179,device);rows=[];identities=[];tick=time.monotonic()
    old={x['case_id']:x for x in json.loads((PRETRAINED/'RAW_TEST270.json').read_text())['summary']['per_case']}
    with torch.inference_mode():
        for start in range(0,270,2):
            items=[grid[i] for i in range(start,start+2)];image=torch.stack([x['image'] for x in items]).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
            for k,it in enumerate(items):
                cid=it['case_id'];nat=native[start+k];assert cid==nat['case_id'];geo=native.geometry[cid]
                label=nat['label'][(slice(None),)+canvas_slices(geo)].to(device)
                up=F.interpolate(logits[k:k+1],size=tuple(geo['extent']),mode='trilinear',align_corners=False)[0]
                d=float(dice_per_case(up[None],label[None])[0]);flat=int(up[0].flatten().argmax());h,w=up.shape[-2:]
                point=[flat//(h*w),(flat%(h*w))//w,flat%w]
                rows.append({'case_id':cid,'subject_id':it['subject_id'],'locator_dsc':d,'automatic_point':point,
                             'automatic_point_hits_gt':bool(label[0,*point]>.5),'crop_extent':geo['extent']})
                reproduced=float(dice_per_case(logits[k:k+1],it['label'][None].to(device))[0])
                identities.append({'case_id':cid,'recomputed':reproduced,'reported':old[cid]['locator_dsc']})
    validate_rows(rows);error=max(abs(x['recomputed']-x['reported']) for x in identities)
    raw={'utc':now(),'status':'complete_reference_projection' if error<=1e-6 else 'identity_check_failed','summary':summarize_cases(rows),'grid128_identity':identities,
         'max_grid128_dice_error':error,'checkpoint_sha256':REF_SHA,'seconds':time.monotonic()-tick,'test_access':True,
         'grid':'native grid from trilinearly resampled 128^3 logits','not_new_candidate':True}
    write(S/'PRETRAINED179_NATIVE_RAW.json',raw)
    if error>1e-6:raise RuntimeError(f'fixed179 128 reproduction error {error}; no automatic repeat')
    return raw


def candidate(name):
    r=RECORDS[name]
    if (r/'RAW_TEST270.json').exists():return json.loads((r/'RAW_TEST270.json').read_text())
    assert (r/'PREACCESS.json').exists();assert sha(r/'checkpoint_selected.pth')==HASHES[name]
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2);seed_all(20260819);numeric_mode()
    model=load_model(r/'checkpoint_selected.pth',EPOCHS[name],device)
    ids=[x['case_id'] for x in read_rows(MANIFEST['test'])]
    if name=='native':
        receipt=json.loads((r/'CACHE_RECEIPT.json').read_text());assert receipt['historical_payload_exact']
        assert sha(TEST_CACHE/'index.json')==receipt['index_sha256'];ds=NativeTestDataset(TEST_CACHE,ids)
    else:ds=TestBrainCropDataset(CANONICAL_TEST,ids)
    write_new(r/'MODEL_PREACCESS.json',{'utc':now(),'epoch':EPOCHS[name],'checkpoint_sha256':HASHES[name],
              'selection_receipt_sha256':sha(r/'SELECTION_RECEIPT.json'),'freeze_sha256':sha(S/'FREEZE.json'),
              'precision':'bfloat16','batch':2,'threshold_raw_logit':0,'one_forward':True,'test_access':True,
              'grid':'native grid; padding excluded' if name=='native' else '128^3','current_test_tuning':False})
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    if name=='patient':result=validate(model,DataLoader(ds,batch_size=2,shuffle=False,num_workers=2,pin_memory=True),device)
    else:
        rows=[]
        with torch.inference_mode():
            for start in range(0,270,2):
                items=[ds[i] for i in range(start,start+2)];image=torch.stack([x['image'] for x in items]).to(device);label=torch.stack([x['label'] for x in items]).to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits
                for k,it in enumerate(items):
                    m=case_metrics(logits[k],label[k],ds.geometry[it['case_id']]);m.update(case_id=it['case_id'],subject_id=it['subject_id']);rows.append(m)
        result=summarize_cases(rows)
    torch.cuda.synchronize();validate_rows(result['per_case'])
    raw={'utc':now(),'status':'complete_candidate_inference','candidate':name,'epoch':EPOCHS[name],'checkpoint_sha256':HASHES[name],
         'summary':result,'test_access':True,'not_pristine_blind':True,'not_external':True,'prior_test_informed_development':True,
         'current_test_used_for_selection_or_tuning':False,'grid':'native crop 160x208x168 canvas' if name=='native' else '128^3',
         'runtime':{'inference_seconds':time.monotonic()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name()}}
    write(r/'RAW_TEST270.json',raw)
    return raw


def main(name):
    verify();numeric_mode();r=RECORDS[name]
    if (r/'RESULT.json').exists():return
    raw=candidate(name);rows=raw['summary']['per_case'];validate_rows(rows)
    if name=='patient':
        refs={'pretrained179':json.loads((PRETRAINED/'RAW_TEST270.json').read_text())['summary']['per_case'],
              'pure193':json.loads((PURE/'RAW_TEST270.json').read_text())['summary']['per_case']}
    else:
        ids=[x['case_id'] for x in read_rows(MANIFEST['test'])];native=NativeTestDataset(TEST_CACHE,ids);grid=TestBrainCropDataset(CANONICAL_TEST,ids)
        projected=reference_projection(native,grid,torch.device('cuda:0'))
        assert projected['max_grid128_dice_error']<=1e-6
        old=json.loads((OLD_NATIVE/'RAW_TEST270.json').read_text())
        refs={'pretrained179_native':projected['summary']['per_case'],'native173':old['candidate']['summary']['per_case'],
              'pure193_native':old['baseline_e193']['native_trilinear_logits']['per_case']}
    for x in refs.values():validate_rows(x)
    d=summary_ci(rows);d.update(status='complete_test270',utc=now(),candidate=name,epoch=EPOCHS[name],checkpoint_sha256=HASHES[name],
        point_hits=sum(x['automatic_point_hits_gt'] for x in rows),point_denominator=270,grid=raw['grid'],
        comparisons={key:paired(rows,value) for key,value in refs.items()},reference_summaries={key:summary_ci(value) for key,value in refs.items()},
        runtime=raw['runtime'],test_access=True,not_pristine_blind=True,not_external=True,prior_test_informed_development=True,
        current_test_used_for_selection_or_retuning=False,new_outer_confirmation_access=False,validation_gate_passed=False,
        evaluation_status='user-authorized exploratory test after failed validation gate')
    files=[r/'checkpoint_selected.pth',r/'SELECTION_RECEIPT.json',r/'PREACCESS.json',r/'MODEL_PREACCESS.json',r/'CACHE_RECEIPT.json',r/'RAW_TEST270.json',S/'FREEZE.json']
    if name=='native':files.extend([S/'PRETRAINED179_NATIVE_RAW.json',S/'REFERENCE_MODEL_PREACCESS.json'])
    d['evidence']={str(p.relative_to(ROOT)):sha(p) for p in files};write(r/'RESULT.json',d)
    print(json.dumps({k:d[k] for k in ['candidate','scan_dice','subject_dice','point_hits','comparisons']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException as e:
        write(RECORDS[sys.argv[1]]/'FAILURE.json',{'utc':now(),'error':repr(e),'traceback':traceback.format_exc(),
              'candidate_model_access':(RECORDS[sys.argv[1]]/'MODEL_PREACCESS.json').exists(),'candidate_raw_saved':(RECORDS[sys.argv[1]]/'RAW_TEST270.json').exists()});raise
