"""User-directed one-shot exploratory test after failed Flip8 validation gates."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from scripts.segresnet_plain972_common import ROOT,TRAIN_CACHE,sha,write,now,model_for,seed_all,summary_ci
from scripts.evaluate_plain972_e193_flip8_validation import AXES,predictions,rows_for,paired
from scripts.evaluate_t1ce_wt_plain_locator_test270 import TestBrainCropDataset
from scripts.train_t1ce_wt_capg3d import summarize_cases
from scripts.sammed3d_full972_protocol import check_roles,TEST
R=ROOT/'archive/research_records/2026-09-18_plain972_e193_flip8_test270'
CACHE=ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'

def main():
    if (R/'PREACCESS.json').exists():raise RuntimeError('test already attempted; no automatic repeat')
    freeze=json.loads((R/'FREEZE.json').read_text())
    for name,h in freeze['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('frozen file changed: '+name)
    check_roles()
    rows=[json.loads(line) for line in TEST.read_text().splitlines() if line.strip()]
    ids=[r['case_id'] for r in rows]
    assert len(ids)==270 and len(set(ids))==270
    checkpoint=ROOT/freeze['checkpoint_path']
    write(R/'PREACCESS.json',{'utc':now(),'purpose':'user-directed fixed Flip8 exploratory test after failed validation gates','user_authorization':freeze['user_authorization'],'test_pixels_access':True,'test_is_reused_internal':True,'prior_test_informed_development':True,'validation_gates_passed':False,'new_training':False,'outer_confirmation_access':False,'checkpoint_sha256':sha(checkpoint),'freeze_sha256':sha(R/'FREEZE.json')})
    index=json.loads((CACHE/'index.json').read_text())
    assert index['count']==270 and index['test_access'] is True and index['model_shape']==[128,128,128]
    assert index['preprocessing']==json.loads((TRAIN_CACHE/'index.json').read_text())['preprocessing']
    hashes={}
    for filename,key in [('images.f32','images_sha256'),('labels.u8','labels_sha256')]:
        hashes[filename]=sha(CACHE/filename)
        assert hashes[filename]==index[key]
    write(R/'CACHE_RECEIPT.json',{'utc':now(),'path':str(CACHE.relative_to(ROOT)),'index_sha256':sha(CACHE/'index.json'),'payload_sha256':hashes,'reused_existing_cache':True,'new_cache_bytes':0})
    dataset=TestBrainCropDataset(CACHE,expected_case_ids=ids)
    loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
    torch.set_num_threads(2);torch.cuda.set_device(0);seed_all(20260819);device=torch.device('cuda:0')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['epoch']==193
    model=model_for(device);model.load_state_dict(state['model_state_dict'],strict=True);del state
    model.eval();model.requires_grad_(False)
    identity=[];flip=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for batch in loader:
        image=batch['image'].to(device);target=batch['label'].to(device)
        z,avg=predictions(model,image)
        identity+=rows_for(z,target,batch);flip+=rows_for(avg,target,batch)
        if len(identity)%30==0 or len(identity)==270:print(json.dumps({'utc':now(),'complete':len(identity),'total':270}),flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    raw={'utc':now(),'epoch':193,'cohort':'reused internal test270/119','identity':summarize_cases(identity),'flip8':summarize_cases(flip),'checkpoint_sha256':sha(checkpoint),'test_pixels_access':True,'validation_gates_passed':False,'user_directed_exploratory':True,'runtime_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'case_forwards':2160,'axes':AXES,'aggregation':'ordered float32 raw-logit mean; raw-logit threshold >0'}
    write(R/'RAW_TEST270.json',raw)
    assert len(identity)==len(flip)==270 and len({r['subject_id'] for r in identity})==119
    original=json.loads((ROOT/freeze['baseline_test_report']).read_text())['summary']['per_case']
    assert [(r['case_id'],r['subject_id']) for r in identity]==[(r['case_id'],r['subject_id']) for r in original]
    error=max(abs(a['locator_dsc']-b['locator_dsc']) for a,b in zip(identity,original))
    points=all(a['automatic_point']==b['automatic_point'] and a['automatic_point_hits_gt']==b['automatic_point_hits_gt'] for a,b in zip(identity,original))
    write(R/'PARITY.json',{'utc':now(),'baseline_raw_sha256':sha(ROOT/freeze['baseline_test_report']),'max_dice_error':error,'points_exact':points})
    if error>1e-7 or not points:raise RuntimeError('identity parity failed; do not interpret candidate')
    comparison=paired(identity,flip)
    result={'status':'exploratory_test_completed_after_failed_validation_gates','utc':now(),'identity':summary_ci(identity),'flip8':summary_ci(flip),'comparison':comparison,'point_hits':{'identity':sum(r['automatic_point_hits_gt'] for r in identity),'flip8':sum(r['automatic_point_hits_gt'] for r in flip)},'validation_gates_passed':False,'validation_gates_unchanged':True,'test_user_directed':True,'test_pixels_access':True,'test_reused_internal':True,'prior_test_informed_exploratory':True,'outer_confirmation_access':False,'new_training':False,'identity_parity':{'max_dice_error':error,'points_exact':points},'runtime_seconds':elapsed,'peak_allocated_bytes':raw['peak_allocated_bytes'],'cost':{'baseline_forwards':1,'candidate_forwards':8,'actual_test_case_forwards':2160,'extra_trained_parameters':0,'new_image_cache_bytes':0},'interpretation_gate':{'positive_scan_point_estimate':comparison['scan_delta']>0,'paired_scan_ci_lower_positive':comparison['scan_delta_ci95'][0]>0,'baseline_remains_pure193':True,'no_further_test_tuning':True},'evidence':{str(p.relative_to(ROOT)):sha(p) for p in [R/'RAW_TEST270.json',R/'PARITY.json',checkpoint,R/'FREEZE.json',R/'PREACCESS.json',R/'CACHE_RECEIPT.json']}}
    write(R/'RESULT.json',result);write(R/'DONE.json',{'utc':now(),'status':'complete','result_sha256':sha(R/'RESULT.json')});print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(R/'FAILURE.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc(),'test_access_attempted':(R/'PREACCESS.json').exists()});raise
