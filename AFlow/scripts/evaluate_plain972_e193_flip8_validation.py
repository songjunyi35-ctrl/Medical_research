"""Fixed Flip8 migration on validation only; the pure193 baseline is immutable."""
import sys,json,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.segresnet_plain972_common import ROOT,VAL_CACHE,sha,write,now,model_for,seed_all,summary_ci
from scripts.train_t1ce_wt_capg3d import BrainCropDataset,dice_per_case,summarize_cases
from sammed3d.t1ce_wt_capg3d import automatic_prompt_points,LocatorOutput
from scripts.sammed3d_full972_protocol import check_roles
R=ROOT/'archive/research_records/2026-09-18_plain972_e193_flip8_validation'
AXES=((),(-1,),(-2,),(-2,-1),(-3,),(-3,-1),(-3,-2),(-3,-2,-1))


@torch.inference_mode()
def predictions(model,image):
    total=None;identity=None
    for axes in AXES:
        transformed=torch.flip(image,axes) if axes else image
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=image.is_cuda):z=model(transformed).segmentation_logits
        restored=(torch.flip(z,axes) if axes else z).float()
        if identity is None:identity=restored.clone();total=torch.zeros_like(restored)
        total.add_(restored)
    return identity,total/8


def rows_for(z,y,batch):
    dice=dice_per_case(z.float(),y);points=automatic_prompt_points(LocatorOutput(z,None)).long()
    rows=[]
    for i,case_id in enumerate(batch['case_id']):
        a,b,c=[int(v) for v in points[i,0].tolist()]
        rows.append({'case_id':case_id,'subject_id':batch['subject_id'][i],'locator_dsc':float(dice[i]),'automatic_point':[a,b,c],'automatic_point_hits_gt':bool(y[i,0,a,b,c]>.5)})
    return rows


def paired(left,right):
    assert [(x['case_id'],x['subject_id']) for x in left]==[(x['case_id'],x['subject_id']) for x in right]
    subjects=sorted({r['subject_id'] for r in left});n=np.array([sum(r['subject_id']==s for r in left) for s in subjects])
    dif=np.array([b['locator_dsc']-a['locator_dsc'] for a,b in zip(left,right)])
    sums=np.array([sum(d for r,d in zip(left,dif) if r['subject_id']==s) for s in subjects]);ix=np.random.default_rng(20260918).integers(0,len(subjects),(10000,len(subjects)))
    return {'scan_delta':float(sums.sum()/n.sum()),'subject_delta':float((sums/n).mean()),'scan_delta_ci95':np.quantile(sums[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist(),'subject_delta_ci95':np.quantile((sums/n)[ix].mean(1),[.025,.975]).tolist(),'point_hits_delta':int(sum(r['automatic_point_hits_gt'] for r in right)-sum(r['automatic_point_hits_gt'] for r in left)),'improved_scans':int((dif>0).sum()),'worse_scans':int((dif<0).sum()),'tied_scans':int((dif==0).sum()),'bootstrap':'10000 patient cluster paired percentile95 seed20260918'}


def main():
    if (R/'PREACCESS.json').exists():raise RuntimeError('validation already attempted; no automatic repeats')
    freeze=json.loads((R/'FREEZE.json').read_text())
    for name,h in freeze['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('frozen source changed: '+name)
    _,rows=check_roles();ids=[r['case_id'] for r in rows]
    write(R/'PREACCESS.json',{'utc':now(),'purpose':'fixed108 validation migration vs immutable pure193 baseline','validation_pixels_access':True,'test_pixels_access':False,'outer_confirmation_access':False,'checkpoint_sha256':sha(R/'checkpoint_epoch193.pth'),'freeze_sha256':sha(R/'FREEZE.json'),'prior_test_informed_development':True})
    index=json.loads((VAL_CACHE/'index.json').read_text())
    assert index['count']==108 and index['test_access'] is False
    hashes={}
    for filename,key in [('images.f32','images_sha256'),('labels.u8','labels_sha256')]:hashes[filename]=sha(VAL_CACHE/filename);assert hashes[filename]==index[key]
    write(R/'CACHE_RECEIPT.json',{'utc':now(),'path':str(VAL_CACHE.relative_to(ROOT)),'index_sha256':sha(VAL_CACHE/'index.json'),'payload_sha256':hashes,'validation_only':True})
    dataset=BrainCropDataset(VAL_CACHE,expected_split='validation',case_ids=ids);loader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
    torch.set_num_threads(2);torch.cuda.set_device(0);seed_all(20260819);device=torch.device('cuda:0')
    state=torch.load(R/'checkpoint_epoch193.pth',map_location='cpu',weights_only=False);assert state['epoch']==193
    model=model_for(device);model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval();model.requires_grad_(False)
    identity=[];flip=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        image=batch['image'].to(device);target=batch['label'].to(device)
        z,avg=predictions(model,image)
        identity+=rows_for(z,target,batch);flip+=rows_for(avg,target,batch)
        if len(identity)%20==0 or len(identity)==108:print(json.dumps({'utc':now(),'complete':len(identity),'total':108}),flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    assert len(identity)==len(flip)==108 and len({r['subject_id'] for r in identity})==50
    original=json.loads((ROOT/freeze['baseline_validation_report']).read_text())['per_case']
    assert [r['case_id'] for r in identity]==[r['case_id'] for r in original]
    error=max(abs(a['locator_dsc']-b['locator_dsc']) for a,b in zip(identity,original));points=all(a['automatic_point']==b['automatic_point'] and a['automatic_point_hits_gt']==b['automatic_point_hits_gt'] for a,b in zip(identity,original))
    raw={'utc':now(),'epoch':193,'cohort':'reused validation108/50','identity':summarize_cases(identity),'flip8':summarize_cases(flip),'identity_max_abs_dice_error':error,'identity_points_exact':points,'checkpoint_sha256':sha(R/'checkpoint_epoch193.pth'),'test_pixels_access':False,'runtime_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'case_forwards':864,'axes':AXES,'aggregation':'ordered float32 raw-logit mean; threshold0'}
    write(R/'RAW_VALIDATION.json',raw)
    if error>1e-7 or not points:raise RuntimeError('identity parity failed; do not interpret candidate')
    comparison=paired(identity,flip)
    gates={'scan_delta_at_least_0p003':comparison['scan_delta']>=.003,'subject_nonnegative':comparison['subject_delta']>=0,'point_loses_at_most_one':comparison['point_hits_delta']>=-1,'scan_paired_ci_lower_positive':comparison['scan_delta_ci95'][0]>0}
    result={'status':'validation_completed','utc':now(),'identity':summary_ci(identity),'flip8':summary_ci(flip),'comparison':comparison,'gates':gates,'all_gates_passed':all(gates.values()),'test_authorized':False,'test_pixels_access':False,'new_training':False,'prior_test_informed_exploratory':True,'runtime_seconds':elapsed,'peak_allocated_bytes':raw['peak_allocated_bytes'],'identity_parity':{'max_dice_error':error,'points_exact':points},'cost':{'baseline_forwards':1,'candidate_forwards':8,'actual_validation_case_forwards':864,'extra_trained_parameters':0,'new_image_cache_bytes':0},'evidence':{str(p.relative_to(ROOT)):sha(p) for p in [R/'RAW_VALIDATION.json',R/'checkpoint_epoch193.pth',R/'FREEZE.json',R/'PREACCESS.json',R/'CACHE_RECEIPT.json']}}
    write(R/'RESULT.json',result);write(R/'DONE.json',{'utc':now(),'status':'complete','result_sha256':sha(R/'RESULT.json')});print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:write(R/'FAILURE.json',{'utc':now(),'error':repr(exc),'traceback':traceback.format_exc(),'validation_attempted':(R/'PREACCESS.json').exists(),'test_access':False});raise
