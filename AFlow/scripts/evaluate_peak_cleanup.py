"""One fixed-rule development readout, with exact historical baseline parity."""
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from sammed3d.peak_protected_cleanup import cleanup_logits
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts import peak_cleanup_protocol as p


def score(x,target,case,subject):
    pred=x>0;nt=int(target.sum());npred=int(pred.sum());inter=int((pred&target).sum());assert nt>0
    point=int(x.argmax())
    return {'case_id':case,'subject_id':subject,'dice':2*inter/(npred+nt),'point_hit':bool(target.ravel()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':nt}


def gates_for(c,all_points_equal):
    return {'subject_gain':c['subject_delta']>=.003,'scan_positive':c['scan_delta']>0,'point_exact':all_points_equal and c['point_delta']==0,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'scan_cluster_ci_safety':c['scan_cluster_95ci'][0]>=-.003}


@torch.inference_mode()
def main():
    out=p.RECORD/'RAW_DEVELOPMENT.json';pre=p.RECORD/'DEV_PREACCESS.json'
    if out.exists() or pre.exists():raise FileExistsError('one readout already started')
    controls=p.source_controls();_,dev=p.check_roles()
    smoke_path=p.RECORD/'SMOKE_RECEIPT.json';smoke=json.loads(smoke_path.read_text());assert smoke['status']=='passed' and all(smoke['gates'].values())
    for name,digest in smoke['files'].items():p.checked(p.ROOT/name,digest)
    source_path=p.source.RECORD/'RAW_DEVELOPMENT.json';p.checked(source_path,'80e9bd449d7c72a61de0ecdbdfbc5bb3a61814e7770bbcc1c983ac097b4cd2c6');source=json.loads(source_path.read_text())
    cache=json.loads((p.source.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256']);p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    free,total=torch.cuda.mem_get_info(device)
    if free<14*1024**3:raise RuntimeError('wait for >=14GiB; no development pixels/forwards yet')
    torch.cuda.set_per_process_memory_fraction(.3,device);torch.cuda.reset_peak_memory_stats(device)
    files=['scripts/evaluate_peak_cleanup.py','scripts/peak_cleanup_protocol.py','scripts/smoke_peak_cleanup.py','sammed3d/peak_protected_cleanup.py','tests/test_peak_protected_cleanup.py','scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','archive/research_records/2026-09-10_peak_protected_cleanup/IDEA.md','archive/research_records/2026-09-10_peak_protected_cleanup/PREREG.md']
    files.append('tests/test_peak_cleanup_evaluation.py')
    p.write_new(pre,{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'fixed_single_cleanup_rule_before_new_dev_forwards','files':{name:p.sha(p.ROOT/name) for name in files},'controls':controls,'source_raw':{'path':str(source_path),'sha256':p.sha(source_path)},'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'cache_index_sha256':p.INDEX_SHA,'cache_images_sha256':cache['source_images_sha256'],'cache_labels_sha256':cache['source_labels_sha256'],'smoke_sha256':p.sha(smoke_path),'rule':{'connectivity':26,'relative_largest_fraction':.01,'absolute_voxels':8,'always_keep_largest_ties':True,'always_keep_argmax_component':True,'removed_logit':-1},'inference':{'batch_size':8,'dtype':'bfloat16_autocast_float32_logits','threshold':0,'tta':False,'expected_new_forwards':360,'new_forwards_at_write':0},'bootstrap':{'samples':10000,'seeds':[2026091061,2026091062],'unit':'patient_cluster','selection_correction':False},'output':str(out),'test_access':False,'weak_elastic_candidate_development_access':False,'gpu_free_bytes_before':free})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,shuffle=False,num_workers=0)
    results={};comparisons={};allgates={};forwards=0;start=time.monotonic();partial={}
    try:
        for i,seed in enumerate(p.SEEDS):
            model=AutomaticPromptLocator(center_aware=False).to(device).eval()
            ck=torch.load(controls[str(seed)]['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
            base_rows=[];clean_rows=[];diagnostics=[]
            partial={'seed':seed,'base_rows':base_rows,'clean_rows':clean_rows,'diagnostics':diagnostics}
            for batch in loader:
                image=batch['image'].to(device)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(image).segmentation_logits.float()
                forwards+=len(image)
                assert torch.isfinite(logits).all()
                arrays=logits[:,0].cpu().numpy();truth=batch['label'][:,0].numpy()>.5
                for j,case in enumerate(batch['case_id']):
                    base=score(arrays[j],truth[j],case,batch['subject_id'][j])
                    expected=source['results'][f'seed{seed}_control']['per_case'][len(base_rows)]
                    if base!=expected:raise RuntimeError(f'historical base parity failed at {seed}/{case}: {base} vs {expected}')
                    cleaned,info=cleanup_logits(arrays[j]);clean=score(cleaned,truth[j],case,batch['subject_id'][j])
                    assert base['point_flat_index']==clean['point_flat_index'] and base['point_hit']==clean['point_hit']
                    removed=(arrays[j]>0)&(cleaned<=0)
                    diagnostics.append({'case_id':case,**info,'removed_true_positive_voxels':int((removed&truth[j]).sum()),'removed_false_positive_voxels':int((removed&~truth[j]).sum())})
                    base_rows.append(base);clean_rows.append(clean)
                if torch.cuda.max_memory_allocated(device)>12*1024**3:raise RuntimeError('12GiB inference memory gate')
            base=summarize(base_rows,2026091061+i);clean=summarize(clean_rows,2026091061+i);c=compare(base,clean,2026091061+i)
            points_equal=all(a['point_flat_index']==b['point_flat_index'] and a['point_hit']==b['point_hit'] for a,b in zip(base_rows,clean_rows,strict=True))
            results[str(seed)]={'base':base,'clean':clean,'component_diagnostics':diagnostics};comparisons[str(seed)]=c;allgates[str(seed)]=gates_for(c,points_equal)
            del model,ck,logits,image;torch.cuda.empty_cache()
        average=statistics.fmean(c['scan_delta'] for c in comparisons.values());passed=all(all(g.values()) for g in allgates.values()) and average>=.003
        assert forwards==360
        raw={'schema':'aflow.peak_protected_cleanup.reused_development','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':comparisons,'gates':allgates,'average_scan_delta':average,'all_gates_passed':passed,'base_historical_per_case_exact':True,'dev_preaccess_sha256':p.sha(pre),'new_forwards':forwards,'seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),'access':{'reused_development180':True,'weak_elastic_new_checkpoints':False,'test':False,'confirmation':False,'fixed_validation108':False,'outer':False,'external':False},'decision':'eligible_only_for_conditional_independent_full864_refit' if passed else 'close_fixed_cleanup_rule_no_size_fraction_connectivity_search'}
        p.write_new(out,raw)
        print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2),flush=True)
    except BaseException as error:
        p.write_new(p.RECORD/'DEVELOPMENT_FAILURE.json',{'status':'failed','created_utc':datetime.now(timezone.utc).isoformat(),'error':repr(error),'completed_results':results,'partial_current_model':partial,'new_forwards':forwards,'dev_preaccess_sha256':p.sha(pre),'test_access':False});raise


if __name__=='__main__':main()
