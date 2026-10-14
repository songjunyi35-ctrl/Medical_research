"""One fixed8-view application to two frozen patient-sampling checkpoints."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts import structured_background_protocol as p
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import flip8_logits,REFLECTION_AXES
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts.evaluate_data_recipe_fusion import score

RECORD=p.ROOT/'archive/research_records/2026-09-10_subject_fixed_flip8'
SOURCE=p.ROOT/'archive/research_records/2026-09-07_subject_balanced_sampling'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','development'],required=True);args=parser.parse_args()
    p.check_freeze();fit,dev=p.check_roles();out=RECORD/('SMOKE_RECEIPT.json' if args.phase=='smoke' else 'RAW_DEVELOPMENT.json')
    if out.exists():raise FileExistsError(out)
    source_result=json.loads((SOURCE/'RESULT.json').read_text());p.checked(SOURCE/'RAW_DEVELOPMENT.json',source_result['raw_sha256']);source=json.loads((SOURCE/'RAW_DEVELOPMENT.json').read_text())
    models={}
    for seed in p.SEEDS:
        path=p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed{seed}_subject_balanced/TRAIN_DONE.json';d=json.loads(path.read_text())
        assert d['fixed_epochs']==168 and d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints'];p.checked(d['checkpoint']['path'],d['checkpoint']['sha256']);models[str(seed)]={'train_report':{'path':str(path),'sha256':p.sha(path)},'checkpoint':d['checkpoint']}
    files=['scripts/evaluate_subject_fixed_flip8.py','scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py','scripts/evaluate_data_recipe_fusion.py','scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','archive/research_records/2026-09-10_subject_fixed_flip8/IDEA.md','archive/research_records/2026-09-10_subject_fixed_flip8/PREREG.md']
    if args.phase=='development':
        if (RECORD/'DEV_PREACCESS.json').exists():raise FileExistsError('single readout already started')
        smoke=json.loads((RECORD/'SMOKE_RECEIPT.json').read_text());assert smoke['status']=='passed'
        for name,digest in smoke['files'].items():p.checked(p.ROOT/name,digest)
        cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
        p.write_new(RECORD/'DEV_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'models':models,'files':{n:p.sha(p.ROOT/n) for n in files},'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'source_raw_sha256':p.sha(SOURCE/'RAW_DEVELOPMENT.json'),'cache_index_sha256':p.INDEX_SHA,'views':[list(a) for a in REFLECTION_AXES],'fusion':'ordered float32 raw-logit mean after inverse reflection','threshold':0,'bootstrap':{'samples':10000,'seeds':[2026091121,2026091122],'unit':'patient_cluster'},'expected_forwards':2880,'new_forwards':0,'fixed_test_seed_if_eligible':2026090711,'test_access':False})
    selected=fit[:8] if args.phase=='smoke' else dev
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in selected]),batch_size=8,num_workers=0,shuffle=False)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats(device)
    result={};comparisons={};gates={};forwards=0;start=time.monotonic();partial={}
    try:
        with torch.inference_mode():
            for i,seed in enumerate(p.SEEDS):
                model=AutomaticPromptLocator(center_aware=False).to(device).eval();ck=torch.load(models[str(seed)]['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True)
                rows={'identity':[],'flip8':[]};partial={'seed':seed,'rows':rows}
                for batch in loader:
                    image=batch['image'].to(device);truth=batch['label'].to(device)>.5
                    identity,flip8=flip8_logits(model,image);forwards+=len(image)*8
                    assert torch.isfinite(identity).all() and torch.isfinite(flip8).all()
                    if args.phase=='development':rows['identity'].extend(score(identity,truth,batch));rows['flip8'].extend(score(flip8,truth,batch))
                if args.phase=='development':
                    assert rows['identity']==source['results'][str(seed)]['candidate']['per_case'],f'identity parity {seed}'
                    summaries={k:summarize(v,2026091121+i) for k,v in rows.items()};c=compare(summaries['identity'],summaries['flip8'],2026091121+i)
                    g={'scan_positive':c['scan_delta']>0,'subject_positive':c['subject_delta']>0,'own_point_safety':c['point_delta']>=-1,'scan_ci_safety':c['scan_cluster_95ci'][0]>=-.003,'subject_ci_safety':c['subject_95ci'][0]>=-.003,'native_point_floor':summaries['flip8']['point_hits']>=[169,162][i]}
                    result[str(seed)]=summaries;comparisons[str(seed)]=c;gates[str(seed)]=g
                del model,ck,image,truth,identity,flip8;torch.cuda.empty_cache()
        if args.phase=='smoke':
            assert forwards==128
            body={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.ROOT/n) for n in files},'forwards':forwards,'fit_cases':[r['case_id'] for r in selected],'seconds':time.monotonic()-start,'peak_bytes':torch.cuda.max_memory_allocated(device),'development_access':False,'test_access':False}
        else:
            assert forwards==2880
            avg=statistics.fmean(c['scan_delta'] for c in comparisons.values());passed=all(all(g.values()) for g in gates.values()) and avg>=.003
            body={'schema':'aflow.subject_fixed_flip8.development','status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'results':result,'comparisons':comparisons,'gates':gates,'average_scan_delta':avg,'all_gates_passed':passed,'identity_per_case_exact':True,'forwards':forwards,'seconds':time.monotonic()-start,'peak_bytes':torch.cuda.max_memory_allocated(device),'dev_preaccess_sha256':p.sha(RECORD/'DEV_PREACCESS.json'),'access':{'reused_development180':True,'test':False,'confirmation':False,'outer':False,'external':False},'decision':'eligible_fixed_seed11_test_requires_separate_freeze' if passed else 'close_fixed_subject_flip8_no_view_weight_threshold_seed_search'}
        p.write_new(out,body);print(json.dumps({k:v for k,v in body.items() if k!='results'},indent=2),flush=True)
    except BaseException as error:
        p.write_new(RECORD/f'{args.phase.upper()}_FAILURE.json',{'error':repr(error),'completed_forwards_lower_bound':forwards,'partial':partial,'completed_results':result,'test_access':False});raise


if __name__=='__main__':main()
