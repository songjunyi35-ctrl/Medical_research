"""One fixed cross-data-recipe fusion study; no member/weight grid or training."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import time
import torch
from torch.utils.data import DataLoader
from sammed3d.data_recipe_fusion import equal_recipe_logits
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_structured_background_aux import summarize,compare
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-10_data_recipe_ensemble'
SUBJECT=p.ROOT/'archive/research_records/2026-09-07_subject_balanced_sampling'
OCCUPANCY=p.ROOT/'archive/research_records/2026-09-10_fractional_occupancy'


def sources():
    p.check_freeze();p.check_roles();reports={};raws={}
    for name,record in [('sampling',SUBJECT),('occupancy',OCCUPANCY)]:
        result=json.loads((record/'RESULT.json').read_text());verification=json.loads((record/'VERIFICATION_RECEIPT.json').read_text())
        assert verification['status']=='passed';p.checked(record/'RESULT.json',verification['result_sha256']);p.checked(record/'RAW_DEVELOPMENT.json',result['raw_sha256'])
        assert not result['all_gates_passed'];raws[name]=json.loads((record/'RAW_DEVELOPMENT.json').read_text())
        for seed in p.SEEDS:
            folder='subject_balanced_sampling_20260907' if name=='sampling' else 'fractional_occupancy_20260910'
            arm='subject_balanced' if name=='sampling' else 'fractional_occupancy'
            path=p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/{folder}/seed{seed}_{arm}/TRAIN_DONE.json';d=json.loads(path.read_text())
            assert d['fixed_epochs']==168 and d['fit_sha256']==p.FIT_SHA and not d['teacher_checkpoints'];p.checked(d['checkpoint']['path'],d['checkpoint']['sha256'])
            reports[f'{seed}_{name}']={'train_report':{'path':str(path),'sha256':p.sha(path)},'checkpoint':d['checkpoint']}
    return reports,raws


def score(logits,truth,batch):
    result=[]
    for i,case in enumerate(batch['case_id']):
        pred=logits[i,0]>0;target=truth[i,0];inter=int((pred&target).sum());npred=int(pred.sum());nt=int(target.sum());point=int(logits[i,0].flatten().argmax());assert nt>0
        result.append({'case_id':case,'subject_id':batch['subject_id'][i],'dice':2*inter/(npred+nt),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':nt})
    return result


def gates_for(ensemble,members,native_comparison,member_comparisons):
    return {'scan_above_both_members':ensemble['scan_dice']>max(m['scan_dice'] for m in members),
            'subject_above_both_members':ensemble['subject_dice']>max(m['subject_dice'] for m in members),
            'best_member_point_safety':ensemble['point_hits']>=max(m['point_hits'] for m in members)-1,
            'native_subject_gain':native_comparison['subject_delta']>=.003,
            'native_scan_positive':native_comparison['scan_delta']>0,
            'all_member_scan_ci_safety':all(c['scan_cluster_95ci'][0]>=-.003 for c in member_comparisons),
            'all_member_subject_ci_safety':all(c['subject_95ci'][0]>=-.003 for c in member_comparisons)}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['smoke','development'],required=True);args=parser.parse_args()
    reports,raws=sources();fit,dev=p.check_roles()
    output=RECORD/('SMOKE_RECEIPT.json' if args.phase=='smoke' else 'RAW_DEVELOPMENT.json')
    if output.exists():raise FileExistsError(output)
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats(device)
    files=['sammed3d/data_recipe_fusion.py','tests/test_data_recipe_fusion.py','scripts/evaluate_data_recipe_fusion.py','scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','archive/research_records/2026-09-10_data_recipe_ensemble/IDEA.md','archive/research_records/2026-09-10_data_recipe_ensemble/PREREG.md']
    if args.phase=='development':
        if (RECORD/'DEV_PREACCESS.json').exists():raise FileExistsError('single fusion readout already started')
        smoke=json.loads((RECORD/'SMOKE_RECEIPT.json').read_text());assert smoke['status']=='passed'
        for name,digest in smoke['files'].items():p.checked(p.ROOT/name,digest)
        cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text());p.checked(p.CACHE/'images.f32',cache['source_images_sha256']);p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
        p.write_new(RECORD/'DEV_PREACCESS.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.ROOT/n) for n in files},'models':reports,'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'cache_index_sha256':p.INDEX_SHA,'source_raws':{n:{'path':str(r/'RAW_DEVELOPMENT.json'),'sha256':p.sha(r/'RAW_DEVELOPMENT.json')} for n,r in [('sampling',SUBJECT),('occupancy',OCCUPANCY)]},'fusion':'0.5 raw sampling logits +0.5 raw occupancy logits','threshold':0,'bootstrap':{'samples':10000,'seeds':[2026091111,2026091112],'unit':'patient_cluster'},'new_forwards':0,'expected_forwards':720,'fixed_test_seed_if_eligible':2026090711,'test_access':False})
    selected=fit[:8] if args.phase=='smoke' else dev
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in selected]),batch_size=8,num_workers=0,shuffle=False)
    native=json.loads((p.RECORD/'RAW_DEVELOPMENT.json').read_text());results={};comparisons={};gates={};forwards=0;start=time.monotonic();partial={}
    try:
        for i,seed in enumerate(p.SEEDS):
            models={}
            for name in ('sampling','occupancy'):
                model=AutomaticPromptLocator(center_aware=False).to(device).eval();ck=torch.load(reports[f'{seed}_{name}']['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True);models[name]=model
            collected={k:[] for k in ('sampling','occupancy','ensemble')};partial={'seed':seed,'collected':collected}
            for batch in loader:
                image=batch['image'].to(device);truth=batch['label'].to(device)>.5;logits={}
                for name,model in models.items():
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits[name]=model(image).segmentation_logits.float()
                    forwards+=len(image)
                logits['ensemble']=equal_recipe_logits(logits['sampling'],logits['occupancy'])
                assert torch.cuda.max_memory_allocated(device)<12*1024**3
                if args.phase=='development':
                    for name in collected:collected[name].extend(score(logits[name],truth,batch))
            if args.phase=='development':
                for name in ('sampling','occupancy'):assert collected[name]==raws[name]['results'][str(seed)]['candidate']['per_case'],f'{seed}/{name} old member parity failed'
                summaries={name:summarize(value,2026091111+i) for name,value in collected.items()}
                contrasts={name:compare(summaries[name],summaries['ensemble'],2026091111+i) for name in ('sampling','occupancy')}
                contrasts['native']=compare(native['results'][f'seed{seed}_control'],summaries['ensemble'],2026091111+i)
                results[str(seed)]=summaries;comparisons[str(seed)]=contrasts;gates[str(seed)]=gates_for(summaries['ensemble'],[summaries[n] for n in ('sampling','occupancy')],contrasts['native'],[contrasts[n] for n in ('sampling','occupancy')])
            del models,model,ck,logits,image,truth;torch.cuda.empty_cache()
        if args.phase=='smoke':
            assert forwards==32
            body={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),'fit_cases':[r['case_id'] for r in selected],'new_forwards':forwards,'peak_bytes':torch.cuda.max_memory_allocated(device),'seconds':time.monotonic()-start,'files':{n:p.sha(p.ROOT/n) for n in files},'development_access':False,'test_access':False}
        else:
            assert forwards==720
            average=statistics.fmean(c['native']['scan_delta'] for c in comparisons.values());passed=all(all(g.values()) for g in gates.values()) and average>=.003
            body={'schema':'aflow.data_recipe_ensemble.development','created_utc':datetime.now(timezone.utc).isoformat(),'status':'completed','results':results,'comparisons':comparisons,'gates':gates,'average_native_scan_delta':average,'all_gates_passed':passed,'member_per_case_parity_exact':True,'new_forwards':forwards,'seconds':time.monotonic()-start,'peak_bytes':torch.cuda.max_memory_allocated(device),'dev_preaccess_sha256':p.sha(RECORD/'DEV_PREACCESS.json'),'access':{'reused_development180':True,'test':False,'confirmation':False,'outer':False,'external':False},'decision':'eligible_for_separately_frozen_fixed_seed11_test' if passed else 'close_fixed_equal_recipe_fusion_no_weight_member_seed_search'}
        p.write_new(output,body);print(json.dumps({k:v for k,v in body.items() if k!='results'},indent=2),flush=True)
    except BaseException as error:
        p.write_new(RECORD/f'{args.phase.upper()}_FAILURE.json',{'error':repr(error),'new_forwards':forwards,'partial':partial,'completed_results':results,'test_access':False});raise


if __name__=='__main__':main()
