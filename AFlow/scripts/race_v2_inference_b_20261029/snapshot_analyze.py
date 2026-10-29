"""Final-only pair validation; choose both partner and recipe inside each fold."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import (metrics,paired,mix,decode,record,read_atoms,worsened)
from scripts.race_v2_inference_b_20261029.snapshot_partner import verify_prepared
S=R/'final_snapshot'

def configurations(partner):
    return [dict(id=f'{partner}__candidate{weight:g}_{space}_{decoder}',partner=partner,
        candidate_weight=weight,space=space,decoder=decoder) for weight in (.5,.75,.9)
        for space in ('logit','probability') for decoder in ('raw','full')]

@torch.inference_mode()
def main(cache,candidate_ready,handoff_path):
    from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import aggregate
    package=verify_prepared(handoff_path);lock=gpu_lock();numeric_mode()
    index,images,labels=data(cache);cohort=rows();config=read(RECIPE);index_sha=sha(Path(cache)/'index.json')
    partner_entries={p['id']:p for p in [package['selected_partner']]+[f['selected_partner'] for f in package['patient_folds']]}
    sources={'candidate':read_atoms(candidate_ready,package['candidate_checkpoint_sha256'],index_sha)}
    for name,partner in partner_entries.items():sources[name]=read_atoms(partner['atoms_ready'],partner['checkpoint_sha256'],index_sha)
    candidates=[c for partner in sorted(partner_entries) for c in configurations(partner)];assert len(candidates)<=24
    reference_ids=[f'{model}_{kind}' for model in sources for kind in ('identity','equal8','full')]
    records={key:[] for key in reference_ids+[c['id'] for c in candidates]};overlaps={name:[] for name in partner_entries}
    for slot,row in enumerate(cohort):
        guard();sl=slices(index['geometry'][row['case_id']]);image=np.array(images[slot][sl],np.float32)
        target=np.array(labels[slot][sl]>0);values={};tensors={}
        for model,source in sources.items():
            logits={}
            for view in range(8):
                entry=source[slot,view];path=Path(entry['path']);assert sha(path)==entry['sha256']
                value=np.load(path,allow_pickle=False);assert value.dtype==np.float32 and value.shape==image.shape
                assert list(value.shape)==entry['shape'];logits[view]=torch.from_numpy(value).cuda()
            tensors[model]=logits;mean_logit=sum(logits.values()).div(8)
            identity=logits[0].sigmoid();equal8=mean_logit.sigmoid()
            probability8=sum(z.sigmoid() for z in logits.values()).div(8)
            full,point=aggregate(logits,config['aggregation'],torch.from_numpy(image).cuda())
            values[model]=dict(identity=identity,equal8=equal8,probability8=probability8,full=full,point=point)
            for kind,p in [('identity',identity),('equal8',equal8),('full',full)]:
                mask,cut=decode(p,image,config,kind=='full',(identity>.5).cpu().numpy())
                point_score=point if kind=='full' else (logits[0] if kind=='identity' else mean_logit)
                if kind!='full':mask=(point_score>0).cpu().numpy()
                records[f'{model}_{kind}'].append(record(row,slot,mask,point_score,target,cut))
        a=(values['candidate']['equal8']>.5).cpu().numpy()
        for partner in partner_entries:
            b=(values[partner]['equal8']>.5).cpu().numpy()
            overlaps[partner].append(dict(case_id=row['case_id'],subject_id=row['subject_id'],
                candidate_fp_corrected_by_partner=int((a&~b&~target).sum()),candidate_fn_corrected_by_partner=int((~a&b&target).sum()),
                partner_fp_corrected_by_candidate=int((~a&b&~target).sum()),partner_fn_corrected_by_candidate=int((a&~b&target).sum()),
                disagreement_voxels=int((a^b).sum())))
        for candidate in candidates:
            partner=candidate['partner'];full=candidate['decoder']=='full';weight=candidate['candidate_weight'];space=candidate['space']
            if full:p=mix(values['candidate']['full'],values[partner]['full'],weight,space);point_score=p
            elif space=='logit':
                point_score=sum(tensors['candidate'].values()).div(8)*weight+sum(tensors[partner].values()).div(8)*(1-weight)
                p=point_score.sigmoid()
            else:p=mix(values['candidate']['probability8'],values[partner]['probability8'],weight,space);point_score=p
            original=mix(values['candidate']['identity'],values[partner]['identity'],weight,space)>.5
            mask,cut=decode(p,image,config,full,original.cpu().numpy())
            if not full and space=='logit':mask=(point_score>0).cpu().numpy()
            records[candidate['id']].append(record(row,slot,mask,point_score,target,cut))
        write(S/'DECODE_JOURNAL.json',dict(at=now(),completed=slot+1,total=108,pair_configurations=len(candidates)))
    write(S/'RAW_VALIDATION.json',dict(at=now(),per_configuration=records,complementarity=overlaps,
        model_forwards_added=0,GT_used_for_scoring_only=True,test_access=False))
    global_partner=package['selected_partner']['id']
    global_candidates=[c for c in candidates if c['partner']==global_partner]
    final_reference_ids=[f'{m}_{k}' for m in ('candidate',global_partner) for k in ('identity','equal8','full')]
    strongest_id=min(final_reference_ids,key=lambda key:(-metrics(records[key])['scan_dice'],key))
    strongest=records[strongest_id];reference=records['candidate_full'];ranking=[]
    for c in global_candidates:
        ranking.append(dict(**c,metrics=metrics(records[c['id']]),paired_to_candidate_full=paired(records[c['id']],reference),
            paired_to_strongest_single=paired(records[c['id']],strongest),models=2,forwards_per_scan=16))
    ranking.sort(key=lambda c:(-c['metrics']['scan_dice'],-c['metrics']['subject_dice'],c['id']))
    folds=[];oof=[None]*108
    for fold in package['patient_folds']:
        train=fold['development_slots'];test=fold['held_out_slots'];partner=fold['selected_partner']['id']
        local=[c for c in candidates if c['partner']==partner];assert len(local)==12
        selected=min(local,key=lambda c:(-np.mean([records[c['id']][i]['locator_dsc'] for i in train]),c['id']))
        for i in test:oof[i]=records[selected['id']][i]
        folds.append(dict(fold=fold['fold'],partner=partner,partner_selection=fold,selected=selected,
            held_out_metrics=metrics([records[selected['id']][i] for i in test]),
            held_out_paired_to_candidate_full=paired([records[selected['id']][i] for i in test],[reference[i] for i in test])))
    assert all(r is not None for r in oof)
    result=dict(at=now(),state='complete',split='validation',scans=108,subjects=50,training=False,test_access=False,
        frozen_partner=package,global_partner=global_partner,evaluated_partners=sorted(partner_entries),
        cached_pair_configurations=len(candidates),grid_per_partner=12,maximum_comparisons=24,
        single_models={key:metrics(records[key]) for key in reference_ids},fusion_ranking=ranking,
        strongest_single_reference=dict(id=strongest_id,metrics=metrics(strongest)),patient_folds=folds,
        oof_metrics=metrics(oof),oof_paired_to_candidate_full=paired(oof,reference),oof_paired_to_strongest_single=paired(oof,strongest),
        best_worsened_cases=worsened(records[ranking[0]['id']],reference),
        best_worsened_cases_vs_strongest_single=worsened(records[ranking[0]['id']],strongest),
        model_forwards=dict(candidate_new_cache=864,partner_cached_reused=864*len(partner_entries),per_pair_deployment=16,new_during_decode=0),
        interpretation='Global partner is a bounded heuristic, not an optimal/causal complementarity finding. Each fold selects partner and12-recipe using only four folds. Checkpoint/old models/decoder remain historically selected on reused data; exploratory, not independent confirmation')
    write(S/'RESULT.json',result);best=ranking[0]
    table='\n'.join(f"|{c['id']}|{c['metrics']['scan_dice']:.9f}|{c['metrics']['subject_dice']:.9f}|{c['metrics']['point_hits']}/108|" for c in ranking)
    (S/'RESULT.md').write_text(f"# 最终新候选/固定伙伴有限互补融合\n\n最终部署伙伴{global_partner}，统一12融合配置；每个部署仅2模型、16前向/例。分折各自仅四折选伙伴和12配方，第五折只评分；最多复用两旧伙伴、24缓存解码比较，不新增网络前向。\n\n|配置|Scan|Subject|Point|\n|---|---:|---:|---:|\n{table}\n\n最高{best['id']}；按患者分折合并Scan {result['oof_metrics']['scan_dice']:.9f}。伙伴规则仅有界启发式，不称最优或因果互补。模型选择和历史test-tuned仍复用数据，不是独立确认；本阶段无270访问。\n")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--candidate-ready',required=True);p.add_argument('--handoff',required=True);a=p.parse_args()
    try:main(a.cache,a.candidate_ready,a.handoff)
    except BaseException as error:
        import traceback
        write(S/f'ANALYSIS_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),model_forwards_added=0));raise
