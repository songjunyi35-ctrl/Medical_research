"""Twelve frozen global fusion choices; GT enters scoring only."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *

def configurations():
    return [dict(id=f'race{weight:g}_{space}_{decoder}',race_weight=weight,space=space,decoder=decoder)
            for weight in (.5,.75,.9) for space in ('logit','probability') for decoder in ('raw','full')]

def metrics(records):
    values=np.array([r['locator_dsc'] for r in records],np.float64)
    subjects=sorted({r['subject_id'] for r in records})
    return dict(scan_dice=float(values.mean()),subject_dice=float(np.mean([
        values[[i for i,r in enumerate(records) if r['subject_id']==s]].mean() for s in subjects])),
        point_hits=sum(r['automatic_point_hits_gt'] for r in records),scans=len(records),subjects=len(subjects))

def paired(records,reference):
    assert [r['case_id'] for r in records]==[r['case_id'] for r in reference]
    delta=np.array([r['locator_dsc']-b['locator_dsc'] for r,b in zip(records,reference)])
    subjects=sorted({r['subject_id'] for r in records})
    groups=[np.array([i for i,r in enumerate(records) if r['subject_id']==s]) for s in subjects]
    sums=np.array([delta[g].sum() for g in groups]); counts=np.array([len(g) for g in groups])
    rng=np.random.default_rng(20261029); take=rng.integers(len(groups),size=(10000,len(groups)))
    boot=sums[take].sum(1)/counts[take].sum(1)
    subject_delta=sums/counts;subject_boot=subject_delta[take].mean(1)
    return dict(scan_delta=float(delta.mean()),scan_delta_ci95=np.quantile(boot,[.025,.975]).tolist(),
         subject_delta=float(subject_delta.mean()),subject_delta_ci95=np.quantile(subject_boot,[.025,.975]).tolist(),
         improved=int((delta>0).sum()),worsened=int((delta<0).sum()),equal=int((delta==0).sum()),
         worsened_at_least_001=int((delta<=-.01).sum()),
         bootstrap='10000 patient-cluster paired, seed20261029; descriptive post-selection interval')

def mix(left,right,weight,space):
    if space=='probability': return left*weight+right*(1-weight)
    return (torch.logit(left.clamp(1e-7,1-1e-7))*weight+
            torch.logit(right.clamp(1e-7,1-1e-7))*(1-weight)).sigmoid()

def decode(probability,image,config,full,original):
    if not full: return (probability>.5).cpu().numpy(),None
    from scripts.segresnet_auto_inference12h_20261028.promote_v6 import single_cut
    from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
    from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import apply
    from scripts.segresnet_auto_inference12h_20261028.postprocess_sweep_v1 import apply_global
    valid=supports(image)[config['aggregation']['margin']]
    codes=(probability.clamp(0,1)*65535+.5).long().cpu().numpy()
    cut=single_cut(np.bincount(codes[valid],minlength=65536),config['decoder'])
    mask=(codes>=cut)&valid
    for rule in config['postprocessing']:
        if rule.get('kind') in ('minimum','relative','largest'):mask=apply(mask,rule)
        else:mask=apply_global(mask,original,rule,{'selected':mask})
    return mask,int(cut)

def record(row,slot,mask,probability,target,cut):
    point=list(map(int,np.unravel_index(int(probability.argmax()),mask.shape)))
    tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
    return dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,
        locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=pred-tp,fn=positive-tp,
        automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),quantized_cut=cut)

def read_atoms(path,expected_sha,index_sha):
    ready=read(path);assert ready['checkpoint_sha256']==expected_sha
    assert ready['manifest_sha256']==sha(MANIFEST) and ready['index_sha256']==index_sha
    atoms={(a['slot'],a['view']):a for a in ready['atoms']}
    assert len(atoms)==864 and set(atoms)=={(s,v) for s in range(108) for v in range(8)}
    return atoms

def worsened(records,reference):
    return sorted([dict(case_id=r['case_id'],subject_id=r['subject_id'],delta=r['locator_dsc']-b['locator_dsc'],
        dice=r['locator_dsc'],reference_dice=b['locator_dsc']) for r,b in zip(records,reference)
        if r['locator_dsc']<b['locator_dsc']],key=lambda r:r['delta'])

@torch.inference_mode()
def main(cache,race_ready):
    from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import aggregate
    lock=gpu_lock();verify();numeric_mode();index,images,labels=data(cache);cohort=rows();config=read(RECIPE)
    index_sha=sha(Path(cache)/'index.json')
    sources={'race':read_atoms(race_ready,RACE_SHA,index_sha),'w32':read_atoms(OUT/'ATOMS_READY.json',CHECKPOINT_SHA,index_sha)}
    candidates=configurations();reference_ids=[f'{m}_{kind}' for m in sources for kind in ('identity','equal8','full')]
    records={key:[] for key in reference_ids+[r['id'] for r in candidates]};overlaps=[]
    for slot,row in enumerate(cohort):
        guard();cid=row['case_id'];sl=slices(index['geometry'][cid]);image=np.array(images[slot][sl],np.float32)
        target=np.array(labels[slot][sl]>0);values={};tensors={}
        for model,source in sources.items():
            logits={}
            for view in range(8):
                entry=source[slot,view];path=Path(entry['path']);assert sha(path)==entry['sha256']
                value=np.load(path,allow_pickle=False);assert value.dtype==np.float32 and value.shape==image.shape
                assert list(value.shape)==entry['shape'];logits[view]=torch.from_numpy(value).cuda()
            tensors[model]=logits
            mean_logit=sum(logits.values()).div(8)
            identity=logits[0].sigmoid();equal8=mean_logit.sigmoid()
            probability8=sum(z.sigmoid() for z in logits.values()).div(8)
            full,point=aggregate(logits,config['aggregation'],torch.from_numpy(image).cuda())
            values[model]=dict(identity=identity,equal8=equal8,probability8=probability8,full=full,point=point)
            for kind,p in [('identity',identity),('equal8',equal8),('full',full)]:
                mask,cut=decode(p,image,config,kind=='full',(identity>.5).cpu().numpy())
                point_score=point if kind=='full' else (logits[0] if kind=='identity' else mean_logit)
                if kind!='full': mask=(point_score>0).cpu().numpy()
                records[f'{model}_{kind}'].append(record(row,slot,mask,point_score,target,cut))
        masks={m:(v['equal8']>.5).cpu().numpy() for m,v in values.items()}
        a,b=masks['race'],masks['w32']
        overlaps.append(dict(case_id=cid,subject_id=row['subject_id'],
            race_fp_corrected_by_w32=int((a&~b&~target).sum()),race_fn_corrected_by_w32=int((~a&b&target).sum()),
            w32_fp_corrected_by_race=int((~a&b&~target).sum()),w32_fn_corrected_by_race=int((a&~b&target).sum()),
            disagreement_voxels=int((a^b).sum())))
        for candidate in candidates:
            full=candidate['decoder']=='full';weight=candidate['race_weight'];space=candidate['space']
            if full:
                p=mix(values['race']['full'],values['w32']['full'],weight,space)
                point_score=p
            elif space=='logit':
                point_score=sum(tensors['race'].values()).div(8)*weight+sum(tensors['w32'].values()).div(8)*(1-weight)
                p=point_score.sigmoid()
            else:
                p=mix(values['race']['probability8'],values['w32']['probability8'],weight,space);point_score=p
            original=mix(values['race']['identity'],values['w32']['identity'],weight,space)>.5
            mask,cut=decode(p,image,config,full,original.cpu().numpy())
            if not full and space=='logit':mask=(point_score>0).cpu().numpy()
            records[candidate['id']].append(record(row,slot,mask,point_score,target,cut))
        write(R/'DECODE_JOURNAL.json',dict(at=now(),completed=slot+1,total=108))
    write(R/'RAW_VALIDATION.json',dict(at=now(),per_configuration=records,complementarity=overlaps,
         model_forwards_added=0,GT_used_for_scoring_only=True,test_access=False))
    reference=records['race_full'];ranking=[]
    strongest_single_id=min(reference_ids,key=lambda key:(-metrics(records[key])['scan_dice'],key))
    strongest_single=records[strongest_single_id]
    for candidate in candidates:
        ranking.append(dict(**candidate,metrics=metrics(records[candidate['id']]),
            paired_to_race_full=paired(records[candidate['id']],reference),
            paired_to_strongest_single=paired(records[candidate['id']],strongest_single),models=2,forwards_per_scan=16))
    ranking.sort(key=lambda c:(-c['metrics']['scan_dice'],-c['metrics']['subject_dice'],c['id']))
    # Every subject stays in one fold. Choices use four folds; fold five is scored once.
    subjects=sorted({r['subject_id'] for r in cohort},key=lambda s:hashlib.sha256(('race_v2_20261029:'+s).encode()).hexdigest())
    fold_for={s:i%5 for i,s in enumerate(subjects)};oof=[None]*108;folds=[]
    for fold in range(5):
        train=[i for i,r in enumerate(cohort) if fold_for[r['subject_id']]!=fold]
        test=[i for i,r in enumerate(cohort) if fold_for[r['subject_id']]==fold]
        selected=min(candidates,key=lambda c:(-np.mean([records[c['id']][i]['locator_dsc'] for i in train]),c['id']))
        for i in test:oof[i]=records[selected['id']][i]
        folds.append(dict(fold=fold,subjects=[s for s in subjects if fold_for[s]==fold],selected=selected,
            held_out_metrics=metrics([records[selected['id']][i] for i in test]),
            held_out_paired_to_race_full=paired([records[selected['id']][i] for i in test],[reference[i] for i in test])))
    race_d=np.array([r['locator_dsc'] for r in records['race_equal8']]);w32_d=np.array([r['locator_dsc'] for r in records['w32_equal8']])
    result=dict(at=now(),state='complete',split='validation',scans=108,subjects=50,training=False,test_access=False,
        single_models={key:metrics(records[key]) for key in reference_ids},fusion_ranking=ranking,
        strongest_single_reference=dict(id=strongest_single_id,metrics=metrics(strongest_single),
            selection='highest Scan among6 predefined single-model reference configurations; same choice for all cases'),
        patient_folds=folds,oof_metrics=metrics(oof),oof_paired_to_race_full=paired(oof,reference),
        oof_paired_to_strongest_single=paired(oof,strongest_single),
        complementarity=dict(equal8_dice_correlation=float(np.corrcoef(race_d,w32_d)[0,1]),
            w32_better_scans=int((w32_d>race_d).sum()),race_better_scans=int((race_d>w32_d).sum()),
            equal_scans=int((race_d==w32_d).sum()),voxel_totals={k:sum(r[k] for r in overlaps) for k in overlaps[0] if k not in ('case_id','subject_id')}),
        model_forwards=dict(race=864,w32=864,total=1728,per_scan=16,new_during_decode=0),
        caveat='Reused validation and historically test-tuned checkpoints/decoder; patient folds assess stability, not independent confirmation',
        best_worsened_cases=worsened(records[ranking[0]['id']],reference),
        best_worsened_cases_vs_strongest_single=worsened(records[ranking[0]['id']],strongest_single))
    write(R/'RESULT.json',result)
    best=ranking[0]
    table='\n'.join(f"|{c['id']}|{c['metrics']['scan_dice']:.9f}|{c['metrics']['subject_dice']:.9f}|{c['metrics']['point_hits']}/108|" for c in ranking)
    (R/'RESULT.md').write_text(f"# RACE164/w32e171有限互补融合\n\n原108/50、单T1ce、native1mm去填充；12个事前冻结统一融合配置，两个模型各8视图，共16次前向/例。GT仅评分，无病例路由。\n\n|配置|Scan|Subject|Point|\n|---|---:|---:|---:|\n{table}\n\n最高{best['id']}；相对RACE完整配方差{best['paired_to_race_full']['scan_delta']:+.9f}，患者配对描述性95%区间{best['paired_to_race_full']['scan_delta_ci95']}。按患者5折选择合并Scan {result['oof_metrics']['scan_dice']:.9f}。历史test-tuned、复用验证选择，非独立确认；本阶段无270访问。\n\n[机器结果](RESULT.json) · [逐例结果](RAW_VALIDATION.json)\n")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--race-ready',default=str(A/'ATOMS_READY.json'));a=p.parse_args()
    try:main(a.cache,a.race_ready)
    except BaseException as error:
        import traceback
        write(R/f'ANALYSIS_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),model_forwards_added=0));raise
