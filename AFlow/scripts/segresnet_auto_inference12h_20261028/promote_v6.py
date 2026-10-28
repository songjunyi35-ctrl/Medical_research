"""Independent full-mask reconstruction and search-matrix audit of a global winner."""
import argparse
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028 import common as shared
shared.guard=guard
from scripts.segresnet_auto_inference12h_20261028.histograms_triton_v1 import histogram


def job_for(identifier):
    for queue in ['analysis_queue_v2','analysis_queue_aux_v1','cpu_queue_v2']:
        p=R/queue/(identifier+'.json')
        if p.exists():return read(p)
    raise KeyError(identifier)


def single_cut(counts,param):
    """Independent scalar float64 reconstruction of the selected global decoder."""
    bins=len(counts);descending=np.asarray(counts[::-1],np.int64);k=np.cumsum(descending,dtype=np.int64)
    p=np.linspace(1,0,bins).clip(1e-7,1-1e-7);z=np.log(p)-np.log1p(-p)
    q=1/(1+np.exp(-(z+param['bias'])/param['post_temperature']))
    prefix=np.cumsum(q*descending,dtype=np.float64);total=prefix[-1]
    index=int(np.argmax(2*prefix/(k+param.get('beta',1.)**2*total)))
    desired=int(np.rint(k[index]*param.get('volume_scale',1.)).clip(1,int(k[-1])))
    right=int(np.searchsorted(k,desired,side='left').clip(0,bins-1));left=max(right-1,0)
    chosen=left if abs(int(k[left])-desired)<=abs(int(k[right])-desired) else right
    return bins-1-chosen


def adaptive_single_cut(counts,param):
    p=np.linspace(1,0,len(counts)).clip(1e-7,1-1e-7);descending=np.asarray(counts[::-1],np.int64)
    mass=float(np.dot(p,descending));uncertainty=float(np.dot(p*(1-p),descending)/max(mass,1e-10))
    volume=int(counts[len(counts)//2:].sum());ratio=np.clip(max(volume,1)/20000.,.01,100.)
    u=np.clip((uncertainty+.05)/.25,.1,10.)
    temperature=np.clip(param['temperature_base']*ratio**param['volume_exponent']*u**param['uncertainty_exponent'],.25,12.)
    return single_cut(counts,dict(post_temperature=temperature,bias=param['bias'],beta=1.,volume_scale=param['volume_scale']))


def probability_for(cid,recipe,source_job,image):
    kind=recipe['kind']
    if kind=='group_mix':
        import importlib
        group=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+source_job['handler'])
        return group.probability(group.prepare(cid,source_job),recipe)
    if kind=='consensus':
        from scripts.segresnet_auto_inference12h_20261028 import consensus_v1 as consensus
        return consensus.probability(consensus.prepare(cid,source_job['core_source']),recipe)
    if kind=='local_bilateral_mrf':
        from scripts.segresnet_auto_inference12h_20261028.local_mrf_v1 import base_probability,refine
        return refine(base_probability(cid,source_job['core_source']),image,recipe)
    import importlib
    core=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+source_job['handler'])
    logits=[torch.from_numpy(np.load(TMP/'groups'/source_job['core_source']/'atoms'/f'flip{i}'/(cid+'.npy'))).cuda() for i in range(8)]
    return core.probability(core.prepare(logits),recipe)


def scalar_gate():
    from scripts.segresnet_auto_inference12h_20261028.calibration_v1 import parameters,all_cuts
    from scripts.segresnet_auto_inference12h_20261028.adaptive_calibration_v1 import parameters as adaptive_parameters,cuts_from_counts
    counts=np.random.default_rng(1).integers(0,20,256)
    cuts=all_cuts(counts);params=parameters()
    for i in [0,11,123,len(params)//2,len(params)-1]:assert single_cut(counts,params[i])==int(cuts[i])
    cuts,_=cuts_from_counts(counts);params=adaptive_parameters()
    for i in [0,11,123,len(params)//2,len(params)-1]:assert adaptive_single_cut(counts,params[i])==int(cuts[i])
    return dict(at=now(),status='passed',independent_scalar_vs_batched_cuts_exact=True,probes=10)


@torch.inference_mode()
def main(analysis,candidate,recipe_index=None,parameter_index=None):
    from scripts.followup_test270_common import summary_ci,paired,validate_rows
    numeric_mode();torch.cuda.set_per_process_memory_fraction(.2);guard();scalar=scalar_gate()
    search=read(R/'analyses'/analysis/'RANKING.json');best=search['best'];job=job_for(analysis);verify_job(job)
    source=job_for(job['source_analysis']) if job['handler'] in ['calibration_v1','calibration_extended_v1','calibration_parallel_extended_v1','calibration_consistency_v1','calibration_fine_v1','adaptive_calibration_v1'] else job
    for path,digest in source['source_sha256'].items():assert sha(ROOT/path)==digest,path
    if recipe_index is not None or parameter_index is not None:
        assert recipe_index is not None and parameter_index is not None
        assert job['handler'] in ['calibration_v1','calibration_fine_v1','calibration_extended_v1','calibration_parallel_extended_v1','calibration_consistency_v1']
        import copy,importlib
        calibration=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+job['handler'])
        best=copy.deepcopy(next(x for x in search['best_by_recipe'] if x['recipe_index']==recipe_index))
        best['parameter_index']=parameter_index;best['parameters']=calibration.parameters()[parameter_index]
        manifest=ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
        cohort=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
        values=np.array([np.load(TMP/'analyses'/analysis/'cases'/str(recipe_index)/(row['case_id']+'.npz'))['dice'][parameter_index] for row in cohort],np.float32)
        best['scan_dice']=float(values.mean(dtype=np.float64))
        subjects=sorted({r['subject_id'] for r in cohort})
        best['subject_dice']=float(np.mean([values[[i for i,r in enumerate(cohort) if r['subject_id']==subject]].mean(dtype=np.float64) for subject in subjects]))
        best['selection_note']='one global override from prespecified top-score beam; never per-case GT routing'
    folder=R/'candidates'/candidate;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'RESULT.json').exists(),'already audited'
    write(folder/'AUDIT_FREEZE.json',dict(at=now(),analysis=analysis,job=job,source_job=source,best=best,source_sha256=sha(Path(__file__)),scalar_gate=scalar,test_tuned=True))
    rows,geometry,images,labels=data();reference=read(OLD/'RAW_TEST270.json')['per_case'];previous={r['case_id']:r for r in reference}
    records=[];files=[];cuts=[];start=time.monotonic()
    RankSEG=None
    if job['handler']=='rankseg_v1':
        from scripts.segresnet_auto_inference12h_20261028.rankseg_v1 import decode,vendor
        RankSEG=vendor()
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];image=torch.from_numpy(np.array(images[slot][slices(geometry[cid])])).cuda()
        p=probability_for(cid,best['recipe'],source,image)
        if job['handler']=='rankseg_v1':
            mask=decode(p,best['parameters'],RankSEG);cut=None
        else:
            if 'parameters' in best:
                counts=histogram(p,torch.zeros_like(p,dtype=torch.bool)).sum(0,dtype=np.int64)
                if job['handler']=='adaptive_calibration_v1':cut=adaptive_single_cut(counts,best['parameters'])
                else:cut=single_cut(counts,best['parameters'])
            else:cut=int(np.floor(best['threshold']*65535))+1
            codes=(p.clamp(0,1)*65535+.5).long();mask=codes>=cut
        if best['recipe']['kind']=='local_bilateral_mrf':point=previous[cid]['automatic_point']
        elif best['recipe']['kind']=='consensus':
            from scripts.segresnet_auto_inference12h_20261028.consensus_v1 import prepare as consensus_prepare
            point_p=consensus_prepare(cid,source['core_source'])['p'].mean(0)
            point=list(map(int,np.unravel_index(int(point_p.argmax()),point_p.shape)))
        else:point=list(map(int,np.unravel_index(int(p.argmax()),p.shape)))
        mask=mask.cpu().numpy();target=np.array(labels[slot][slices(geometry[cid])]>0)
        tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
        ri=best['recipe_index']
        if job['handler'] in ['calibration_v1','calibration_extended_v1','calibration_parallel_extended_v1','calibration_consistency_v1','calibration_fine_v1','adaptive_calibration_v1']:
            expected=float(np.load(TMP/'analyses'/analysis/'cases'/str(ri)/(cid+'.npz'))['dice'][best['parameter_index']])
        else:
            pi=best['parameter_index'] if job['handler']=='rankseg_v1' else best['threshold_index']
            expected=float(np.load(TMP/'analyses'/analysis/'dice.npy',mmap_mode='r')[ri,pi,slot])
        assert score==expected,dict(case_id=cid,audit=score,search=expected)
        path=folder/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=cid,subject_id=row['subject_id'],locator_dsc=score,automatic_point=point,
              automatic_point_hits_gt=bool(target[tuple(point)]),tp=tp,fp=pred-tp,fn=positive-tp,source_slot=slot))
        files.append(dict(case_id=cid,path=str(path.relative_to(ROOT)),sha256=sha(path)));cuts.append(dict(case_id=cid,quantized_cut=cut))
        if slot%25==0:write(folder/'STATUS.json',dict(at=now(),completed=slot+1,total=270,seconds=time.monotonic()-start))
    validate_rows(records);metrics=summary_ci(records);metrics['point_hits']=sum(r['automatic_point_hits_gt'] for r in records)
    assert abs(metrics['scan_dice']-best['scan_dice'])<1e-12 and metrics['point_hits']==best['point_hits']
    contrast=paired(records,reference);contrast['bootstrap']='10000 patient-cluster paired seed20260918;95% descriptive,97.5% sensitivity only; not multiplicity correction';contrast['interpretation']='post-selection test-tuned descriptive; not independent confirmation'
    write(folder/'RAW_TEST270.json',dict(at=now(),per_case=records,predictions=files,test_tuned=True,current_test_retuning=True,training=False,model_forwards_added=0))
    write(folder/'RECIPE.json',dict(at=now(),checkpoint_sha256=CHECKPOINT_SHA,aggregation=best['recipe'],decoder=best.get('parameters',dict(threshold=best.get('threshold'))),
          decoder_kind=job['handler'],selected_on_test270=True,GT_input_to_inference=False,source_job=source,point_rule='pre-local-MRF five-view argmax' if best['recipe']['kind']=='local_bilateral_mrf' else ('mean8 probability argmax before consensus' if best['recipe']['kind']=='consensus' else 'aggregation probability argmax'),quantization_bins=None if job['handler']=='rankseg_v1' else 65536,realized_cuts=cuts))
    method='segresnet_w32_e171_auto12h_'+candidate+'_testtuned_20261028'
    cost=best['recipe'].get('neural_case_forwards',len(best['recipe']['views']))
    result=dict(at=now(),state='audited_candidate',method=method,metrics=metrics,comparison_to_previous_incumbent=contrast,
          checkpoint_sha256=CHECKPOINT_SHA,analysis=analysis,best_configuration=best,locator_forwards=cost,
          raw_sha256=sha(folder/'RAW_TEST270.json'),recipe_sha256=sha(folder/'RECIPE.json'),training=False,test_tuned=True,current_test_retuning=True,
          all270_masks_independently_verified=True,model_forwards_added=0,seconds=time.monotonic()-start)
    write(folder/'RESULT.json',result)
    (folder/'RESULT.md').write_text(f'''# 本轮已审计候选 `{candidate}`\n\n同一w32第171轮、单T1ce、原270/119、native评分。本轮直接test-tuned全局搜索后选择，非独立确认。\n\nScan **{metrics['scan_dice']:.9f}**，Subject **{metrics['subject_dice']:.9f}**，Point **{metrics['point_hits']}/270**。对研究前五镜像配方差{contrast['scan_delta']:+.9f}，患者配对描述性95%区间{contrast['scan_delta_ci95']}。选择后区间不证明独立泛化。\n\n候选由`{analysis}`完整270排名选出；重建全部270掩膜TP/FP/FN与搜索矩阵逐例精确一致，定位点命中亦一致。GT只进入评分，不参与预测/参数路由。新增模型训练0，审计复用已保存logits，不增加神经前向；部署平均前向{cost}次，具体配方/全部正负结果见原分析档。研究窗口尚未结题。\n\n[逐例结果](RAW_TEST270.json) · [配方](RECIPE.json) · [机器结果](RESULT.json)\n''')
    old=read(R/'INCUMBENT.json')
    if metrics['scan_dice']>old['scan_dice']:
        write(R/'INCUMBENT.json',dict(at=now(),state='interim_audited_test_tuned_incumbent',method=method,scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],raw_path=str((folder/'RAW_TEST270.json').relative_to(ROOT)),result_path=str((folder/'RESULT.json').relative_to(ROOT)),recipe_path=str((folder/'RECIPE.json').relative_to(ROOT)),test_tuned=True,research_continues=True))
    print(json.dumps(dict(method=method,metrics=metrics,delta=contrast['scan_delta']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--recipe-index',type=int);parser.add_argument('--parameter-index',type=int);args=parser.parse_args();main(args.analysis,args.candidate,args.recipe_index,args.parameter_index)
