"""Finite GPU1 cached-logit study. This module never constructs a neural model."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import metrics,paired,read_atoms,worsened
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
from scripts.race_v2_inference_b_20261029.innovation_b.core import definitions,prepare,fuse,cpu_gate

I=R/'innovation_b';SCRATCH=OUT/'innovation_b'

def verify_innovation():
    verify();frozen=read(I/'EXECUTION_FREEZE.json')
    for source,digest in frozen['source_sha256'].items():assert sha(ROOT/source)==digest,source
    for source,digest in frozen['evidence_sha256'].items():assert sha(ROOT/source)==digest,source
    assert definitions()==read(I/'PLAN.json')['configurations']

def describe(tensor,where=None):
    value=tensor if where is None else tensor[where]
    if value.numel()==0:return dict(count=0)
    return dict(count=value.numel(),mean=float(value.mean()),minimum=float(value.min()),maximum=float(value.max()),
        quantiles=torch.quantile(value.float(),torch.tensor([.05,.5,.95],device=value.device)).cpu().tolist())

def score_record(row,slot,score,target,alpha=None,gate=None,recipe=None):
    mask=(score>0).cpu().numpy();point=list(map(int,np.unravel_index(int(score.argmax()),mask.shape)))
    tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
    record=dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,
        locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=pred-tp,fn=positive-tp,
        automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]))
    if recipe is not None:
        path=SCRATCH/'predictions'/recipe['id']/f'{slot:03}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            old=np.load(path,allow_pickle=False);previous=np.unpackbits(old['mask_bits'])[:mask.size].reshape(mask.shape)
            assert np.array_equal(previous,mask) and old['point'].tolist()==point
        else:
            temporary=path.with_name(path.name+'.tmp')
            with temporary.open('xb') as handle:np.savez_compressed(handle,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
            os.replace(temporary,path)
        record.update(prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape),
            alpha=describe(alpha),conflict_gate=describe(gate),alpha_at_point=float(alpha[tuple(point)]),
            changed_weight_over005_fraction=float(((alpha-.75).abs()>.05).float().mean()))
    return record

@torch.inference_mode()
def gpu_gate():
    generator=torch.Generator().manual_seed(20261029)
    cpu_left=[torch.randn((7,9,11),generator=generator)*.3+.2 for _ in range(8)]
    cpu_right=[torch.randn((7,9,11),generator=generator)*.4-.1 for _ in range(8)]
    cpu=prepare(cpu_left,cpu_right);gpu=prepare([z.cuda() for z in cpu_left],[z.cuda() for z in cpu_right]);maximum=0.
    for recipe in definitions():
        expected=fuse(cpu,recipe)[0];actual=fuse(gpu,recipe)[0].cpu()
        maximum=max(maximum,float((expected-actual).abs().max()));assert torch.allclose(expected,actual,atol=1e-5,rtol=1e-5)
        assert torch.equal(expected>0,actual>0)
    del cpu,gpu;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    base=torch.linspace(-2,2,int(np.prod(CANVAS)),device='cuda').reshape(CANVAS)
    left=[base+.1*(i-3.5)*torch.sin(base*(i+1)) for i in range(8)]
    right=[base*.8-.1+.13*(i-3.5)*torch.cos(base*(i+1)) for i in range(8)]
    full=prepare(left,right)
    for recipe in definitions():
        vr,vw,cov=full['moments'];assert bool((vr*vw-cov.square()>=-1e-7).all())
        score,alpha,gate=fuse(full,recipe)
        assert score.shape==CANVAS and bool(score.isfinite().all())
        assert bool(((alpha>=.5)&(alpha<=.95)).all()) and bool(((gate>=0)&(gate<=1)).all())
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<38
    write(I/'GPU_GATE.json',dict(at=now(),status='passed',all12_full_native_finite=True,
        canvas=list(CANVAS),CPU_GPU_small_score_max_absolute_error=maximum,CPU_GPU_small_masks_exact=True,
        peak_reserved_gib=peak,patient_arrays_loaded=False,neural_models_constructed=0,model_forwards=0))
    del base,left,right,full,score,alpha,gate;torch.cuda.empty_cache()

@torch.inference_mode()
def main():
    verify_innovation();lock=gpu_lock();numeric_mode();torch.cuda.set_per_process_memory_fraction(.9)
    gpu_gate()
    cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows();index_sha=sha(cache/'index.json')
    sources=[read_atoms(A/'ATOMS_READY.json',RACE_SHA,index_sha),read_atoms(OUT/'ATOMS_READY.json',CHECKPOINT_SHA,index_sha)]
    old=read(R/'RAW_VALIDATION.json')['per_configuration'];references={
        'race_equal8':old['race_equal8'],'w32_equal8':old['w32_equal8'],'old_best_fixed075':old['race0.75_logit_raw']}
    exclusive(I/'DECODER_PREACCESS.json',dict(at=now(),scope='validation108 cached-logit decoding only',
        split='validation',models=2,maximum_deployment_forwards=16,model_forwards_added=0,
        atom_ready_sha256=[sha(A/'ATOMS_READY.json'),sha(OUT/'ATOMS_READY.json')],
        freeze_sha256=sha(I/'EXECUTION_FREEZE.json'),test_access=False))
    recipes=definitions();records={recipe['id']:[] for recipe in recipes};controls={key:[] for key in references}
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();baseline_checks=[]
    for slot,row in enumerate(cohort):
        guard();loaded=[]
        for source in sources:
            views=[]
            for view in range(8):
                entry=source[slot,view];path=Path(entry['path']);assert sha(path)==entry['sha256']
                value=np.load(path,allow_pickle=False);assert value.dtype==np.float32 and list(value.shape)==entry['shape']
                assert tuple(value.shape)==tuple(index['geometry'][row['case_id']]['extent'])
                views.append(torch.from_numpy(value).cuda())
            loaded.append(views)
        prepared=prepare(*loaded)
        # Predictions are finalized before this scorer receives the label crop.
        predicted=[(recipe,*fuse(prepared,recipe)) for recipe in recipes]
        target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
        control_scores={'race_equal8':prepared['left_mean'],'w32_equal8':prepared['right_mean'],
            'old_best_fixed075':prepared['left_mean']*.75+prepared['right_mean']*.25}
        for name,score in control_scores.items():
            record=score_record(row,slot,score,target);expected=references[name][slot]
            for key in ('case_id','subject_id','tp','fp','fn','locator_dsc','automatic_point','automatic_point_hits_gt'):
                assert record[key]==expected[key],(name,row['case_id'],key,record[key],expected[key])
            controls[name].append(record)
        baseline_checks.append(dict(case_id=row['case_id'],three_references_integer_counts_points_exact=True))
        for recipe,score,alpha,gate in predicted:
            records[recipe['id']].append(score_record(row,slot,score,target,alpha,gate,recipe))
        write(I/'CASE_JOURNAL'/f'{slot:03}.json',dict(at=now(),slot=slot,
            records={key:value[-1] for key,value in records.items()},three_controls_exact=True,model_forwards_added=0))
        write(I/'HEARTBEAT.json',dict(at=now(),stage='useful_GPU_covariance_and_fixed_decoding',completed=slot+1,total=108,
            configurations=12,model_forwards_added=0,seconds=time.monotonic()-started))
        del prepared,predicted,loaded,views,score,alpha,gate
    write(I/'BASELINE_REPRODUCED.json',dict(at=now(),status='passed',scans=108,controls=baseline_checks,
        original_race_equal8_and_w32_equal8_and_fixed075_exact=True))
    write(I/'RAW_VALIDATION.json',dict(at=now(),per_configuration={**controls,**records},training=False,
        new_model_forwards=0,test_access=False,predictions_use_GT_or_case_lookup=False))
    ranking=[]
    for recipe in recipes:
        current=records[recipe['id']]
        ranking.append(dict(recipe=recipe,metrics=summary_ci(current),
            paired_to_race_equal8=paired(current,controls['race_equal8']),
            paired_to_strongest_frozen_single=paired(current,controls['race_equal8']),
            paired_to_old_best_fusion=paired(current,controls['old_best_fixed075']),
            all_worsened_vs_equal8=worsened(current,controls['race_equal8']),
            all_worsened_vs_old_best_fusion=worsened(current,controls['old_best_fixed075'])))
    ranking.sort(key=lambda entry:(-entry['metrics']['scan_dice'],-entry['metrics']['subject_dice'],entry['recipe']['id']))
    subjects=sorted({r['subject_id'] for r in cohort},key=lambda sid:hashlib.sha256(('race_v2_20261029:'+sid).encode()).hexdigest())
    subject_fold={sid:i%5 for i,sid in enumerate(subjects)};folds=[];held_out_records=[None]*108
    for fold in range(5):
        development=[i for i,row in enumerate(cohort) if subject_fold[row['subject_id']]!=fold]
        held_out=[i for i,row in enumerate(cohort) if subject_fold[row['subject_id']]==fold]
        def key(recipe):
            value=metrics([records[recipe['id']][i] for i in development])
            return -value['scan_dice'],-value['subject_dice'],recipe['id']
        selected=min(recipes,key=key)
        for slot in held_out:held_out_records[slot]=records[selected['id']][slot]
        folds.append(dict(fold=fold,selected=selected,development_subjects=40,held_out_subjects=10,
            held_out_metrics=summary_ci([held_out_records[i] for i in held_out])))
    result=dict(at=now(),state='complete',run_id='innovation_b_paired_covariance_20261029',models=2,
        new_configurations=12,model_forwards_added=0,per_case_deployment_forwards=16,test_access=False,
        references={name:summary_ci(rows) for name,rows in controls.items()},ranking=ranking,best=ranking[0],
        patient_folds=folds,held_out_metrics=summary_ci(held_out_records),
        held_out_paired_to_equal8=paired(held_out_records,controls['race_equal8']),
        held_out_paired_to_old_best_fusion=paired(held_out_records,controls['old_best_fixed075']),
        all_held_out_worsened_vs_equal8=worsened(held_out_records,controls['race_equal8']),
        seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        baseline_integer_counts_points_exact=True,
        interpretation='Paired deterministic mirror perturbation covariance is a heuristic, not calibrated error covariance; checkpoints and anchor historically selected, reused108 folds are stability only, not independent confirmation')
    write(I/'RESULT.json',result)
    write(I/'SELECTED_RECIPE.json',dict(at=now(),candidate=ranking[0]['recipe'],
        selected_on='reused validation108 only',new_model_forwards=0,test_access=False,
        source_sha256=sha(ROOT/'scripts/race_v2_inference_b_20261029/innovation_b/core.py'),
        deployment_interface='deploy(logits_primary,logits_partner,image,recipe)',
        dynamic_input='two aligned8-view crops; same shape as image; no labels or case lookup',
        model_order=['RACE164 primary','w32e171 partner'],checkpoint_sha256=[RACE_SHA,CHECKPOINT_SHA],
        condition='anchor.75 was selected on full108 before this study; five folds are conditional stability, not fresh OOF'))
    table='\n'.join(f"|{r['recipe']['id']}|{r['metrics']['scan_dice']:.9f}|{r['metrics']['subject_dice']:.9f}|{r['metrics']['point_hits']}/108|" for r in ranking)
    best=ranking[0]
    (I/'RESULT.md').write_text(f"# 配对镜像协方差收缩融合：108完成\n\n原108/50、单T1ce、native1mm去填充；12事前冻结连续函数、2模型/16前向成本，缓存解码新增网络0，无270访问。\n\n|配置|Scan|Subject|Point|\n|---|---:|---:|---:|\n{table}\n\n最高{best['recipe']['id']}，对RACE equal8差{best['paired_to_race_equal8']['scan_delta']:+.9f}、患者95%区间{best['paired_to_race_equal8']['scan_delta_ci95']}；对旧最强固定融合差{best['paired_to_old_best_fusion']['scan_delta']:+.9f}、区间{best['paired_to_old_best_fusion']['scan_delta_ci95']}。五折仅四折选择后合并Scan{result['held_out_metrics']['scan_dice']:.9f}、Subject{result['held_out_metrics']['subject_dice']:.9f}、Point{result['held_out_metrics']['point_hits']}/108。全部变差病例在机器结果。\n\n扰动协方差不是GT误差协方差，不能据此声称最优Dice；历史权重/锚点已用复用数据选择，分折不构成独立泛化确认。\n")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-gate',action='store_true');args=parser.parse_args()
    if args.cpu_gate:write(I/'CPU_GATE.json',dict(at=now(),**cpu_gate()))
    else:
        try:main()
        except BaseException as error:
            import traceback
            write(I/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_model_forwards=0));raise
