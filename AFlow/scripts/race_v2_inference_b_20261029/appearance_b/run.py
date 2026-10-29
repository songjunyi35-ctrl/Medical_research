"""Four frozen cached-logit appearance-unary configurations, zero neural models."""
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import metrics,paired,read_atoms,worsened
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
from scripts.race_v2_inference_b_20261029.appearance_b.core import definitions,prepare,fuse,cpu_gate
from scripts.race_v2_inference_b_20261029.innovation_b.run import score_record,describe
P=R/'appearance_b';SCRATCH=OUT/'appearance_b';DEADLINE=1793271300

def guard_stage():guard();assert time.time()<DEADLINE,'18:55 last-family deadline'

@torch.inference_mode()
def gpu_gate(base_recipe):
    probes=[];torch.cuda.reset_peak_memory_stats()
    for shape in ((160,208,160),(160,208,168)):
        image=torch.ones(shape,device='cuda');image[:20]=2.
        logits=torch.full(shape,-6.,device='cuda');logits[:20]=5.
        cache=prepare([logits]*8,[logits]*8,image,base_recipe)
        torch.cuda.synchronize();begun=time.monotonic()
        for recipe in definitions(base_recipe):
            score,shift,trust,info=fuse(cache,recipe)
            assert score.shape==shape and bool(score.isfinite().all()) and bool((shift.abs()<=3+1e-6).all())
        torch.cuda.synchronize();probes.append(dict(canvas=list(shape),four_config_seconds=time.monotonic()-begun))
        del image,logits,cache,score,shift
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<38
    predicted=max(p['four_config_seconds'] for p in probes)*108*2+90
    assert time.time()+predicted<DEADLINE,'full108 four-config throughput cannot fit18:55'
    write(P/'GPU_GATE.json',dict(at=now(),status='passed',probes=probes,peak_reserved_gib=peak,
        patient_arrays_loaded=False,new_neural_forwards=0,predicted_complete108_seconds=predicted))
    exclusive(P/'BUDGET_FREEZE.json',dict(at=now(),deadline_unix=DEADLINE,predicted_seconds=predicted,
        GPU_gate_sha256=sha(P/'GPU_GATE.json'),execution_freeze_sha256=sha(P/'EXECUTION_FREEZE.json'),new_configurations=4,new_neural_forwards=0))
    torch.cuda.empty_cache()

@torch.inference_mode()
def main():
    verify();frozen=read(P/'EXECUTION_FREEZE.json')
    for source,digest in {**frozen['source_sha256'],**frozen['evidence_sha256']}.items():assert sha(ROOT/source)==digest,source
    lock=gpu_lock();numeric_mode();base_recipe=read(R/'innovation_b/SELECTED_RECIPE.json')['candidate'];gpu_gate(base_recipe)
    cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows();digest=sha(cache/'index.json')
    for actual,persisted in [(A/'ATOMS_READY.json',ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/ATOMS_READY.json'),(OUT/'ATOMS_READY.json',R/'ATOMS_READY.json')]:assert sha(actual)==sha(persisted)
    sources=[read_atoms(A/'ATOMS_READY.json',RACE_SHA,digest),read_atoms(OUT/'ATOMS_READY.json',CHECKPOINT_SHA,digest)]
    old=read(R/'RAW_VALIDATION.json')['per_configuration'];cov=read(R/'innovation_b/RAW_VALIDATION.json')['per_configuration'][base_recipe['id']]
    contra_recipe=read(R/'image_conflict_b/DEPLOYMENT_RECIPE.json')['candidate'];guided_recipe=read(R/'edge_input_b/DEPLOYMENT_RECIPE.json')['candidate']
    references={'race_equal8':old['race_equal8'],'old_best_fixed075':old['race0.75_logit_raw'],'old_selected_covariance':cov,
        'old_selected_contralateral':read(R/'image_conflict_b/RAW_VALIDATION.json')['per_configuration'][contra_recipe['id']],
        'old_selected_guided':read(R/'edge_input_b/RAW_VALIDATION.json')['per_configuration'][guided_recipe['id']]}
    exclusive(P/'DECODER_PREACCESS.json',dict(at=now(),scope='validation108 only, appearance fitting gets image/logits notGT',
        maximum_configurations=4,new_neural_forwards=0,budget_freeze_sha256=sha(P/'BUDGET_FREEZE.json'),test_access=False))
    recipes=definitions(base_recipe);assert recipes==read(P/'PLAN.json')['configurations'];records={r['id']:[] for r in recipes};begun=time.monotonic()
    for slot,row in enumerate(cohort):
        guard_stage();loaded=[]
        for source in sources:
            views=[]
            for view in range(8):
                atom=source[slot,view];path=Path(atom['path']);assert sha(path)==atom['sha256'];array=np.load(path,allow_pickle=False)
                assert array.dtype==np.float32 and list(array.shape)==atom['shape'];views.append(torch.from_numpy(array).cuda())
            loaded.append(views)
        image=torch.from_numpy(np.array(images[slot][slices(index['geometry'][row['case_id']])])).cuda()
        prepared=prepare(*loaded,image,base_recipe);predicted=[(recipe,*fuse(prepared,recipe)) for recipe in recipes]
        # All fitting and corrected scores are complete before full-crop GT enters scorer.
        target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
        for name,score in [('race_equal8',prepared['left_mean']),('old_best_fixed075',prepared['left_mean']*.75+prepared['right_mean']*.25),('old_selected_covariance',prepared['base_score'])]:
            check=score_record(row,slot,score,target);expected=references[name][slot]
            for key in ('tp','fp','fn','locator_dsc','automatic_point','automatic_point_hits_gt'):assert check[key]==expected[key],(name,slot,key)
        for recipe,score,shift,trust,info in predicted:
            record=score_record(row,slot,score,target);mask=(score>0).cpu().numpy();point=record['automatic_point']
            path=SCRATCH/'predictions'/recipe['id']/f'{slot:03}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.name+'.tmp')
            with temporary.open('xb') as handle:np.savez_compressed(handle,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
            assert not path.exists();os.replace(temporary,path)
            record.update(prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape),appearance_trust=trust,
                GMM=info,appearance_shift=describe(shift),appearance_shift_in_nonzero_brain=describe(shift,image!=0),
                appearance_shift_in_lesion=describe(shift,torch.from_numpy(target).cuda()),
                changed_threshold_class_voxels=int(((score>0)!=(prepared['base_score']>0)).sum()))
            records[recipe['id']].append(record)
        write(P/'HEARTBEAT.json',dict(at=now(),completed=slot+1,total=108,configurations=4,new_neural_forwards=0,seconds=time.monotonic()-begun))
        del prepared,predicted,loaded,views,image,score,shift
    write(P/'RAW_VALIDATION.json',dict(at=now(),per_configuration={**references,**records},new_neural_forwards=0,whole_crop_FN_FP=True,GT_only_scorer=True,test_access=False))
    ranking=[]
    for recipe in recipes:
        current=records[recipe['id']];entry=dict(recipe=recipe,metrics=summary_ci(current))
        for name,ref in references.items():entry['paired_to_'+name]=paired(current,ref);entry['all_worsened_vs_'+name]=worsened(current,ref)
        ranking.append(entry)
    ranking.sort(key=lambda r:(-r['metrics']['scan_dice'],-r['metrics']['subject_dice'],r['recipe']['id']))
    choices=list(records)+['old_best_fixed075','old_selected_covariance','old_selected_contralateral','old_selected_guided'];all_records={**references,**records}
    subjects=sorted({r['subject_id'] for r in cohort},key=lambda sid:hashlib.sha256(('race_v2_20261029:'+sid).encode()).hexdigest());fold={sid:i%5 for i,sid in enumerate(subjects)};folds=[];held=[None]*108
    for f in range(5):
        development=[i for i,r in enumerate(cohort) if fold[r['subject_id']]!=f];hold=[i for i,r in enumerate(cohort) if fold[r['subject_id']]==f]
        def key(name):
            m=metrics([all_records[name][i] for i in development]);return -m['scan_dice'],-m['subject_dice'],name
        selected=min(choices,key=key)
        for i in hold:held[i]=all_records[selected][i]
        folds.append(dict(fold=f,selected=selected,development_subjects=40,held_out_subjects=10,held_out_metrics=summary_ci([held[i] for i in hold])))
    bestall=min(choices,key=lambda name:(-metrics(all_records[name])['scan_dice'],-metrics(all_records[name])['subject_dice'],name))
    result=dict(at=now(),state='complete',family='image_appearance_unary',new_configurations=4,models=2,forwards_per_scan=16,
        EM_iterations_per_class=5,new_neural_forwards=0,new_training_epochs=0,test_access=False,ranking=ranking,best_new=ranking[0],
        best_including_frozen_references=dict(id=bestall,metrics=summary_ci(all_records[bestall])),patient_folds=folds,held_out_metrics=summary_ci(held),
        references={name:summary_ci(value) for name,value in references.items()},held_out_comparisons={name:paired(held,value) for name,value in references.items()},
        all_held_out_worsened={name:worsened(held,value) for name,value in references.items()},seconds=time.monotonic()-begun,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,interpretation='Same-image seeded appearance unary is heuristic, not independent Bayes evidence; reused108/checkpoint/anchor history means conditional folds, no independent confirmation. Last new family before final evaluation freeze.')
    write(P/'RESULT.json',result)
    write(P/'DEPLOYMENT_RECIPE.json',dict(at=now(),candidate=ranking[0]['recipe'],source_sha256=sha(ROOT/'scripts/race_v2_inference_b_20261029/appearance_b/core.py'),
        innovation_entry=dict(mode='image_appearance_unary',deployment_interface='deploy(logits_primary,logits_partner,image,recipe)'),
        model_order=['RACE primary','globally selected base SegResNet partner'],checkpoint_sha256=[RACE_SHA,CHECKPOINT_SHA],models=2,forwards_per_scan=16,EM_iterations_per_class=5,new_neural_forwards=0,test_access=False))
    table='\n'.join(f"|{r['recipe']['id']}|{r['metrics']['scan_dice']:.9f}|{r['metrics']['subject_dice']:.9f}|{r['metrics']['point_hits']}/108|" for r in ranking)
    (P/'RESULT.md').write_text(f'# 同图像外观unary：108完成\n\n4固定项，新增神经前向0；原RACE8+w32base8，部署2网络16forward+每类5EM。全crop GT只评分。\n\n|候选|Scan|Subject|Point|\n|---|---:|---:|---:|\n{table}\n\n完整区间/所有变差/患者条件5折/种子计数、回退与trust见RESULT/RAW。强置信错误可污染种子，不能称独立Bayesian后验。本family后不再新开网格，进入原80终态和最终至多2候选闭环。\n')
    write(P/'GPU_RELEASE.json',dict(at=now(),state='complete',GPU_released=True,new_neural_forwards=0,result_sha256=sha(P/'RESULT.json')))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(P/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_neural_forwards=0));raise
