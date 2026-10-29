"""Eight predeclared, cache-only GPU1 image-evidence decoders. No neural model."""
import numpy as np
import torch
import torch.nn.functional as F
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import metrics,paired,read_atoms,worsened
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
from scripts.race_v2_inference_b_20261029.image_conflict_b.core import definitions,prepare,fuse,cpu_gate
from scripts.race_v2_inference_b_20261029.innovation_b import run as old_run
from scripts.race_v2_inference_b_20261029.innovation_b.core import prepare as cov_prepare,fuse as cov_fuse
I=R/'image_conflict_b';SCRATCH=OUT/'image_conflict_b'
old_run.SCRATCH=SCRATCH
score_record=old_run.score_record
describe=old_run.describe

@torch.inference_mode()
def gpu_gate():
    generator=torch.Generator().manual_seed(20261029)
    shape=(24,17,19);image=torch.rand(shape,generator=generator)+.5
    left=[torch.randn(shape,generator=generator)-2 for _ in range(8)]
    right=[torch.randn(shape,generator=generator)-2 for _ in range(8)]
    cpu=prepare(left,right,image);gpu=prepare([z.cuda() for z in left],[z.cuda() for z in right],image.cuda())
    error=0.
    for recipe in definitions():
        expected=fuse(cpu,recipe)[0];actual=fuse(gpu,recipe)[0].cpu()
        error=max(error,float((expected-actual).abs().max()));assert torch.allclose(expected,actual,atol=2e-5,rtol=1e-5)
    del cpu,gpu;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for shape in ((160,208,160),(160,208,168)):
        image=torch.ones(shape,device='cuda');image[20:45]+=1
        left=[torch.full(shape,-5.,device='cuda')]*8;right=[torch.full(shape,-5.,device='cuda')]*8
        left[0]=left[0].clone();left[0][20:45]=20
        cache=prepare(left,right,image)
        for recipe in definitions():
            score,alpha,conflict=fuse(cache,recipe)
            assert score.shape==shape and bool(score.isfinite().all())
            assert bool(((alpha>=.5)&(alpha<=.95)).all())
        del image,left,right,cache,score,alpha,conflict
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<38
    write(I/'GPU_GATE.json',dict(at=now(),status='passed',dynamic_canvases=[[160,208,160],[160,208,168]],
        CPU_GPU_max_absolute_score_error=error,peak_reserved_gib=peak,patient_arrays_loaded=False,new_model_forwards=0))
    torch.cuda.empty_cache()

def diagnostics(cache,alpha,target):
    support=cache['support'];conflict=cache['conflict']>0
    gt=torch.from_numpy(target).to(alpha.device)
    boundary=(F.max_pool3d(gt.float()[None,None],3,1,1)[0,0]>0)&~(-F.max_pool3d(-gt.float()[None,None],3,1,1)[0,0]>0)
    output={}
    for name,region in [('effective_brain',cache['image']!=0),('conflict',conflict),('lesion',gt),('boundary',boundary)]:
        count=int(region.sum())
        output[name]=dict(count=count,trust=describe(cache['healthy'],region),conflict=describe(cache['conflict'],region),
            alpha_shift=describe(alpha-.75,region),fallback_fraction=float(((alpha==.75)&region).sum()/max(1,count)))
    output.update(plane_center_crop=cache['center'],plane_support_dice=cache['plane_dice'],
        missing_reference_fraction=float((~support).float().mean()))
    return output

@torch.inference_mode()
def main():
    verify();freeze=read(I/'EXECUTION_FREEZE.json')
    for path,digest in {**freeze['source_sha256'],**freeze['evidence_sha256']}.items():assert sha(ROOT/path)==digest,path
    assert definitions()==read(I/'PLAN.json')['configurations']
    lock=gpu_lock();numeric_mode();gpu_gate()
    cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows();index_sha=sha(cache/'index.json')
    ready_paths=[A/'ATOMS_READY.json',OUT/'ATOMS_READY.json']
    for actual,reference in zip(ready_paths,[ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/ATOMS_READY.json',R/'ATOMS_READY.json']):assert sha(actual)==sha(reference)
    sources=[read_atoms(ready_paths[0],RACE_SHA,index_sha),read_atoms(ready_paths[1],CHECKPOINT_SHA,index_sha)]
    old=read(R/'RAW_VALIDATION.json')['per_configuration'];cov=read(R/'innovation_b/RAW_VALIDATION.json')['per_configuration']
    cov_recipe=read(R/'innovation_b/SELECTED_RECIPE.json')['candidate']
    refs={'race_equal8':old['race_equal8'],'old_best_fixed075':old['race0.75_logit_raw'],
        'old_selected_covariance':cov[cov_recipe['id']]}
    exclusive(I/'DECODER_PREACCESS.json',dict(at=now(),scope='validation108 image/logits only before scoring',
        freeze_sha256=sha(I/'EXECUTION_FREEZE.json'),new_model_forwards=0,test_access=False))
    recipes=definitions();records={r['id']:[] for r in recipes};controls={key:[] for key in refs}
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for slot,row in enumerate(cohort):
        guard();loaded=[]
        for source in sources:
            views=[]
            for view in range(8):
                atom=source[slot,view];path=Path(atom['path']);assert sha(path)==atom['sha256']
                value=np.load(path,allow_pickle=False);assert value.dtype==np.float32 and list(value.shape)==atom['shape']
                assert tuple(value.shape)==tuple(index['geometry'][row['case_id']]['extent'])
                views.append(torch.from_numpy(value).cuda())
            loaded.append(views)
        image=torch.from_numpy(np.array(images[slot][slices(index['geometry'][row['case_id']])])).cuda()
        prepared=prepare(*loaded,image)
        # Predictor receives only image and logits. Full crop GT first enters after all scores finalized.
        predicted=[(recipe,*fuse(prepared,recipe)) for recipe in recipes]
        cc=cov_prepare(*loaded);cov_score=cov_fuse(cc,cov_recipe)[0]
        target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
        scores={'race_equal8':prepared['left_mean'],'old_best_fixed075':prepared['left_mean']*.75+prepared['right_mean']*.25,
            'old_selected_covariance':cov_score}
        for name,score in scores.items():
            record=score_record(row,slot,score,target);expected=refs[name][slot]
            for key in ('case_id','subject_id','tp','fp','fn','locator_dsc','automatic_point','automatic_point_hits_gt'):assert record[key]==expected[key],(name,row['case_id'],key)
            controls[name].append(record)
        for recipe,score,alpha,conflict in predicted:
            record=score_record(row,slot,score,target,alpha,conflict,recipe)
            record['regional_diagnostics']=diagnostics(prepared,alpha,target)
            records[recipe['id']].append(record)
        write(I/'CASE_JOURNAL'/f'{slot:03}.json',dict(at=now(),slot=slot,records={key:value[-1] for key,value in records.items()},three_controls_exact=True,new_model_forwards=0))
        write(I/'HEARTBEAT.json',dict(at=now(),stage='GPU_image_evidence_conflict_decoding',completed=slot+1,total=108,
            configurations=8,new_model_forwards=0,seconds=time.monotonic()-started))
        del prepared,predicted,loaded,views,image,cc,cov_score,scores,score,alpha,conflict
    all_records={**controls,**records}
    write(I/'RAW_VALIDATION.json',dict(at=now(),per_configuration=all_records,new_model_forwards=0,test_access=False,
        GT_used_for_scoring_only=True,all_three_controls_exact=True))
    ranking=[]
    for recipe in recipes:
        current=records[recipe['id']]
        entry=dict(recipe=recipe,metrics=summary_ci(current))
        for reference in controls:
            entry['paired_to_'+reference]=paired(current,controls[reference]);entry['all_worsened_vs_'+reference]=worsened(current,controls[reference])
        ranking.append(entry)
    ranking.sort(key=lambda v:(-v['metrics']['scan_dice'],-v['metrics']['subject_dice'],v['recipe']['id']))
    subjects=sorted({r['subject_id'] for r in cohort},key=lambda sid:hashlib.sha256(('race_v2_20261029:'+sid).encode()).hexdigest())
    subject_fold={sid:i%5 for i,sid in enumerate(subjects)};folds=[];held=[None]*108
    choices=list(records)+['old_best_fixed075','old_selected_covariance']
    for fold in range(5):
        development=[i for i,r in enumerate(cohort) if subject_fold[r['subject_id']]!=fold]
        hold=[i for i,r in enumerate(cohort) if subject_fold[r['subject_id']]==fold]
        def key(name):
            m=metrics([all_records[name][i] for i in development]);return -m['scan_dice'],-m['subject_dice'],name
        selected=min(choices,key=key)
        for slot in hold:held[slot]=all_records[selected][slot]
        folds.append(dict(fold=fold,selected=selected,development_subjects=40,held_out_subjects=10,held_out_metrics=summary_ci([held[i] for i in hold])))
    best_all=min(choices,key=lambda name:(-metrics(all_records[name])['scan_dice'],-metrics(all_records[name])['subject_dice'],name))
    result=dict(at=now(),state='complete',family='contralateral_image_evidence',new_configurations=8,models=2,
        per_case_deployment_forwards=16,new_model_forwards=0,test_access=False,ranking=ranking,best_new=ranking[0],
        best_including_frozen_references=dict(id=best_all,metrics=summary_ci(all_records[best_all])),
        references={name:summary_ci(value) for name,value in controls.items()},patient_folds=folds,held_out_metrics=summary_ci(held),
        held_out_comparisons={name:paired(held,value) for name,value in controls.items()},
        all_held_out_worsened={name:worsened(held,value) for name,value in controls.items()},
        seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        interpretation='Image positive intensity difference is not proven enhancement or lesion evidence. Anchor/checkpoints and two frozen references selected using historical108; folds conditional stability, not fresh OOF or independent confirmation.')
    write(I/'RESULT.json',result)
    write(I/'DEPLOYMENT_RECIPE.json',dict(at=now(),candidate=ranking[0]['recipe'],source_sha256=sha(ROOT/'scripts/race_v2_inference_b_20261029/image_conflict_b/core.py'),
        innovation_entry=dict(mode='contralateral_image_evidence',deployment_interface='deploy(logits_primary,logits_partner,image,recipe)'),
        selected_on='reused validation108 only',test_access=False,model_order=['RACE164 primary','w32e171 partner'],
        checkpoint_sha256=[RACE_SHA,CHECKPOINT_SHA],forwards_per_scan=16,new_neural_forwards=0))
    table='\n'.join(f"|{v['recipe']['id']}|{v['metrics']['scan_dice']:.9f}|{v['metrics']['subject_dice']:.9f}|{v['metrics']['point_hits']}/108|" for v in ranking)
    (I/'RESULT.md').write_text(f'# 对侧图像证据冲突裁决：108完成\n\n固定8配置，2模型/16前向；复用缓存新增神经前向0。单T1ce正向强度差不等同实证强化。\n\n|配置|Scan|Subject|Point|\n|---|---:|---:|---:|\n{table}\n\n8新+2冻结参考的五折条件稳定性Scan {result["held_out_metrics"]["scan_dice"]:.9f}。完整配对Scan/Subject区间、全部变差病例、冲突/有效脑/病灶/边界内trust与权重改变量在RESULT/RAW。GT仅最终评分。无270访问，无独立泛化确认。\n')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(I/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_model_forwards=0));raise
