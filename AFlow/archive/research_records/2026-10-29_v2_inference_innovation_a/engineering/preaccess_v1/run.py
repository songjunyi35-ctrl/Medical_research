"""Real GPU0 local-dependency decoding; one cache read,16 fixed recipes."""
import fcntl,subprocess,time
import numpy as np
import torch
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_innovation_a_20261029.algorithm import prepare,refine,numerical_gate
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

def preverify():
    guard();f=read(R/'PREPARATION_FREEZE.json')
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(R/'PLAN.json')==f['plan_sha256']
    return f

def synthetic_gate(recipes):
    gate=numerical_gate('cuda')
    generator=torch.Generator().manual_seed(32);small=[torch.randn(19,21,23,generator=generator) for _ in range(8)]
    cpu=prepare(small);gpu=prepare([x.cuda() for x in small]);differences=[];mask_differences=[]
    for recipe in recipes:
        a,_=refine(cpu,recipe);b,_=refine(gpu,recipe);b=b.cpu()
        differences.append(float((a-b).abs().max()));mask_differences.append(int(((a>0)!=(b>0)).sum()))
    assert max(differences)<3e-5
    del cpu,gpu,small
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(33)
    started=time.monotonic();native=prepare([torch.randn(160,208,160,device='cuda') for _ in range(8)])
    for recipe in recipes:value,_=refine(native,recipe)
    torch.cuda.synchronize();seconds=time.monotonic()-started;peak=torch.cuda.max_memory_reserved()/2**30
    assert peak<=16
    estimated=seconds*108*1.5+180
    assert estimated<=2700,dict(full_case_seconds=seconds,estimated_wall_seconds=estimated)
    del native,value;torch.cuda.empty_cache()
    return dict(at=now(),**gate,CPU_GPU_max_logit_error=max(differences),CPU_GPU_mask_differences=mask_differences,
        full_native_canvas=[160,208,160],full_native16_seconds=seconds,peak_reserved_gib=peak,
        estimated108_wall_seconds=estimated,clinical_arrays_loaded=False,neural_forwards=0)

def save_case(folder,mask,point,base_point):
    folder.parent.mkdir(parents=True,exist_ok=True)
    if folder.exists():
        previous=np.load(folder,allow_pickle=False);old=np.unpackbits(previous['mask_bits'])[:mask.size].reshape(mask.shape).astype(bool)
        assert np.array_equal(old,mask) and previous['point'].tolist()==point
    else:
        temp=folder.with_name(folder.name+'.tmp')
        with temp.open('xb') as f:np.savez_compressed(f,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point),base_point=np.array(base_point))
        os.replace(temp,folder);folder.chmod(0o444)
    return sha(folder)

def case_record(row,slot,identifier,mask,point,base_point,target,diagnostic):
    tp=int((mask&target).sum());prediction=int(mask.sum());positive=int(target.sum());den=prediction+positive
    path=TMP/'predictions'/identifier/(row['case_id']+'.npz');digest=save_case(path,mask,point,base_point)
    return dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,configuration_id=identifier,
        locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=prediction-tp,fn=positive-tp,
        automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),
        base_automatic_point=base_point,base_automatic_point_hits_gt=bool(target[tuple(base_point)]),
        prediction_path=str(path),prediction_sha256=digest,prediction_shape=list(mask.shape),diagnostic=diagnostic)

def analyze(records,recipes,gpu_seconds,wall_seconds,peak):
    plan=read(R/'PLAN.json');ids=['equal8_logit_t05']+[r['id'] for r in recipes];base=records[ids[0]]
    ranking=[]
    for order,identifier in enumerate(ids):
        values=records[identifier];metrics=summary_ci(values)
        ranking.append(dict(id=identifier,order=order,metrics=metrics,paired_vs_equal8=paired(values,base),
            point_refined=metrics['point_hits'],point_base=sum(r['base_automatic_point_hits_gt'] for r in values)))
    ranking.sort(key=lambda r:(-r['metrics']['scan_dice'],-r['metrics']['subject_dice'],-r['point_refined'],r['order']))
    folds=[];heldout=[]
    for fold in range(5):
        development=[i for i,row in enumerate(rows()) if plan['folds'][row['subject_id']]!=fold]
        held=[i for i,row in enumerate(rows()) if plan['folds'][row['subject_id']]==fold]
        winner=min(ids,key=lambda key:(-summary_ci([records[key][i] for i in development])['scan_dice'],
            -summary_ci([records[key][i] for i in development])['subject_dice'],ids.index(key)))
        chosen=[records[winner][i] for i in held];heldout.extend(chosen)
        folds.append(dict(fold=fold,selected_id=winner,heldout=summary_ci(chosen),reference=summary_ci([base[i] for i in held])))
    best=ranking[0];best_records=records[best['id']]
    worse=sorted([dict(case_id=a['case_id'],subject_id=a['subject_id'],candidate=a['locator_dsc'],reference=b['locator_dsc'],delta=a['locator_dsc']-b['locator_dsc']) for a,b in zip(best_records,base) if a['locator_dsc']<b['locator_dsc']],key=lambda r:r['delta'])
    selected=(dict(id='equal8_logit_t05',kind='reference_equal8',neural_case_forwards=8) if best['id']==ids[0] else next(r for r in recipes if r['id']==best['id']))
    write(R/'RAW_VALIDATION108.json',dict(at=now(),per_configuration=records,recipes=recipes,new_network_forwards=0,new270_scoring=False))
    result=dict(at=now(),state='complete',best=best,ranking=ranking,patient_fivefold=folds,
        heldout_selected_metrics=summary_ci(heldout),heldout_paired_vs_equal8=paired(sorted(heldout,key=lambda r:r['source_slot']),base),
        worsened_cases=worse,new_configurations=16,reference_configurations=1,checkpoint_sha256=CHECKPOINT_SHA,
        new_network_forwards=0,retained_neural_views_per_case=8,model_count=1,training_updates=0,
        gpu_algorithm_event_seconds=gpu_seconds,wall_seconds=wall_seconds,peak_reserved_gib=peak,
        point_primary='refined score argmax',point_auxiliary='equal8 argmax',GT_or_ID_in_deploy=False,
        reused_validation_after_old32=True,test_tuned_history=True,independent_confirmation=False,
        source_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),raw_sha256=sha(R/'RAW_VALIDATION108.json'),
        new270_scoring=False,root_final_policy_must_be_updated_before270=True)
    write(R/'RESULT.json',result);write(R/'SELECTED_RECIPE.json',dict(at=now(),recipe=selected,
        selection_on='validation108',source_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),result_sha256=sha(R/'RESULT.json'),
        deploy_callable='scripts.race_v2_inference_innovation_a_20261029.algorithm.deploy',new270_authorization=False))
    (R/'RESULT.md').write_text(f'''# RACE八视图局部依赖融合108验证结果

已完成16个事前冻结新配置及equal8参照；最佳`{best['id']}`，Scan **{best['metrics']['scan_dice']:.9f}**、Subject **{best['metrics']['subject_dice']:.9f}**、Point_refined **{best['point_refined']}/108**、Point_base **{best['point_base']}/108**。对equal8 Scan差{best['paired_vs_equal8']['scan_delta']:+.9f}，患者配对描述性95%区间{best['paired_vs_equal8']['scan_delta_ci95']}。

患者5折四折选择/留出汇总Scan {result['heldout_selected_metrics']['scan_dice']:.9f}。全部继承既有test-tuned历史、复用验证选择，非独立确认；全部配置和变差病例保留，未按270选择。

本轮网络前向0、训练0；复用RACE164既有8视图。GPU0算法CUDA event区间累计{gpu_seconds:.2f}秒，墙钟{wall_seconds:.2f}秒，reserved峰值{peak:.3f}GiB；event区间可含dispatch间隙，不冒充GPU忙碌率。部署仍单模型8前向，局部矩阵与上采样另增计算。

本轮新增270评分0。最终270必须等总控更新最终冻结政策；新训练best只移植已选一个配方，不重新16搜索。

[完整逐例](RAW_VALIDATION108.json) · [排名/区间/分折](RESULT.json) · [精确计划](PLAN.json) · [选择](SELECTED_RECIPE.json)
''')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';TMP.mkdir(parents=True,exist_ok=True);pre=preverify()
    lock_path=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    write(R/'GPU_LOCK.json',dict(at=now(),worker=1154121,gpu=0,pid=os.getpid(),lock_path=str(lock_path),
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()))
    recipes=read(R/'PLAN.json')['recipes'];gate=synthetic_gate(recipes);write(R/'GPU_GATE.json',gate)
    write(R/'EXECUTION_FREEZE.json',dict(pre,at=now(),preparation_freeze_sha256=sha(R/'PREPARATION_FREEZE.json'),GPU_gate_sha256=sha(R/'GPU_GATE.json')))
    verify();index=read(CACHE/'index.json');canvas=tuple(index['canvas']);cohort=rows()
    assert index['case_ids']==[r['case_id'] for r in cohort] and len(cohort)==108
    ready=read(BASE/'CACHE_READY.json');assert sha(CACHE/'index.json')==ready['index_sha256']
    assert sha(CACHE/'images.f32')==ready['payload_sha256']['images.f32']
    assert sha(CACHE/'labels.u8')==ready['payload_sha256']['labels.u8']
    atoms=read(BASE/'ATOMS_READY.json');lookup={(a['slot'],a['view']):a for a in atoms['atoms']}
    assert len(lookup)==864 and atoms['checkpoint_sha256']==CHECKPOINT_SHA
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*canvas));labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*canvas))
    reference=read(BASE/'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05'];assert len(reference)==108
    write(R/'VALIDATION_DECODE_PREACCESS.json',dict(at=now(),source_atoms_manifest_sha256=sha(BASE/'ATOMS_READY.json'),
        source_cache_index_sha256=sha(CACHE/'index.json'),split='validation108',new_configurations=16,new_network_forwards=0,new270_scoring=False))
    started=time.monotonic();gpu_seconds=0.;records={identifier:[] for identifier in ['equal8_logit_t05']+[r['id'] for r in recipes]};baseline_audits=[]
    torch.cuda.reset_peak_memory_stats()
    for slot,row in enumerate(cohort):
        verify();assert time.monotonic()-started<2700
        sl=slices(index['geometry'][row['case_id']]);target=np.array(labels[slot][sl]>0);logits=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];z=np.load(atom['path'],allow_pickle=False)
            assert list(z.shape)==atom['shape']==index['geometry'][row['case_id']]['extent'];logits.append(torch.from_numpy(z).cuda())
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record();cache=prepare(logits);last.record();last.synchronize();gpu_seconds+=first.elapsed_time(last)/1000
        base_point=list(map(int,np.unravel_index(int(cache['base'].argmax()),target.shape)));mask=(cache['base']>0).cpu().numpy()
        baseline=case_record(row,slot,'equal8_logit_t05',mask,base_point,base_point,target,{})
        expected=reference[slot];assert baseline['case_id']==expected['case_id']
        assert all(baseline[k]==expected[k] for k in ('tp','fp','fn','locator_dsc','automatic_point','automatic_point_hits_gt'))
        records['equal8_logit_t05'].append(baseline);baseline_audits.append(dict(case_id=row['case_id'],all_counts_dice_point_exact=True))
        for recipe in recipes:
            first.record();value,diagnostic=refine(cache,recipe);point=list(map(int,np.unravel_index(int(value.argmax()),target.shape)));last.record();last.synchronize();gpu_seconds+=first.elapsed_time(last)/1000
            records[recipe['id']].append(case_record(row,slot,recipe['id'],(value>0).cpu().numpy(),point,base_point,target,diagnostic))
        write(R/'HEARTBEAT.json',dict(at=now(),stage='GPU0_local_dependency_validation',completed=slot+1,total=108,
            new_configurations=16,new_network_forwards=0,gpu_algorithm_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-started,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(TMP/'case_records'/f'{slot:03}.json',{key:values[-1] for key,values in records.items()})
        del cache,logits,value
    peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=16
    write(R/'BASELINE_EXACT_AUDIT.json',dict(at=now(),status='passed',cases=baseline_audits,all108_counts_dice_point_exact=True))
    analyze(records,recipes,gpu_seconds,time.monotonic()-started,peak)
    write(R/'DONE.json',dict(at=now(),state='complete',new_configurations=16,validation_scans=108,new_network_forwards=0,new270_scoring=False))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(R/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_network_forwards=0));raise
