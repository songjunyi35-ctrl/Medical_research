"""Two guided w32 inputs, eight mirrors each; four fixed fusion decoders."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.forward import load_model
from scripts.race_v2_inference_b_20261029.analyze import metrics,paired,read_atoms,worsened
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
from scripts.race_v2_inference_b_20261029.edge_input_b.core import transform
from scripts.race_v2_inference_b_20261029.innovation_b.core import prepare,fuse
from scripts.race_v2_inference_b_20261029.innovation_b import run as previous
E=R/'edge_input_b';SCRATCH=OUT/'edge_input_b';DEADLINE=1793271000
INPUTS=[dict(id='guided_beta025',beta=.25,kernel=5,epsilon=.04),dict(id='guided_beta050',beta=.5,kernel=5,epsilon=.04)]
previous.SCRATCH=SCRATCH;score_record=previous.score_record

def verify_edge():
    verify();frozen=read(E/'EXECUTION_FREEZE.json')
    for source,digest in {**frozen['source_sha256'],**frozen['evidence_sha256']}.items():assert sha(ROOT/source)==digest,source
    assert time.time()<DEADLINE,'18:50 bounded research deadline'

def guard_edge():guard();assert time.time()<DEADLINE,'18:50 bounded research deadline'

@torch.inference_mode()
def numeric_gate(model):
    probes=[]
    for canvas in ((160,208,160),(160,208,168)):
        for batch in (4,8):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            try:
                image=torch.zeros((batch,1,*canvas),device='cuda');image[:,:,16:-16,16:-16,16:-16]=.7
                image[:,:,40:60,40:65,40:70]=2.
                for beta in (.25,.5):
                    changed=transform(image[0,0],beta);assert changed.shape==canvas and bool((changed[image[0,0]==0]==0).all())
                    for axis in range(3):assert torch.allclose(transform(image[0,0].flip(axis),beta).flip(axis),changed,atol=2e-6)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(image).segmentation_logits
                assert output.shape==image.shape and bool(output.isfinite().all())
                torch.cuda.synchronize();begun=time.monotonic()
                for _ in range(3):
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(image).segmentation_logits
                torch.cuda.synchronize();speed=3*batch/(time.monotonic()-begun);peak=torch.cuda.max_memory_reserved()/2**30
                probes.append(dict(canvas=list(canvas),batch=batch,passed=peak<=38,cases_per_second=speed,peak_reserved_gib=peak))
                del image,output,changed
            except torch.cuda.OutOfMemoryError as error:
                probes.append(dict(canvas=list(canvas),batch=batch,passed=False,error=repr(error),patient_forwards=0))
                for key in ('image','output','changed'):
                    if key in locals():del locals()[key]
                torch.cuda.empty_cache()
            write(E/'NUMERIC_PROBE_JOURNAL.json',dict(at=now(),probes=probes,patient_forwards=0))
    viable=[r for r in probes if r['canvas']==list(CANVAS) and r['passed']];assert viable
    fastest=max(p['cases_per_second'] for p in viable);batch=min(p['batch'] for p in viable if p['cases_per_second']>=.97*fastest)
    for canvas,tail in [((160,208,160),108%batch),((160,208,168),270%batch)]:
        if tail:
            image=torch.zeros((tail,1,*canvas),device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(image).segmentation_logits
            assert output.shape==image.shape and bool(output.isfinite().all());del image,output
    tail_peak=torch.cuda.max_memory_reserved()/2**30;assert tail_peak<38
    speed=min(p['cases_per_second'] for p in viable if p['batch']==batch)
    predicted_seconds=1728/speed*2+180
    assert time.time()+predicted_seconds<DEADLINE,'complete budget cannot fit18:50'
    assert predicted_seconds<1200
    result=dict(at=now(),status='passed',probes=probes,batch=batch,validation_tail=108%batch,test_synthetic_tail=270%batch,
        strict_cpu_reload=True,tail_peak_reserved_gib=tail_peak,model_forwards=0,patient_arrays_loaded=False,full160_168_gate=True,
        predicted_full_two_input_forward_and_decode_seconds=predicted_seconds,hardware_only_batch=True)
    write(E/'NUMERIC_GATE.json',result)
    exclusive(E/'BUDGET_FREEZE.json',dict(at=now(),numeric_gate_sha256=sha(E/'NUMERIC_GATE.json'),batch=batch,
        planned_inputs=INPUTS,case_forwards_limit=1728,decode_configurations=4,deadline_unix=DEADLINE,
        predicted_seconds=predicted_seconds,execution_freeze_sha256=sha(E/'EXECUTION_FREEZE.json')))
    return batch

@torch.inference_mode()
def forward():
    verify_edge();lock=gpu_lock();numeric_mode();torch.cuda.set_per_process_memory_fraction(.94)
    model=load_model();batch=numeric_gate(model)
    cache=TMP/'shared_validation/native';index,images,_=data(cache);cohort=rows();guard_edge()
    exclusive(E/'MODEL_PREACCESS.json',dict(at=now(),checkpoint_sha256=CHECKPOINT_SHA,maximum_case_forwards=1728,
        inputs=INPUTS,views=list(range(8)),split='validation108',automatic_neural_retry=False,
        budget_freeze_sha256=sha(E/'BUDGET_FREEZE.json'),GT_not_predictor_input=True))
    total=0;begun=time.monotonic()
    for spec in INPUTS:
        entries=[];calls=0;started=time.monotonic()
        for first in range(0,108,batch):
            guard_edge();slots=list(range(first,min(first+batch,108)))
            image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
            for k in range(len(slots)):image[k,0]=transform(image[k,0],spec['beta'])
            for view in range(8):
                guard_edge();dims=tuple(a+2 for a in range(3) if view&(1<<a));changed=torch.flip(image,dims) if dims else image
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(changed).segmentation_logits
                output=output.float();output=torch.flip(output,dims) if dims else output;assert bool(output.isfinite().all());calls+=1
                for k,slot in enumerate(slots):
                    value=output[(k,0)+slices(index['geometry'][cohort[slot]['case_id']])].contiguous().cpu().numpy()
                    path=SCRATCH/spec['id']/'atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_name(path.name+'.tmp')
                    with temp.open('xb') as handle:np.save(handle,value,allow_pickle=False)
                    assert not path.exists();os.replace(temp,path)
                    entries.append(dict(slot=slot,view=view,path=str(path),shape=list(value.shape),sha256=sha(path)));total+=1
                assert total<=1728
                write(E/'HEARTBEAT.json',dict(at=now(),stage='w32_guided_input_eight_mirrors',input_spec=spec,
                    case_forwards=total,maximum=1728,seconds=time.monotonic()-begun))
            del image,changed,output
        assert len(entries)==864
        ready=dict(at=now(),status='complete',model_id='w32e171_guided',base_model_id='w32e171',
            atom_source='guided_input_w32',input_spec=spec,input_source_sha256=sha(ROOT/'scripts/race_v2_inference_b_20261029/edge_input_b/core.py'),
            checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,manifest_sha256=sha(MANIFEST),index_sha256=sha(cache/'index.json'),
            canvas=list(CANVAS),cache_path=str(cache),atoms=entries,case_forwards=864,batch=batch,batch_calls=calls,
            seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            new_training_epochs=0,GT_or_ID_input=False,transform_applied_once_before_all_mirrors=True)
        write(E/spec['id']/'ATOMS_READY.json',ready)
        for atom in entries:Path(atom['path']).chmod(0o444)
        for folder in (SCRATCH/spec['id']/'atoms').iterdir():folder.chmod(0o555)
        (SCRATCH/spec['id']/'atoms').chmod(0o555)
    assert total==1728
    write(E/'FORWARD_DONE.json',dict(at=now(),state='complete',case_forwards=total,seconds=time.monotonic()-begun))

@torch.inference_mode()
def analyze():
    verify_edge();assert read(E/'FORWARD_DONE.json')['case_forwards']==1728
    lock=gpu_lock();numeric_mode();cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows();digest=sha(cache/'index.json')
    assert sha(A/'ATOMS_READY.json')==sha(ROOT/'archive/research_records/2026-10-29_race_v2_inference_a/ATOMS_READY.json')
    race=read_atoms(A/'ATOMS_READY.json',RACE_SHA,digest);partners={s['id']:read_atoms(E/s['id']/'ATOMS_READY.json',CHECKPOINT_SHA,digest) for s in INPUTS}
    cov_recipe=read(R/'innovation_b/SELECTED_RECIPE.json')['candidate']
    old=read(R/'RAW_VALIDATION.json')['per_configuration'];cov=read(R/'innovation_b/RAW_VALIDATION.json')['per_configuration'][cov_recipe['id']]
    contra_recipe=read(R/'image_conflict_b/DEPLOYMENT_RECIPE.json')['candidate'];contra=read(R/'image_conflict_b/RAW_VALIDATION.json')['per_configuration'][contra_recipe['id']]
    references={'race_equal8':old['race_equal8'],'old_best_fixed075':old['race0.75_logit_raw'],'old_selected_covariance':cov,'old_selected_contralateral':contra}
    recipes=[dict(id=spec['id']+'_'+decoder,kind='guided_input_fusion',input_spec=spec,decoder=decoder,
        anchor=.75,models=2,forwards_per_scan=16,covariance_recipe=cov_recipe if decoder=='covariance' else None) for spec in INPUTS for decoder in ('fixed075','covariance')]
    records={r['id']:[] for r in recipes};begun=time.monotonic()
    exclusive(E/'DECODER_PREACCESS.json',dict(at=now(),new_configurations=4,network_forwards_by_decode=0,test_access=False))
    for slot,row in enumerate(cohort):
        guard_edge();loaded={}
        for key,source in {'race':race,**partners}.items():
            views=[]
            for view in range(8):
                atom=source[slot,view];path=Path(atom['path']);assert sha(path)==atom['sha256']
                value=np.load(path,allow_pickle=False);assert list(value.shape)==atom['shape'] and value.dtype==np.float32
                views.append(torch.from_numpy(value).cuda())
            loaded[key]=views
        predicted=[]
        for spec in INPUTS:
            cache2=prepare(loaded['race'],loaded[spec['id']])
            for recipe in [r for r in recipes if r['input_spec']==spec]:
                if recipe['decoder']=='fixed075':
                    score=.75*cache2['left_mean']+.25*cache2['right_mean'];alpha=torch.full_like(score,.75);gate=torch.ones_like(score)
                else:score,alpha,gate=fuse(cache2,cov_recipe)
                predicted.append((recipe,score,alpha,gate))
        target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
        for recipe,score,alpha,gate in predicted:records[recipe['id']].append(score_record(row,slot,score,target,alpha,gate,recipe))
        write(E/'DECODE_HEARTBEAT.json',dict(at=now(),completed=slot+1,total=108,configurations=4,network_forwards_added=0))
        del loaded,predicted,cache2,score,alpha,gate
    write(E/'RAW_VALIDATION.json',dict(at=now(),per_configuration={**references,**records},new_model_forwards=1728,
        network_forwards_by_decode=0,GT_only_scorer=True,test_access=False))
    ranking=[]
    for recipe in recipes:
        current=records[recipe['id']];entry=dict(recipe=recipe,metrics=summary_ci(current))
        for name,ref in references.items():entry['paired_to_'+name]=paired(current,ref);entry['all_worsened_vs_'+name]=worsened(current,ref)
        ranking.append(entry)
    ranking.sort(key=lambda r:(-r['metrics']['scan_dice'],-r['metrics']['subject_dice'],r['recipe']['id']))
    all_records={**references,**records};choices=list(records)+['old_best_fixed075','old_selected_covariance','old_selected_contralateral']
    subjects=sorted({r['subject_id'] for r in cohort},key=lambda sid:hashlib.sha256(('race_v2_20261029:'+sid).encode()).hexdigest());fold={sid:i%5 for i,sid in enumerate(subjects)};folds=[];held=[None]*108
    for f in range(5):
        development=[i for i,r in enumerate(cohort) if fold[r['subject_id']]!=f];hold=[i for i,r in enumerate(cohort) if fold[r['subject_id']]==f]
        def key(name):
            m=metrics([all_records[name][i] for i in development]);return -m['scan_dice'],-m['subject_dice'],name
        selected=min(choices,key=key)
        for i in hold:held[i]=all_records[selected][i]
        folds.append(dict(fold=f,selected=selected,development_subjects=40,held_out_subjects=10,held_out_metrics=summary_ci([held[i] for i in hold])))
    bestall=min(choices,key=lambda name:(-metrics(all_records[name])['scan_dice'],-metrics(all_records[name])['subject_dice'],name))
    result=dict(at=now(),state='complete',family='self_guided_edge_preserving_input',new_configurations=4,
        models=2,forwards_per_scan=16,new_model_forwards=1728,new_training_epochs=0,test_access=False,
        ranking=ranking,best_new=ranking[0],best_including_frozen_references=dict(id=bestall,metrics=summary_ci(all_records[bestall])),
        references={name:summary_ci(value) for name,value in references.items()},patient_folds=folds,held_out_metrics=summary_ci(held),
        held_out_comparisons={name:paired(held,value) for name,value in references.items()},
        all_held_out_worsened={name:worsened(held,value) for name,value in references.items()},decode_seconds=time.monotonic()-begun,
        interpretation='Input filtering is a new deterministic view, not a new trained model; all checkpoints/old decoders historically selected. Reused108 conditional folds and postselection descriptive intervals, no independent confirmation.')
    write(E/'RESULT.json',result)
    write(E/'DEPLOYMENT_RECIPE.json',dict(at=now(),candidate=ranking[0]['recipe'],source_sha256=sha(ROOT/'scripts/race_v2_inference_b_20261029/edge_input_b/core.py'),
        innovation_entry=dict(mode='guided_input_w32_fusion',input_transform_interface='transform(image,beta)',partner_model_id='w32e171_guided',base_model_id='w32e171',atom_source='guided_input_w32'),
        selected_on='reused108',models=2,forwards_per_scan=16,all_input_views_deployed='one selected input variant only, eight mirrors',test_access=False))
    table='\n'.join(f"|{r['recipe']['id']}|{r['metrics']['scan_dice']:.9f}|{r['metrics']['subject_dice']:.9f}|{r['metrics']['point_hits']}/108|" for r in ranking)
    (E/'RESULT.md').write_text(f'# 自引导边缘保留输入：108完成\n\n2输入各8view只缓存一次，共1728新增病例前向；4固定解码候选。部署只取一个输入，RACE8+w32guided8，2模型/16前向。无训练/270。\n\n|候选|Scan|Subject|Point|\n|---|---:|---:|---:|\n{table}\n\n完整患者簇配对区间、全部变差、条件五折见RESULT。图像滤波缺乏必然分割收益，保留MRI预处理原研究的负证据，不把GT事后oracle列入排行榜。\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('forward','analyze'));args=parser.parse_args()
    try:forward() if args.stage=='forward' else analyze()
    except BaseException as error:
        import traceback
        write(E/f'FAILURE_{time.time_ns()}.json',dict(at=now(),stage=args.stage,error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
