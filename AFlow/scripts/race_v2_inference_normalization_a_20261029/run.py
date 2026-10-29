"""Bounded GPU0 forwarding followed by cached decoding and full mask audit."""
import fcntl,os,subprocess,time
import numpy as np
import torch
from scripts.race_v2_inference_normalization_a_20261029.common import *
from scripts.race_v2_inference_normalization_a_20261029.algorithm import coefficients,transform,forward_view,decode
from scripts.race_v2_inference_a_20261029.predictor import load_model,forward_view as original_forward
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks

def save_atom(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as out:np.save(out,value,allow_pickle=False)
    os.replace(temp,path);path.chmod(0o444)
    return dict(path=str(path),shape=list(value.shape),sha256=sha(path),dtype=str(value.dtype))

def synthetic(model,canvas):
    torch.cuda.reset_peak_memory_stats();torch.manual_seed(179)
    x=torch.randn(8,1,*canvas,device='cuda');support=x!=0
    g=dict(canvas_offset=[0,0,0],extent=list(canvas),reference_enabled=True,
        reflection=[[-1,0,0,canvas[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    started=time.monotonic()
    a=original_forward(model,x,[g]*8,7,canvas);b=forward_view(model,x,support,[g]*8,7,canvas)
    assert all(np.array_equal(u,v) for u,v in zip(a,b))
    changed=transform(x,support,dict(gain=1.075,offset=.075))
    full=forward_view(model,changed,support,[g]*8,7,canvas)
    tail=forward_view(model,changed[:4],support[:4],[g]*4,1,canvas)
    singleton=forward_view(model,changed[:1],support[:1],[g],2,canvas)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    return dict(at=now(),status='passed',native_canvas=list(canvas),batch=8,tail=4,singleton=1,
        identity_original_forward_bit_exact=True,normalized_full_and_tail_finite=True,
        original_support_separately_mirrored=True,seconds=time.monotonic()-started,
        peak_reserved_gib=peak,patient_forwards=0,checkpoint_CPU_strict_reload=True)

def prepare_coefficients(images,index,cohort,lookup,groups):
    result=[];entries=[]
    for slot,row in enumerate(cohort):
        g=index['geometry'][row['case_id']];image=np.asarray(images[slot][slices(g)])
        atom=lookup[slot,0];assert sha(atom['path'])==atom['sha256']
        seed=np.load(atom['path'],allow_pickle=False);assert list(seed.shape)==g['extent']
        coefficients_by_group={};canonical={};seen={}
        for group in groups:
            value=coefficients(image,seed,group);key=(value['gain'],value['offset'])
            alias='identity' if key==(1.,0.) else seen.get(key,group['id'])
            seen[key]=alias;canonical[group['id']]=alias;coefficients_by_group[group['id']]=value
        entry=dict(slot=slot,seed_path=atom['path'],seed_sha256=atom['sha256'],
            coefficients=coefficients_by_group,canonical_group=canonical)
        result.append(entry);entries.append(dict(slot=slot,coefficients=coefficients_by_group,canonical_group=canonical))
    write(R/'INPUT_COEFFICIENTS.json',dict(at=now(),source_seed_ready_sha256=sha(BASE/'ATOMS_READY.json'),
        target_access=False,patient_seed_already_complete=True,per_case=result))
    return entries

def forward_all(model,images,index,cohort,lookup,plan,coeffs):
    canvas=tuple(index['canvas']);atoms=[];new_count=0;gpu_seconds=0.;started=time.monotonic();available={}
    with (R/'MODEL_PREACCESS.json').open('x') as out:
        import json
        json.dump(dict(at=now(),checkpoint_sha256=CHECKPOINT_SHA,split='validation108',maximum_new_forwards=3024,
            seed_forwards_this_stage=0,seed_before_healthy_statistics=True,input_coefficients_sha256=sha(R/'INPUT_COEFFICIENTS.json'),
            execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),automatic_neural_retry=False),out)
    torch.cuda.reset_peak_memory_stats()
    for group in plan['groups']:
        gid=group['id'];eligible=[s for s in range(108) if coeffs[s]['canonical_group'][gid]==gid]
        for first in range(0,len(eligible),8):
            guard();assert time.monotonic()-started<plan['maximum_wall_seconds']
            slots=eligible[first:first+8];x=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda();support=x!=0
            for j,slot in enumerate(slots):x[j]=transform(x[j],support[j],coeffs[slot]['coefficients'][gid])
            geometry=[index['geometry'][cohort[s]['case_id']] for s in slots]
            for view in range(1,8):
                event0=torch.cuda.Event(enable_timing=True);event1=torch.cuda.Event(enable_timing=True);event0.record()
                values=forward_view(model,x,support,geometry,view,canvas);event1.record();event1.synchronize();gpu_seconds+=event0.elapsed_time(event1)/1000.
                for slot,value in zip(slots,values):
                    item=dict(slot=slot,view=view,input_group=gid,**save_atom(TMP/'atoms'/gid/f'{slot:03}'/f'flip{view}.npy',value))
                    available[gid,slot,view]=item;new_count+=1
                write(R/'FORWARD_JOURNAL.json',dict(at=now(),new_case_forwards=new_count,maximum=3024,atoms=list(available.values())))
                write(R/'HEARTBEAT.json',dict(at=now(),stage='real_normalized_RACE_forward',input_group=gid,
                    new_case_forwards=new_count,maximum=3024,gpu_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-started,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            del x,support,values
        for slot in range(108):
            canonical=coeffs[slot]['canonical_group'][gid]
            for view in range(1,8):
                source=lookup[slot,view] if canonical=='identity' else available[canonical,slot,view]
                item=dict(source,input_group=gid,canonical_input_group=canonical,reused_existing_prediction=(canonical!=gid))
                available[gid,slot,view]=item;atoms.append(item)
    assert len(atoms)==3024 and new_count<=3024
    ready=dict(at=now(),status='passed',checkpoint_path=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
        manifest_sha256=sha(ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'),
        index_sha256=sha(CACHE/'index.json'),canvas=list(canvas),input_coefficients_sha256=sha(R/'INPUT_COEFFICIENTS.json'),
        source_seed_ready_sha256=sha(BASE/'ATOMS_READY.json'),atoms=atoms,new_case_forwards=new_count,
        logical_adapted_atoms=3024,seed_forward_count=0,deployment_neural_forwards=8,
        gpu_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-started,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'))
    write(R/'ATOMS_READY.json',ready)
    return ready

def score(ready,index,cohort,lookup,plan):
    canvas=tuple(index['canvas']);labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*canvas))
    assert sha(CACHE/'labels.u8')==read(BASE/'CACHE_READY.json')['payload_sha256']['labels.u8']
    adapted={(a['input_group'],a['slot'],a['view']):a for a in ready['atoms']}
    recipes=plan['recipes'];ids=['equal8_logit_t05']+[r['id'] for r in recipes];records={key:[] for key in ids}
    previous=read(BASE/'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05'];baseline_checks=[]
    for slot,row in enumerate(cohort):
        guard();g=index['geometry'][row['case_id']];target=np.array(labels[slot][slices(g)]>0)
        old=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];z=np.load(atom['path'],allow_pickle=False)
            assert list(z.shape)==g['extent'] and np.isfinite(z).all();old.append(torch.from_numpy(z).cuda())
        base=torch.zeros_like(old[0])
        for value in old:base.add_(value,alpha=.125)
        bp=list(map(int,np.unravel_index(int(base.argmax()),target.shape)))
        base_row=record_mask(row,slot,(base>0).cpu().numpy(),bp,bp,target,TMP/'predictions'/ids[0]/f'{slot:03}.npz')
        assert all(base_row[k]==previous[slot][k] for k in ('tp','fp','fn','locator_dsc','automatic_point','automatic_point_hits_gt'))
        records[ids[0]].append(base_row);baseline_checks.append(dict(slot=slot,all_counts_dice_point_exact=True))
        for group in plan['groups']:
            values=[]
            for view in range(1,8):
                atom=adapted[group['id'],slot,view];assert sha(atom['path'])==atom['sha256'];z=np.load(atom['path'],allow_pickle=False)
                assert list(z.shape)==g['extent'] and np.isfinite(z).all();values.append(torch.from_numpy(z).cuda())
            for recipe in [r for r in recipes if r['input_group']['id']==group['id']]:
                value,point=decode(old[0],values,recipe)
                record=record_mask(row,slot,(value>0).cpu().numpy(),point,bp,target,TMP/'predictions'/recipe['id']/f'{slot:03}.npz')
                record['configuration_id']=recipe['id'];records[recipe['id']].append(record)
            del values,value
        write(R/'HEARTBEAT.json',dict(at=now(),stage='cached_normalization_decode',completed=slot+1,total=108,new_network_forwards=ready['new_case_forwards']))
        del old,base
    write(R/'BASELINE_EXACT_AUDIT.json',dict(at=now(),status='passed',per_case=baseline_checks))
    write(R/'RAW_VALIDATION108.json',dict(at=now(),per_configuration=records,recipes=recipes,
        checkpoint_sha256=CHECKPOINT_SHA,source_atoms_ready_sha256=sha(R/'ATOMS_READY.json'),
        new_network_forwards=ready['new_case_forwards'],new270_scoring=False))
    audit_masks(R/'RAW_VALIDATION108.json',R/'MASK_AUDIT_VALIDATION108.json',cohort,index,labels)
    ranking=[]
    for order,key in enumerate(ids):
        metrics=summary_ci(records[key]);ranking.append(dict(id=key,order=order,metrics=metrics,
            paired_vs_equal8=paired(records[key],records[ids[0]]),point_base=sum(r['base_automatic_point_hits_gt'] for r in records[key])))
    ranking.sort(key=lambda x:(-x['metrics']['scan_dice'],-x['metrics']['subject_dice'],-x['metrics']['point_hits'],x['order']))
    folds=[];heldout=[]
    for fold in range(5):
        dev=[s for s,row in enumerate(cohort) if plan['folds'][row['subject_id']]!=fold]
        held=[s for s,row in enumerate(cohort) if plan['folds'][row['subject_id']]==fold]
        winner=min(ids,key=lambda key:(-np.mean([records[key][s]['locator_dsc'] for s in dev]),
            -np.mean([np.mean([records[key][s]['locator_dsc'] for s in dev if cohort[s]['subject_id']==subject]) for subject in {cohort[s]['subject_id'] for s in dev}]),
            -sum(records[key][s]['automatic_point_hits_gt'] for s in dev),ids.index(key)))
        chosen=[records[winner][s] for s in held];heldout.extend(chosen);folds.append(dict(fold=fold,selected_id=winner,heldout_metrics=summary_ci(chosen)))
    best=ranking[0];best_new=next(x for x in ranking if x['id']!=ids[0]);worse={}
    for key in ids[1:]:
        worse[key]=sorted([dict(case_id=a['case_id'],subject_id=a['subject_id'],candidate=a['locator_dsc'],reference=b['locator_dsc'],delta=a['locator_dsc']-b['locator_dsc']) for a,b in zip(records[key],records[ids[0]]) if a['locator_dsc']<b['locator_dsc']],key=lambda x:x['delta'])
    result=dict(at=now(),state='complete',best=best,best_innovation=best_new,ranking=ranking,patient_fivefold=folds,
        heldout_selected_metrics=summary_ci(heldout),heldout_paired_vs_equal8=paired(sorted(heldout,key=lambda x:x['source_slot']),records[ids[0]]),
        worsened_cases_by_configuration=worse,checkpoint_sha256=CHECKPOINT_SHA,new_configurations=8,
        new_network_forwards=ready['new_case_forwards'],maximum_new_network_forwards=3024,deployment_neural_forwards=8,
        forward_gpu_event_seconds=ready['gpu_event_seconds'],forward_wall_seconds=ready['wall_seconds'],peak_reserved_gib=ready['peak_reserved_gib'],
        source_sha256=read(R/'EXECUTION_FREEZE.json')['source_sha256'],raw_sha256=sha(R/'RAW_VALIDATION108.json'),
        masks_audit_path=str(R/'MASK_AUDIT_VALIDATION108.json'),masks_audit_sha256=sha(R/'MASK_AUDIT_VALIDATION108.json'),
        primary_point='actual final score argmax',auxiliary_point='equal8 argmax',training_updates=0,new270_scoring=False,
        test_tuned_history=True,independent_confirmation=False)
    write(R/'RESULT.json',result)
    selected=next(x for x in recipes if x['id']==best_new['id'])
    write(R/'DEPLOYMENT_RECIPE.json',dict(at=now(),innovation_entry=dict(id=selected['id'],mode='race_seed_healthy_normalization',config=selected,neural_case_forwards=8),
        best_overall_id=best['id'],metrics=best_new['metrics'],source_result_sha256=sha(R/'RESULT.json'),
        raw_sha256=sha(R/'RAW_VALIDATION108.json'),deployment_order=plan['deployment_order'],
        input_callable='scripts.race_v2_inference_normalization_a_20261029.algorithm.coefficients',
        score_callable='scripts.race_v2_inference_normalization_a_20261029.algorithm.decode',
        source_sha256=result['source_sha256'],new270_authorization=False))
    (R/'RESULT.md').write_text(f'''# RACE seed-excluded healthy-support normalization

Eight fixed new configurations completed on108 validation scans. Best overall `{best['id']}` Scan {best['metrics']['scan_dice']:.9f}. Best new `{best_new['id']}` Scan {best_new['metrics']['scan_dice']:.9f}, Subject {best_new['metrics']['subject_dice']:.9f}, Point {best_new['metrics']['point_hits']}/108. Delta versus equal8 {best_new['paired_vs_equal8']['scan_delta']:+.9f}, descriptive patient95% interval {best_new['paired_vs_equal8']['scan_delta_ci95']}.

Deployment uses one original seed and seven normalized mirrors, eight forwards total. New validation neural forwards {ready['new_case_forwards']}/3024; all eight decoders reuse cached logits. No training or new270 scoring. Validation and historical test reuse remain test-tuned; no independent confirmation. Whole original crop and native1mm score unchanged. Full counts, masks, all worsening cases, patient folds, cost and exact recipes are retained in JSON.
''')
    write(R/'DONE.json',dict(at=now(),state='complete',result_sha256=sha(R/'RESULT.json'),new_network_forwards=ready['new_case_forwards'],new270_scoring=False))

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';TMP.mkdir(parents=True,exist_ok=True);freeze=verify();torch.set_num_threads(2)
    LOCK.parent.mkdir(parents=True,exist_ok=True);lock=LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(R/'GPU_LOCK.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=0,lock_path=str(LOCK),
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()))
    plan=read(R/'PLAN.json');index=read(CACHE/'index.json');canvas=tuple(index['canvas']);cohort=rows()
    assert index['case_ids']==[r['case_id'] for r in cohort] and sha(CACHE/'index.json')==plan['source_cache_index_sha256']
    assert sha(CACHE/'images.f32')==read(BASE/'CACHE_READY.json')['payload_sha256']['images.f32']
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*canvas));source=read(BASE/'ATOMS_READY.json')
    assert sha(BASE/'ATOMS_READY.json')==plan['source_atoms_ready_sha256'] and source['checkpoint_sha256']==CHECKPOINT_SHA
    lookup={(a['slot'],a['view']):a for a in source['atoms']};assert len(lookup)==864
    model=load_model();gate=synthetic(model,canvas);write(R/'GPU_GATE.json',gate)
    write(R/'EXECUTION_FREEZE.json',dict(freeze,at=now(),gpu_gate_sha256=sha(R/'GPU_GATE.json'),checkpoint_sha256=CHECKPOINT_SHA))
    coeffs=prepare_coefficients(images,index,cohort,lookup,plan['groups']);verify()
    ready=forward_all(model,images,index,cohort,lookup,plan,coeffs)
    del model;torch.cuda.empty_cache();score(ready,index,cohort,lookup,plan)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(R/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
