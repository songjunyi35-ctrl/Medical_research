"""One field cache per radius, four input groups, one neural pass per new view."""
import fcntl,os,subprocess,time
import numpy as np
import torch
from scripts.race_v2_inference_soft_context_a_20261029.common import *
from scripts.race_v2_inference_soft_context_a_20261029.algorithm import distance,prepare_grid,context_field,transform,decode,numerical_gate
from scripts.race_v2_inference_a_20261029.predictor import load_model
from scripts.race_v2_inference_normalization_a_20261029.algorithm import forward_view
from scripts.race_v2_inference_normalization_a_20261029.deployment import gate_network
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired
from scripts.race_v2_inference_innovation_a_20261029.io_helpers import record_mask,audit_masks

def context_native_gate(model,canvas):
    torch.manual_seed(1091);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    image=torch.randn(8,1,*canvas,device='cuda');support=image!=0
    d=torch.arange(canvas[0],device='cuda',dtype=torch.float32)[:,None,None].expand(canvas).contiguous()
    distances=d[None,None].expand(8,1,*canvas).contiguous()
    g=dict(canvas_offset=[0,0,0],extent=list(canvas),reference_enabled=True,
        reflection=[[-1,0,0,canvas[0]-1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    matrix=torch.tensor([g['reflection']]*8,device='cuda',dtype=torch.float32)
    field=context_field(distances,matrix,torch.ones(8,device='cuda',dtype=torch.bool),10)
    changed=transform(image,field,.75);assert torch.equal(changed!=0,support)
    full=forward_view(model,changed,support,[g]*8,7,canvas)
    tail=forward_view(model,changed[:4],support[:4],[g]*4,1,canvas)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/2**30;assert peak<=38
    return dict(status='passed',canvas=list(canvas),batch=8,tail=4,context_field_then_model_finite=True,
        original_support_exact=True,peak_reserved_gib=peak,seconds=time.monotonic()-start,patient_forwards=0)

def save_array(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as out:np.save(out,value,allow_pickle=False)
    os.replace(temp,path);path.chmod(0o444)
    return dict(path=str(path),sha256=sha(path),shape=list(value.shape),dtype=str(value.dtype))

def prepare_fields(index,cohort,images,lookup,started):
    canvas=tuple(index['canvas']);fields=[];gpu_seconds=0.
    for first in range(0,108,8):
        guard();assert time.monotonic()-started<1200;slots=list(range(first,min(first+8,108)));distances=[];empty=[];geometry=[];seeds=[]
        for slot in slots:
            g=index['geometry'][cohort[slot]['case_id']];atom=lookup[slot,0];assert sha(atom['path'])==atom['sha256']
            seed=np.load(atom['path'],allow_pickle=False);assert list(seed.shape)==g['extent']
            full=np.full(canvas,-99,np.float32);full[slices(g)]=seed
            d,e=distance(full,np.asarray(images[slot])!=0);distances.append(d);empty.append(e);geometry.append(g);seeds.append(atom['sha256'])
        dt=torch.from_numpy(np.stack(distances))[:,None].cuda();matrix=torch.tensor([g['reflection'] for g in geometry],device='cuda',dtype=torch.float32)
        enabled=torch.tensor([g['reference_enabled'] for g in geometry],device='cuda',dtype=torch.bool)
        e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record();grid=prepare_grid(matrix,canvas)
        for radius in (10,20):
            field=context_field(dt,matrix,enabled,radius,grid);values=field[:,0].cpu().numpy()
            for j,(slot,value) in enumerate(zip(slots,values)):
                assert np.isfinite(value).all() and value.min()>=0 and value.max()<=1
                identity=bool(np.all(value[np.asarray(images[slot])!=0]==1))
                fields.append(dict(slot=slot,radius_mm=radius,empty_seed=bool(empty[j]),identity_input=identity,
                    seed_sha256=seeds[j],reference_enabled=geometry[j]['reference_enabled'],
                    **save_array(TMP/'fields'/f'r{radius}'/f'{slot:03}.npy',value)))
        e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.
        write(R/'FIELD_JOURNAL.json',dict(at=now(),fields=fields,gpu_event_seconds=gpu_seconds,patient_network_forwards=0))
        write(R/'HEARTBEAT.json',dict(at=now(),stage='seed_only_context_field_preparation',cases_complete=first+len(slots),total=108,
            patient_network_forwards=0,wall_seconds=time.monotonic()-started))
        del dt,matrix,enabled,grid,field,values,distances
    ready=dict(at=now(),status='passed',fields=fields,source_seed_ready_sha256=sha(BASE/'ATOMS_READY.json'),
        cache_index_sha256=sha(CACHE/'index.json'),gpu_event_seconds=gpu_seconds,GT_access=False)
    write(R/'INPUT_FIELDS_READY.json',ready);return ready

def forward_all(model,plan,index,cohort,images,lookup,fields,started):
    fl={(a['radius_mm'],a['slot']):a for a in fields['fields']};atoms=[];new_count=0;gpu_seconds=0.;forward_started=time.monotonic();canvas=tuple(index['canvas'])
    with (R/'MODEL_PREACCESS.json').open('x') as out:
        import json
        json.dump(dict(at=now(),checkpoint_sha256=CHECKPOINT_SHA,maximum_new_forwards=3024,
            original_seed_forwards_this_stage=0,seed_precedes_field=True,input_fields_ready_sha256=sha(R/'INPUT_FIELDS_READY.json'),
            execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),new270_access=False,automatic_neural_retry=False),out)
    torch.cuda.reset_peak_memory_stats()
    for recipe in plan['recipes']:
        identifier=recipe['id'];eligible=[slot for slot in range(108) if not fl[recipe['radius_mm'],slot]['identity_input']]
        for first in range(0,len(eligible),8):
            guard();assert time.monotonic()-started<1200;slots=eligible[first:first+8]
            image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda();support=image!=0
            values=[]
            for slot in slots:
                item=fl[recipe['radius_mm'],slot];assert sha(item['path'])==item['sha256'];values.append(np.load(item['path'],allow_pickle=False))
            field=torch.from_numpy(np.stack(values))[:,None].cuda();changed=transform(image,field,recipe['floor'])
            assert bool(torch.equal(changed!=0,support));geometry=[index['geometry'][cohort[s]['case_id']] for s in slots]
            for view in range(1,8):
                e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
                outputs=forward_view(model,changed,support,geometry,view,canvas);e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.
                for slot,value in zip(slots,outputs):
                    atoms.append(dict(slot=slot,view=view,input_group=identifier,input_kind='soft_context',
                        **save_array(TMP/'atoms'/identifier/f'{slot:03}'/f'flip{view}.npy',value)));new_count+=1
                write(R/'FORWARD_JOURNAL.json',dict(at=now(),new_case_forwards=new_count,maximum=3024,atoms=atoms))
                write(R/'HEARTBEAT.json',dict(at=now(),stage='soft_context_real_RACE_forward',input_group=identifier,new_case_forwards=new_count,
                    maximum=3024,gpu_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-started))
            del image,support,field,changed,outputs,values
        for slot in range(108):
            if fl[recipe['radius_mm'],slot]['identity_input']:
                for view in range(1,8):atoms.append(dict(lookup[slot,view],input_group=identifier,input_kind='identity_input_reused'))
    assert len(atoms)==3024 and new_count<=3024
    ready=dict(at=now(),status='passed',checkpoint_sha256=CHECKPOINT_SHA,index_sha256=sha(CACHE/'index.json'),
        source_seed_ready_sha256=sha(BASE/'ATOMS_READY.json'),input_fields_ready_sha256=sha(R/'INPUT_FIELDS_READY.json'),
        atoms=atoms,new_case_forwards=new_count,logical_adapted_atoms=3024,deployment_neural_forwards=8,
        gpu_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-forward_started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    write(R/'ATOMS_READY.json',ready);return ready

def analyze(records,plan,ready,started):
    ids=list(records);baseline=records['equal8_logit_t05'];ranking=[]
    for order,key in enumerate(ids):ranking.append(dict(id=key,order=order,metrics=summary_ci(records[key]),
        paired_vs_equal8=paired(records[key],baseline),point_base=sum(r['base_automatic_point_hits_gt'] for r in records[key])))
    ranking.sort(key=lambda x:(-x['metrics']['scan_dice'],-x['metrics']['subject_dice'],-x['metrics']['point_hits'],x['order']))
    folds=[];heldout=[];cohort=rows()
    for fold in range(5):
        dev=[s for s,row in enumerate(cohort) if plan['folds'][row['subject_id']]!=fold];held=[s for s,row in enumerate(cohort) if plan['folds'][row['subject_id']]==fold]
        winner=min(ids,key=lambda key:(-np.mean([records[key][s]['locator_dsc'] for s in dev]),
            -np.mean([np.mean([records[key][s]['locator_dsc'] for s in dev if cohort[s]['subject_id']==subject]) for subject in {cohort[s]['subject_id'] for s in dev}]),
            -sum(records[key][s]['automatic_point_hits_gt'] for s in dev),ids.index(key)))
        selected=[records[winner][s] for s in held];heldout.extend(selected);folds.append(dict(fold=fold,selected_id=winner,heldout=summary_ci(selected)))
    worse={key:sorted([dict(case_id=a['case_id'],subject_id=a['subject_id'],delta=a['locator_dsc']-b['locator_dsc']) for a,b in zip(records[key],baseline) if a['locator_dsc']<b['locator_dsc']],key=lambda x:x['delta']) for key in ids[1:]}
    best=ranking[0];best_new=next(row for row in ranking if row['id']!='equal8_logit_t05')
    result=dict(at=now(),state='complete',best=best,best_innovation=best_new,ranking=ranking,patient_fivefold=folds,
        heldout_selected_metrics=summary_ci(heldout),heldout_paired_vs_equal8=paired(sorted(heldout,key=lambda r:r['source_slot']),baseline),
        worsened_cases_by_configuration=worse,checkpoint_sha256=CHECKPOINT_SHA,new_configurations=4,
        new_network_forwards=ready['new_case_forwards'],deployment_neural_forwards=8,training_updates=0,new270_scoring=False,
        source_sha256=read(R/'EXECUTION_FREEZE.json')['source_sha256'],raw_sha256=sha(R/'RAW_VALIDATION108.json'),
        masks_audit_receipt_path=str(R/'MASK_AUDIT_VALIDATION108.json'),masks_audit_receipt_sha256=sha(R/'MASK_AUDIT_VALIDATION108.json'),
        forward_gpu_event_seconds=ready['gpu_event_seconds'],forward_wall_seconds=ready['wall_seconds'],
        field_gpu_event_seconds=read(R/'INPUT_FIELDS_READY.json')['gpu_event_seconds'],total_wall_seconds=time.monotonic()-started,
        peak_reserved_gib=ready['peak_reserved_gib'],test_tuned_history=True,independent_confirmation=False,
        primary_point='actual final score argmax',auxiliary_point='ordinary equal8 argmax',no_new_patient_information_claim=True)
    write(R/'RESULT.json',result);recipe=next(r for r in plan['recipes'] if r['id']==best_new['id'])
    write(R/'SOFT_CONTEXT_DEPLOYMENT_RECIPE.json',dict(at=now(),innovation_entry=dict(id=recipe['id'],mode='race8_seed_soft_context',config=recipe,neural_case_forwards=8),
        metrics=best_new['metrics'],best_overall_id=best['id'],source_result_sha256=sha(R/'RESULT.json'),
        raw_sha256=sha(R/'RAW_VALIDATION108.json'),masks_audit_sha256=sha(R/'MASK_AUDIT_VALIDATION108.json'),
        seed_dependency='original-image identity logit only, same checkpoint',source_sha256=result['source_sha256'],new270_access=False))
    (R/'RESULT.md').write_text(f'''# RACE soft spatial context108

Four fixed configurations completed. Best overall `{best['id']}` Scan {best['metrics']['scan_dice']:.9f}; best new `{best_new['id']}` Scan {best_new['metrics']['scan_dice']:.9f}, Subject {best_new['metrics']['subject_dice']:.9f}, Point {best_new['metrics']['point_hits']}/108. Difference versus equal8 {best_new['paired_vs_equal8']['scan_delta']:+.9f}, descriptive patient95% interval {best_new['paired_vs_equal8']['scan_delta_ci95']}.

Actual new validation neural forwards {ready['new_case_forwards']}/3024. Deployment original seed1+context7=8; no training or270 scoring. Original native grid, support, reflection and whole-crop GT scoring retained. The weak feature-attention analogy does not establish efficacy or introduce new patient information. All cases/masks/points/counts, worsening cases, patient folds and costs are retained. Historical test tuning and validation reuse prevent independent confirmation claims.
''')
    write(R/'DONE.json',dict(at=now(),state='complete',result_sha256=sha(R/'RESULT.json'),new_network_forwards=ready['new_case_forwards'],new270_scoring=False))

def score(plan,ready,index,cohort,images,lookup,started):
    assert sha(CACHE/'labels.u8')==read(BASE/'CACHE_READY.json')['payload_sha256']['labels.u8']
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*index['canvas']));adapted={(a['input_group'],a['slot'],a['view']):a for a in ready['atoms']}
    records={key:[] for key in ['equal8_logit_t05']+[r['id'] for r in plan['recipes']]};reference=read(BASE/'RAW_VALIDATION108.json')['per_configuration']['equal8_logit_t05'];checks=[]
    for slot,row in enumerate(cohort):
        guard();g=index['geometry'][row['case_id']];target=np.array(labels[slot][slices(g)]>0);old=[]
        for view in range(8):
            atom=lookup[slot,view];assert sha(atom['path'])==atom['sha256'];old.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
        base,bp=decode(old,dict(mode='equal'));record=record_mask(row,slot,(base>0).cpu().numpy(),bp,bp,target,TMP/'predictions/equal8_logit_t05'/f'{slot:03}.npz')
        assert all(record[k]==reference[slot][k] for k in ('tp','fp','fn','locator_dsc','automatic_point','automatic_point_hits_gt'))
        records['equal8_logit_t05'].append(record);checks.append(dict(slot=slot,all_counts_dice_point_exact=True))
        for recipe in plan['recipes']:
            zs=[old[0]]
            for view in range(1,8):
                atom=adapted[recipe['id'],slot,view];assert sha(atom['path'])==atom['sha256'];zs.append(torch.from_numpy(np.load(atom['path'],allow_pickle=False)).cuda())
            value,point=decode(zs,recipe)
            records[recipe['id']].append(record_mask(row,slot,(value>0).cpu().numpy(),point,bp,target,TMP/'predictions'/recipe['id']/f'{slot:03}.npz'))
            del zs,value
        write(R/'HEARTBEAT.json',dict(at=now(),stage='cached_soft_context_decode',completed=slot+1,total=108,new_network_forwards=ready['new_case_forwards']))
        del old,base
    write(R/'BASELINE_EXACT_AUDIT.json',dict(at=now(),status='passed',cases=checks))
    raw=R/'RAW_VALIDATION108.json';write(raw,dict(at=now(),per_configuration=records,recipes=plan['recipes'],checkpoint_sha256=CHECKPOINT_SHA,
        source_atoms_ready_sha256=sha(R/'ATOMS_READY.json'),new_network_forwards=ready['new_case_forwards'],new270_scoring=False))
    audit_masks(raw,R/'MASK_AUDIT_VALIDATION108.json',cohort,index,labels);analyze(records,plan,ready,started)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';TMP.mkdir(parents=True,exist_ok=True);pre=verify();torch.set_num_threads(2);started=time.monotonic()
    lock=LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(R/'GPU_LOCK.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=0,lock=str(LOCK),
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()))
    plan=read(R/'PLAN.json');index=read(CACHE/'index.json');cohort=rows();assert index['case_ids']==[r['case_id'] for r in cohort]
    assert sha(CACHE/'index.json')==plan['cache_index_sha256'] and sha(CACHE/'images.f32')==read(BASE/'CACHE_READY.json')['payload_sha256']['images.f32']
    assert sha(BASE/'ATOMS_READY.json')==plan['source_atoms_ready_sha256'];source=read(BASE/'ATOMS_READY.json');assert source['checkpoint_sha256']==CHECKPOINT_SHA
    lookup={(a['slot'],a['view']):a for a in source['atoms']};assert len(lookup)==864
    model=load_model();gate=numerical_gate('cuda');native=gate_network(model,tuple(index['canvas']),4)
    context_native=context_native_gate(model,tuple(index['canvas']))
    write(R/'GPU_GATE.json',dict(at=now(),**gate,native_model_gate=native,native_context_model_gate=context_native));write(R/'EXECUTION_FREEZE.json',dict(pre,at=now(),gpu_gate_sha256=sha(R/'GPU_GATE.json')))
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*index['canvas']))
    fields=prepare_fields(index,cohort,images,lookup,started);verify();ready=forward_all(model,plan,index,cohort,images,lookup,fields,started)
    del model;torch.cuda.empty_cache();score(plan,ready,index,cohort,images,lookup,started)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(R/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
