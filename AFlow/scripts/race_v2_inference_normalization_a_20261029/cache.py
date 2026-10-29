"""One registered normalization recipe, dynamic split, immutable eight-atom raw."""
import json,os,time
import numpy as np
import torch
from scripts.race_v2_inference_normalization_a_20261029.common import read,write,sha,now,guard,slices
from scripts.race_v2_inference_normalization_a_20261029.algorithm import coefficients,transform,forward_view

def validate_ready(path,metadata,index,cohort,manifest_sha256,index_sha256):
    ready=read(path);assert ready['model_id']==metadata['model_id']
    assert ready['base_model_id']==metadata['base_model_id'] and ready['checkpoint_sha256']==metadata['checkpoint_sha256']
    assert ready['input_recipe']==metadata['input_recipe'] and ready['input_recipe_sha256']==metadata['input_recipe_sha256']
    assert ready['manifest_sha256']==manifest_sha256 and ready['index_sha256']==index_sha256
    assert ready['case_forwards']==len(cohort)*8 and len(ready['atoms'])==len(cohort)*8
    assert {(a['slot'],a['view']) for a in ready['atoms']}=={(s,v) for s in range(len(cohort)) for v in range(8)}
    for atom in ready['atoms']:
        assert sha(atom['path'])==atom['sha256'];value=np.load(atom['path'],mmap_mode='r',allow_pickle=False)
        assert list(value.shape)==atom['shape']==index['geometry'][cohort[atom['slot']]['case_id']]['extent']
        assert value.dtype==np.float32 and np.isfinite(value).all()
    return ready

def seal(metadata,index,cohort,manifest_sha256,index_sha256,folder,selection_sha256=None):
    journal=read(folder/'FORWARD_JOURNAL.json');atoms=journal['atoms']
    assert journal['metadata']==metadata and journal['selection_sha256']==selection_sha256
    assert len(atoms)==len(cohort)*8 and {(a['slot'],a['view']) for a in atoms}=={(s,v) for s in range(len(cohort)) for v in range(8)},'incomplete raw: no automatic neural retry'
    for atom in atoms:
        assert sha(atom['path'])==atom['sha256'];z=np.load(atom['path'],mmap_mode='r',allow_pickle=False)
        assert list(z.shape)==atom['shape']==index['geometry'][cohort[atom['slot']]['case_id']]['extent'] and np.isfinite(z).all()
    assert sha(metadata['checkpoint_path'])==metadata['checkpoint_sha256']
    result=dict(at=now(),status='passed',model_id=metadata['model_id'],base_model_id=metadata['base_model_id'],
        checkpoint_path=metadata['checkpoint_path'],checkpoint_sha256=metadata['checkpoint_sha256'],
        architecture=metadata['architecture'],construction=metadata['construction'],
        atom_source='seed_health_normalization',input_recipe=metadata['input_recipe'],input_recipe_sha256=metadata['input_recipe_sha256'],
        manifest_sha256=manifest_sha256,index_sha256=index_sha256,canvas=index['canvas'],
        case_forwards=len(cohort)*8,actual_new_forwards=journal['actual_new_forwards'],
        new_model_forwards=journal['actual_new_forwards'],seed_new_forwards=journal['seed_new_forwards'],
        normalized_new_forwards=journal['normalized_new_forwards'],normalization_new_forwards=journal['normalized_new_forwards'],seed_dependency=journal['seed_dependency'],
        input_coefficients_path=str(folder/'INPUT_COEFFICIENTS.json'),input_coefficients_sha256=sha(folder/'INPUT_COEFFICIENTS.json'),
        atoms=atoms,root_selection_sha256=selection_sha256,gpu_event_seconds=journal['gpu_event_seconds'],
        journal_path=str(folder/'FORWARD_JOURNAL.json'),journal_sha256=sha(folder/'FORWARD_JOURNAL.json'),
        wall_seconds=journal['wall_seconds'],peak_reserved_gib=journal['peak_reserved_gib'],
        logical_deployment_forwards=8,geometry_contract='original support unchanged; R_view=T@R@T; support mirrored; aligned crop logits',
        raw_complete_no_neural_retry=True)
    write(metadata['atoms_ready_path'],result)
    return result

def generate(network,metadata,index,cohort,images,manifest_sha256,index_sha256,folder,scratch,source_ready_path=None,selection_sha256=None):
    """Caller has GPU0 lock and a strictly reloaded, synthetic-gated network."""
    from pathlib import Path
    folder=Path(folder);scratch=Path(scratch)
    if Path(metadata['atoms_ready_path']).exists():return validate_ready(metadata['atoms_ready_path'],metadata,index,cohort,manifest_sha256,index_sha256)
    if (folder/'MODEL_PREACCESS.json').exists():
        seal(metadata,index,cohort,manifest_sha256,index_sha256,folder,selection_sha256)
        return validate_ready(metadata['atoms_ready_path'],metadata,index,cohort,manifest_sha256,index_sha256)
    folder.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    assert not list(scratch.glob('atoms/*/*.npy'))
    assert sha(metadata['checkpoint_path'])==metadata['checkpoint_sha256']
    entry=metadata['input_recipe'];assert entry['mode']=='race8_seed_health_normalization' and entry['neural_case_forwards']==8
    recipe=entry['config']['input_group'];sources={};dependency=None
    if source_ready_path is not None:
        source=read(source_ready_path);assert source['checkpoint_sha256']==metadata['checkpoint_sha256']
        assert source['manifest_sha256']==manifest_sha256 and source['index_sha256']==index_sha256
        sources={(a['slot'],a['view']):a for a in source['atoms']};assert len(sources)==len(cohort)*8
        dependency=dict(path=str(source_ready_path),sha256=sha(source_ready_path),checkpoint_sha256=source['checkpoint_sha256'],
            used_seed_view=0,original_input_only=True,new_seed_forwards=0)
    with (folder/'MODEL_PREACCESS.json').open('x') as out:
        json.dump(dict(at=now(),metadata=metadata,selection_sha256=selection_sha256,seed_dependency=dependency,
            maximum_new_forwards=len(cohort)*(7 if sources else 8),logical_deployment_forwards=8,
            target_access=False,no_automatic_neural_retry=True),out,ensure_ascii=False,indent=2)
    atoms=[];coefficient_rows=[];new_seed=0;new_norm=0;gpu_seconds=0.;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    canvas=tuple(index['canvas'])
    def persist(slot,view,value,kind):
        path=scratch/'atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
        temp=path.with_name(path.name+'.tmp')
        with temp.open('xb') as out:np.save(out,value,allow_pickle=False)
        os.replace(temp,path);path.chmod(0o444)
        return dict(slot=slot,view=view,path=str(path),shape=list(value.shape),sha256=sha(path),input_kind=kind)
    def journal():
        write(folder/'INPUT_COEFFICIENTS.json',dict(at=now(),target_access=False,seed_before_statistics=True,per_case=coefficient_rows))
        write(folder/'FORWARD_JOURNAL.json',dict(at=now(),metadata=metadata,selection_sha256=selection_sha256,
            atoms=atoms,actual_new_forwards=new_seed+new_norm,seed_new_forwards=new_seed,normalized_new_forwards=new_norm,
            seed_dependency=dependency,gpu_event_seconds=gpu_seconds,wall_seconds=time.monotonic()-started,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(folder/'HEARTBEAT.json',dict(at=now(),stage='fixed_health_normalization_forward',logical_atoms=len(atoms),
            maximum_atoms=len(cohort)*8,actual_new_forwards=new_seed+new_norm))
    for first in range(0,len(cohort),8):
        guard();slots=list(range(first,min(first+8,len(cohort))));geometry=[index['geometry'][cohort[s]['case_id']] for s in slots]
        original=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda();support=original!=0
        if sources:
            seed_values=[]
            for slot in slots:
                atom=sources[slot,0];assert sha(atom['path'])==atom['sha256'];seed=np.load(atom['path'],allow_pickle=False)
                assert list(seed.shape)==atom['shape']==index['geometry'][cohort[slot]['case_id']]['extent'];seed_values.append(seed)
                atoms.append(dict(atom,input_kind='original_seed_reused'))
        else:
            e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
            seed_values=forward_view(network,original,support,geometry,0,canvas);e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.
            for slot,seed in zip(slots,seed_values):atoms.append(persist(slot,0,seed,'original_seed_new'));new_seed+=1
        adapted=original.clone();active=[]
        for j,(slot,g,seed) in enumerate(zip(slots,geometry,seed_values)):
            c=coefficients(np.asarray(images[slot][slices(g)]),seed,recipe)
            coefficient_rows.append(dict(slot=slot,coefficient=c,seed_sha256=next(a['sha256'] for a in atoms if a['slot']==slot and a['view']==0)))
            if sources and (c['gain'],c['offset'])==(1.,0.):
                for view in range(1,8):
                    atom=sources[slot,view];assert sha(atom['path'])==atom['sha256'];atoms.append(dict(atom,input_kind='identity_fallback_reused'))
            else:adapted[j]=transform(original[j],support[j],c);active.append(j)
        journal()
        if active:
            chosen=[slots[j] for j in active];geometries=[geometry[j] for j in active]
            for view in range(1,8):
                e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
                values=forward_view(network,adapted[active],support[active],geometries,view,canvas);e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.
                for slot,value in zip(chosen,values):atoms.append(persist(slot,view,value,'healthy_normalized'));new_norm+=1
                journal()
        del original,adapted,support,seed_values
    return seal(metadata,index,cohort,manifest_sha256,index_sha256,folder,selection_sha256)
