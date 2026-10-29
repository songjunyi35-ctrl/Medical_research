"""One fixed context recipe, seed dependency and atomic eight-view cache."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from scripts.race_v2_inference_soft_context_a_20261029.common import read,write,sha,now,guard,slices
from scripts.race_v2_inference_soft_context_a_20261029.algorithm import distance,prepare_grid,context_field,transform
from scripts.race_v2_inference_normalization_a_20261029.algorithm import forward_view

def validate_ready(metadata,index,cohort,manifest_sha256,index_sha256):
    ready=read(metadata['atoms_ready_path'])
    for key in ('model_id','base_model_id','checkpoint_sha256','input_recipe','input_recipe_sha256'):assert ready[key]==metadata[key],key
    assert ready['atom_source']=='seed_soft_context' and ready['manifest_sha256']==manifest_sha256 and ready['index_sha256']==index_sha256
    assert ready['case_forwards']==len(cohort)*8 and len(ready['atoms'])==len(cohort)*8
    assert {(a['slot'],a['view']) for a in ready['atoms']}=={(s,v) for s in range(len(cohort)) for v in range(8)}
    assert ready['new_model_forwards']==sum(a['input_kind'] in ('original_seed_new','soft_context') for a in ready['atoms'])
    for atom in ready['atoms']:
        assert sha(atom['path'])==atom['sha256'];z=np.load(atom['path'],mmap_mode='r',allow_pickle=False)
        assert list(z.shape)==atom['shape']==index['geometry'][cohort[atom['slot']]['case_id']]['extent'] and z.dtype==np.float32 and np.isfinite(z).all()
    assert sha(ready['journal_path'])==ready['journal_sha256'] and sha(ready['input_fields_path'])==ready['input_fields_sha256']
    seed_sha={a['slot']:a['sha256'] for a in ready['atoms'] if a['view']==0}
    for field in read(ready['input_fields_path'])['fields']:
        assert field['seed_sha256']==seed_sha[field['slot']] and sha(field['path'])==field['sha256']
        value=np.load(field['path'],mmap_mode='r',allow_pickle=False)
        assert list(value.shape)==field['shape']==index['canvas'] and np.isfinite(value).all() and value.min()>=0 and value.max()<=1
    return ready

def seal(metadata,index,cohort,manifest_sha256,index_sha256,folder,selection_sha256=None):
    journal=read(folder/'FORWARD_JOURNAL.json');atoms=journal['atoms']
    assert journal['metadata']==metadata and journal['selection_sha256']==selection_sha256
    assert len(atoms)==len(cohort)*8 and {(a['slot'],a['view']) for a in atoms}=={(s,v) for s in range(len(cohort)) for v in range(8)},'incomplete raw: no automatic neural retry'
    for atom in atoms:
        assert sha(atom['path'])==atom['sha256'];z=np.load(atom['path'],mmap_mode='r',allow_pickle=False)
        assert list(z.shape)==atom['shape']==index['geometry'][cohort[atom['slot']]['case_id']]['extent'] and np.isfinite(z).all()
    assert sha(metadata['checkpoint_path'])==metadata['checkpoint_sha256']
    result=dict(at=now(),status='passed',**metadata,atom_source='seed_soft_context',
        manifest_sha256=manifest_sha256,index_sha256=index_sha256,canvas=index['canvas'],case_forwards=len(cohort)*8,
        new_model_forwards=journal['seed_new_forwards']+journal['context_new_forwards'],
        seed_new_forwards=journal['seed_new_forwards'],context_new_forwards=journal['context_new_forwards'],
        seed_dependency=journal['seed_dependency'],atoms=atoms,root_selection_sha256=selection_sha256,
        input_fields_path=str(folder/'INPUT_FIELDS.json'),input_fields_sha256=sha(folder/'INPUT_FIELDS.json'),
        journal_path=str(folder/'FORWARD_JOURNAL.json'),journal_sha256=sha(folder/'FORWARD_JOURNAL.json'),
        gpu_event_seconds=journal['gpu_event_seconds'],wall_seconds=journal['wall_seconds'],peak_reserved_gib=journal['peak_reserved_gib'],
        logical_deployment_forwards=8,geometry_contract='original native grid/support/R unchanged; R_view=T@R@T; scalar context field only is resampled',
        raw_complete_no_neural_retry=True)
    write(metadata['atoms_ready_path'],result);return validate_ready(metadata,index,cohort,manifest_sha256,index_sha256)

def generate(network,metadata,index,cohort,images,manifest_sha256,index_sha256,folder,scratch,source_ready_path=None,selection_sha256=None):
    folder=Path(folder);scratch=Path(scratch)
    if Path(metadata['atoms_ready_path']).exists():return validate_ready(metadata,index,cohort,manifest_sha256,index_sha256)
    if (folder/'MODEL_PREACCESS.json').exists():return seal(metadata,index,cohort,manifest_sha256,index_sha256,folder,selection_sha256)
    folder.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True);assert not list(scratch.glob('atoms/*/*.npy'))
    assert sha(metadata['checkpoint_path'])==metadata['checkpoint_sha256'];entry=metadata['input_recipe'];recipe=entry['config']
    assert entry['mode']=='race8_seed_soft_context' and entry['neural_case_forwards']==8
    source={};dependency=None
    if source_ready_path is not None:
        ready=read(source_ready_path);assert ready['checkpoint_sha256']==metadata['checkpoint_sha256'] and ready['manifest_sha256']==manifest_sha256 and ready['index_sha256']==index_sha256
        source={(a['slot'],a['view']):a for a in ready['atoms']};assert len(source)==len(cohort)*8
        dependency=dict(path=str(source_ready_path),sha256=sha(source_ready_path),checkpoint_sha256=ready['checkpoint_sha256'],view=0,original_input=True)
    with (folder/'MODEL_PREACCESS.json').open('x') as out:
        json.dump(dict(at=now(),metadata=metadata,selection_sha256=selection_sha256,seed_dependency=dependency,
            maximum_new_forwards=len(cohort)*(7 if source else 8),no_automatic_neural_retry=True,GT_input=False),out,ensure_ascii=False,indent=2)
    atoms=[];fields=[];new_seed=0;new_context=0;gpu_seconds=0.;started=time.monotonic();canvas=tuple(index['canvas']);torch.cuda.reset_peak_memory_stats()
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();temporary=path.with_name(path.name+'.tmp')
        with temporary.open('xb') as out:np.save(out,value,allow_pickle=False)
        os.replace(temporary,path);path.chmod(0o444)
        return dict(path=str(path),shape=list(value.shape),sha256=sha(path))
    def journal():
        write(folder/'INPUT_FIELDS.json',dict(at=now(),fields=fields,GT_input=False,seed_before_field=True))
        write(folder/'FORWARD_JOURNAL.json',dict(at=now(),metadata=metadata,selection_sha256=selection_sha256,atoms=atoms,
            seed_new_forwards=new_seed,context_new_forwards=new_context,seed_dependency=dependency,gpu_event_seconds=gpu_seconds,
            wall_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        write(folder/'HEARTBEAT.json',dict(at=now(),stage='fixed_soft_context_forward',logical_atoms=len(atoms),maximum=len(cohort)*8,new_model_forwards=new_seed+new_context))
    for first in range(0,len(cohort),8):
        guard();slots=list(range(first,min(first+8,len(cohort))));geometry=[index['geometry'][cohort[s]['case_id']] for s in slots]
        original=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda();support=original!=0
        if source:
            seeds=[]
            for slot in slots:
                atom=source[slot,0];assert sha(atom['path'])==atom['sha256'];seeds.append(np.load(atom['path'],allow_pickle=False));atoms.append(dict(atom,input_kind='original_seed_reused'))
        else:
            e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
            seeds=forward_view(network,original,support,geometry,0,canvas);e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.
            for slot,seed in zip(slots,seeds):atoms.append(dict(slot=slot,view=0,input_kind='original_seed_new',**save(scratch/'atoms'/f'{slot:03}'/'flip0.npy',seed)));new_seed+=1
        distances=[];empty=[]
        for slot,g,seed in zip(slots,geometry,seeds):
            assert list(seed.shape)==g['extent'];full=np.full(canvas,-99,np.float32);full[slices(g)]=seed
            d,e=distance(full,np.asarray(images[slot])!=0);distances.append(d);empty.append(e)
        dt=torch.from_numpy(np.stack(distances))[:,None].cuda();matrix=torch.tensor([g['reflection'] for g in geometry],device='cuda',dtype=torch.float32)
        enabled=torch.tensor([g['reference_enabled'] for g in geometry],device='cuda',dtype=torch.bool)
        e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
        field=context_field(dt,matrix,enabled,recipe['radius_mm']);changed=transform(original,field,recipe['floor'])
        e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.;assert torch.equal(changed!=0,support)
        active=[];field_values=field[:,0].cpu().numpy()
        for j,(slot,value) in enumerate(zip(slots,field_values)):
            identity=bool(np.all(value[np.asarray(images[slot])!=0]==1))
            fields.append(dict(slot=slot,seed_sha256=next(a['sha256'] for a in atoms if a['slot']==slot and a['view']==0),
                empty_seed=bool(empty[j]),identity_input=identity,**save(scratch/'fields'/f'{slot:03}.npy',value)))
            if source and identity:
                for view in range(1,8):
                    atom=source[slot,view];assert sha(atom['path'])==atom['sha256'];atoms.append(dict(atom,input_kind='identity_fallback_reused'))
            else:active.append(j)
        journal()
        if active:
            chosen=[slots[j] for j in active];geometries=[geometry[j] for j in active]
            for view in range(1,8):
                e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e0.record()
                values=forward_view(network,changed[active],support[active],geometries,view,canvas);e1.record();e1.synchronize();gpu_seconds+=e0.elapsed_time(e1)/1000.
                for slot,value in zip(chosen,values):atoms.append(dict(slot=slot,view=view,input_kind='soft_context',**save(scratch/'atoms'/f'{slot:03}'/f'flip{view}.npy',value)));new_context+=1
                journal()
        del original,support,seeds,distances,dt,matrix,enabled,field,changed,field_values
    return seal(metadata,index,cohort,manifest_sha256,index_sha256,folder,selection_sha256)
