"""Final dispatcher adding one selected guided input; all old sources preserved."""
import argparse
import subprocess
import sys
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029 import final270 as legacy
from scripts.race_v2_inference_b_20261029.image_conflict_b import final270 as previous
from scripts.race_v2_inference_b_20261029.forward import load_model
from scripts.race_v2_inference_b_20261029.edge_input_b.core import transform
E=R/'edge_input_b';F=R/'final270';WREADY=F/'models/w32e171_guided/ATOMS_READY.json';CORE=ROOT/'scripts/race_v2_inference_b_20261029/edge_input_b/core.py'

def verify_entry():
    previous.verify_entry()
    for path,digest in read(R/'FINAL270_GUIDED_ENTRY_FREEZE.json')['source_sha256'].items():assert sha(ROOT/path)==digest,path

def validate_input_model(model):
    bound=read(E/'INPUT_DEPLOYMENT_RECIPE.json');spec=bound['input_recipe']
    assert model['model_id']=='w32e171_guided' and model['base_model_id']=='w32e171' and model['atom_source']=='guided_input_w32'
    assert model['checkpoint_sha256']==CHECKPOINT_SHA and sha(model['checkpoint_path'])==CHECKPOINT_SHA
    assert model['input_recipe']==spec and Path(model['input_recipe_path'])==E/'INPUT_DEPLOYMENT_RECIPE.json'
    assert sha(model['input_recipe_path'])==model['input_recipe_sha256']
    assert bound['source_sha256']==sha(CORE) and bound['checkpoint_sha256']==CHECKPOINT_SHA
    assert Path(model['atoms_ready_path'])==WREADY
    return spec

def selection(candidate_id=None):
    verify_entry();root,candidate=previous.selection(candidate_id)
    if candidate is not None:
        guided=[m for m in candidate['models'] if m['model_id']=='w32e171_guided']
        if guided:
            assert len(guided)==1 and candidate['models'][1] is guided[0]
            validate_input_model(guided[0]);recipe=read(E/'DEPLOYMENT_RECIPE.json')['candidate']
            assert recipe['input_spec']==guided[0]['input_recipe']
            fusion=candidate['fusion']
            if recipe['decoder']=='fixed075':assert fusion['space']=='logit' and fusion['primary_weight']==.75 and fusion['decoder']=='raw'
            else:
                assert recipe['decoder']=='covariance' and fusion['space']=='paired_covariance_shrinkage'
                assert fusion['algorithm_recipe']==recipe['covariance_recipe']
    return root,candidate

@torch.inference_mode()
def synthetic(model,canvas,beta):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();begun=time.monotonic()
    for batch in (8,6):
        image=torch.zeros((batch,1,*canvas),device='cuda');image[:,:,16:-16,16:-16,16:-16]=.7;image[:,:,40:60,40:65,40:70]=2.
        image[0,0]=transform(image[0,0],beta)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model(image).segmentation_logits
        assert value.shape==image.shape and bool(value.isfinite().all());del image,value
    torch.cuda.synchronize();peak=torch.cuda.max_memory_reserved()/2**30;assert peak<38
    write(WREADY.parent/'NUMERIC_GATE.json',dict(at=now(),status='passed',canvas=list(canvas),batch=8,tail=6,
        strict_cpu_checkpoint_reload=True,peak_reserved_gib=peak,seconds=time.monotonic()-begun,patient_forwards=0))

@torch.inference_mode()
def forward_guided():
    root,_=selection();models=[m for c in root['candidates'] for m in c['models'] if m['model_id']=='w32e171_guided'];assert len(models)==1
    model_info=models[0];spec=validate_input_model(model_info)
    assert not WREADY.exists() and not (WREADY.parent/'MODEL_PREACCESS.json').exists(),'no automatic neural retry or duplicate complete cache'
    lock=gpu_lock();numeric_mode();torch.cuda.set_per_process_memory_fraction(.94);model=load_model()
    canvas=tuple(root['cohort']['canvas']);assert canvas==(160,208,168);synthetic(model,canvas,spec['beta'])
    index,cohort,images,_=legacy.test_data(root,False)
    exclusive(WREADY.parent/'MODEL_PREACCESS.json',dict(at=now(),selection_sha256=sha(STAGE/'FINAL_TEST_SELECTION.json'),
        checkpoint_sha256=CHECKPOINT_SHA,input_recipe=spec,input_recipe_sha256=model_info['input_recipe_sha256'],
        maximum_case_forwards=2160,views=list(range(8)),automatic_neural_retry=False))
    entries=[];started=time.monotonic();calls=0
    for first in range(0,270,8):
        guard();slots=list(range(first,min(first+8,270)));image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
        for k in range(len(slots)):image[k,0]=transform(image[k,0],spec['beta'])
        for view in range(8):
            dims=tuple(a+2 for a in range(3) if view&(1<<a));changed=torch.flip(image,dims) if dims else image
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(changed).segmentation_logits
            output=output.float();output=torch.flip(output,dims) if dims else output;assert bool(output.isfinite().all());calls+=1
            for k,slot in enumerate(slots):
                array=output[(k,0)+slices(index['geometry'][cohort[slot]['case_id']])].contiguous().cpu().numpy()
                path=legacy.FTMP/'models/w32e171_guided/atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+'.tmp')
                with temporary.open('xb') as handle:np.save(handle,array,allow_pickle=False)
                assert not path.exists();os.replace(temporary,path)
                entries.append(dict(slot=slot,view=view,path=str(path),shape=list(array.shape),sha256=sha(path)))
            assert len(entries)<=2160
            write(WREADY.parent/'FORWARD_JOURNAL.json',dict(at=now(),case_forwards=len(entries),maximum=2160,atoms=entries))
        del image,changed,output
    assert len(entries)==2160 and len({(a['slot'],a['view']) for a in entries})==2160
    write(WREADY,dict(at=now(),status='complete',model_id='w32e171_guided',base_model_id='w32e171',atom_source='guided_input_w32',
        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_path=str(CHECKPOINT),architecture='sammed3d.t1ce_wt_capg3d.AutomaticPromptLocator',
        construction=dict(center_aware=False,init_filters=32,dropout=.1),input_recipe=spec,input_recipe_path=model_info['input_recipe_path'],
        input_recipe_sha256=model_info['input_recipe_sha256'],input_source_sha256=sha(CORE),
        manifest_sha256=legacy.TEST_SHA,index_sha256=root['data']['index_sha256'],cache_path=str(legacy.NATIVE),canvas=list(canvas),
        case_forwards=2160,new_model_forwards=2160,batch_calls=calls,seconds=time.monotonic()-started,atoms=entries))
    for a in entries:Path(a['path']).chmod(0o444)
    for folder in (legacy.FTMP/'models/w32e171_guided/atoms').iterdir():folder.chmod(0o555)
    (legacy.FTMP/'models/w32e171_guided/atoms').chmod(0o555)

def score(candidate_id):
    root,candidate=selection(candidate_id)
    for model in candidate['models']:
        if model['model_id']=='w32e171_guided':
            spec=validate_input_model(model);ready=read(model['atoms_ready_path'])
            assert ready['input_recipe']==spec and ready['input_recipe_sha256']==model['input_recipe_sha256']
            assert ready['input_source_sha256']==sha(CORE) and ready['new_model_forwards']==2160
    return previous.score(candidate_id)

def watch():
    verify_entry();exclusive(E/'FINAL_WATCH_START.json',dict(at=now(),pid=os.getpid(),worker=1154121,physical_gpu=1,automatic_neural_retry=False))
    while not (STAGE/'FINAL_TEST_SELECTION.json').exists():guard();time.sleep(30)
    root,_=selection();choices=[c for c in root['candidates'] if c['kind']=='fusion'];assert len(choices)<=1
    if not choices:return
    candidate=choices[0];folder=F/'candidates'/candidate['candidate_id']
    if not (folder/'RAW_TEST270.json').exists():
        if any(m['model_id']=='w32e171_guided' for m in candidate['models']) and not WREADY.exists():
            subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.edge_input_b.final270','forward-guided'],check=True)
        elif any(m['model_id']=='w32e171' for m in candidate['models']) and not legacy.WREADY.exists():
            subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.final270','forward-w32'],check=True)
        while not all(Path(m['atoms_ready_path']).exists() for m in candidate['models']):guard();time.sleep(10)
        subprocess.run([sys.executable,'-u','-m','scripts.race_v2_inference_b_20261029.edge_input_b.final270','score','--candidate-id',candidate['candidate_id']],check=True)
    selection(candidate['candidate_id']);previous.audit_and_publish(candidate)
    write(E/'FINAL_WATCH_STATUS.json',dict(at=now(),state='complete',candidate_id=candidate['candidate_id'],raw_sha256=sha(F/'RAW_FUSION.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('watch','forward-guided','score'));p.add_argument('--candidate-id');a=p.parse_args()
    try:
        if a.stage=='watch':watch()
        elif a.stage=='forward-guided':forward_guided()
        else:assert a.candidate_id;score(a.candidate_id)
    except BaseException as error:
        import traceback
        write(E/f'FINAL_FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
