"""One root-registered final snapshot; fixed decoder transfer, no new search/test.

The watcher is quiet and does no GPU work while waiting. It never reads latest,
never accepts an intermediate epoch, and never retries a patient network pass.
"""
import argparse,fcntl,subprocess,time
import numpy as np
import torch
from scripts.race_v2_inference_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.predictor import forward_view,decode_all
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

QUEUE=STAGE/'SNAPSHOT108_QUEUE'
FREEZE=R/'SNAPSHOT108_FREEZE.json'
ARCHITECTURES={'sammed3d.race_seg.RaceSeg','scripts.race_v2_20261029.model.RaceBenefitSeg'}

def freeze():
    assert not FREEZE.exists();selected=read(R/'SELECTED_RECIPE.json')
    assert read(R/'DONE.json')['state']=='validation108_complete'
    files=[Path(__file__),ROOT/'scripts/race_v2_inference_a_20261029/predictor.py',ROOT/'scripts/race_v2_inference_a_20261029/common.py',ROOT/'scripts/race_v2_inference_a_20261029/statistics.py',ROOT/'sammed3d/race_seg.py',ROOT/'scripts/race_v2_20261029/model.py']
    write(FREEZE,dict(at=now(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},initial_execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),selected_recipe_path=str(R/'SELECTED_RECIPE.json'),selected_recipe_sha256=sha(R/'SELECTED_RECIPE.json'),initial_plan_sha256=sha(R/'PLAN.json'),allowed_architectures=sorted(ARCHITECTURES),registered_job_directory=str(QUEUE),jobs_maximum=1,patient_forwards_per_job=864,required_training_terminal_state='complete and new_epoch==target_new_epochs',references=['raw_identity_t05','equal8_logit_t05','legacy_full'],transferred_selected_id=selected['candidate']['id'],new_search=False,new270_access=False,worker=1154121,physical_gpu=0,root_registration_schema=dict(job_id='safe basename',registered_by='root',approval_scope='final_completed_training_snapshot108_only',run_id='exact handoff run_id',handoff_path='absolute immutable HANDOFF.json',handoff_sha256='SHA256',training_done_path='absolute final DONE.json',training_done_sha256='SHA256',registered_at='root timestamp')))

def verify_snapshot():
    verify();f=read(FREEZE)
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,p
    assert sha(R/'EXECUTION_FREEZE.json')==f['initial_execution_freeze_sha256']
    assert sha(R/'PLAN.json')==f['initial_plan_sha256']
    assert sha(f['selected_recipe_path'])==f['selected_recipe_sha256']
    return f

def load_handoff(job):
    assert job['registered_by']=='root' and job['approval_scope']=='final_completed_training_snapshot108_only'
    assert job['job_id'] and set(job['job_id'])<=set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
    handoff_path=Path(job['handoff_path']);assert sha(handoff_path)==job['handoff_sha256']
    h=read(handoff_path);assert h['run_id']==job['run_id'] and h['architecture'] in ARCHITECTURES
    assert h['strict_cpu_reload'] is True and h['deployment_labels_or_case_lookup'] is False and h['test_pixels'] is False
    done_path=Path(job['training_done_path']);assert done_path.parent==handoff_path.parent
    assert sha(done_path)==job['training_done_sha256'];done=read(done_path)
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']
    assert done['new_updates']==done['new_epoch']*122
    assert h['total_stage_new_epochs']==done['new_epoch'] and h['total_stage_new_updates']==done['new_updates']
    assert 0<h['actual_new_epochs']<=done['new_epoch'],'no new network for epoch0 source reuse'
    assert h['actual_new_updates']==h['actual_new_epochs']*122
    assert h['source_epoch']==164 and h['source_checkpoint_sha256']==CHECKPOINT_SHA and h['optimizer_new'] is True
    assert sha(handoff_path.parent/'EXECUTION_FREEZE.json')==h['execution_freeze_sha256']
    assert sha(handoff_path.parent/'RECIPE.json')==h['recipe_sha256']
    geometry=h['input_geometry'];assert geometry['validation_canvas']==list(CANVAS) and geometry['spacing_mm']==[1,1,1]
    assert 'T@R@T' in geometry['mirror'] and 'valid support' in geometry['mirror']
    expected=dict(width=32,dropout=.1,dim=16,enabled=True)
    assert h['construction']==expected
    path=Path(h['checkpoint_path']);assert path.name not in ('best.pth','latest.pth')
    assert path.parent==handoff_path.parent/'checkpoints' and path.name.startswith('race_v2_fixed_')
    assert sha(path)==h['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['new_epoch']==h['actual_new_epochs'] and state['new_updates']==h['actual_new_updates']
    if h['architecture']=='scripts.race_v2_20261029.model.RaceBenefitSeg':
        from scripts.race_v2_20261029.model import RaceBenefitSeg
        model=RaceBenefitSeg(**expected)
    else:
        from sammed3d.race_seg import RaceSeg
        model=RaceSeg(**expected)
    model.load_state_dict(state['model_state_dict'],strict=True)
    assert all(torch.equal(v,state['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in model.state_dict().items())
    cohort=rows();assert [(r['case_id'],r['subject_id']) for r in h['validation']['per_case']]==sorted((r['case_id'],r['subject_id']) for r in cohort)
    return h,model.eval().requires_grad_(False)

def process(job):
    f=verify_snapshot();h,model=load_handoff(job);job_id=job['job_id'];folder=R/'snapshots'/job_id;scratch=TMP/'snapshots'/job_id
    assert not folder.exists(),'no implicit snapshot network retry'
    folder.mkdir(parents=True);scratch.mkdir(parents=True)
    write(folder/'JOB_FREEZE.json',dict(at=now(),registration=job,handoff=h,snapshot_freeze_sha256=sha(FREEZE),selected_recipe_sha256=f['selected_recipe_sha256'],split='validation108',maximum_case_forwards=864,no_new_search=True,no_test_access=True))
    lock_path=Path('/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock');lock=lock_path.open('a')
    while True:
        guard()
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(2)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    from scripts.followup_test270_common import numeric_mode
    numeric_mode();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.95)
    write(folder/'CPU_STRICT_RELOAD.json',dict(at=now(),status='passed',checkpoint_sha256=h['checkpoint_sha256'],architecture=h['architecture'],construction=h['construction'],GPU_lock=str(lock_path),GPU_uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()))
    model=model.cuda();synthetic=torch.randn(8,1,*CANVAS,device='cuda')
    g=dict(canvas_offset=[0,0,0],extent=list(CANVAS),reference_enabled=True,reflection=[[-1,0,0,159],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    with torch.inference_mode():
        full=forward_view(model,synthetic,[g]*8,7,CANVAS);tail=forward_view(model,synthetic[:4],[g]*4,0,CANVAS)
    reserved=torch.cuda.max_memory_reserved()/2**30;assert reserved<=38
    write(folder/'GPU_SYNTHETIC_GATE.json',dict(at=now(),status='passed',batch=8,tail=4,canvas=list(CANVAS),peak_reserved_gib=reserved,patient_forwards=0))
    del synthetic,full,tail;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    cache_ready=read(R/'CACHE_READY.json');assert cache_ready['all_four_payloads_exact'] and sha(CACHE/'index.json')==cache_ready['index_sha256']
    index=read(CACHE/'index.json');assert index['case_ids']==[r['case_id'] for r in rows()] and tuple(index['canvas'])==CANVAS
    images=np.memmap(CACHE/'images.f32',np.float32,'r',shape=(108,*CANVAS));atoms=[];started=time.monotonic()
    with (folder/'MODEL_PREACCESS.json').open('x') as out:json.dump(dict(at=now(),scans=108,subjects=50,views=list(range(8)),maximum_case_forwards=864,checkpoint_sha256=h['checkpoint_sha256'],training_completed=True,new_test_access=False),out)
    for first in range(0,108,8):
        guard();slots=list(range(first,min(108,first+8)));image=torch.from_numpy(np.array(images[slots],np.float32))[:,None].cuda()
        geometries=[{k:index['geometry'][index['case_ids'][slot]][k] for k in ('canvas_offset','extent','reflection','reference_enabled')} for slot in slots]
        for view in range(8):
            values=forward_view(model,image,geometries,view,CANVAS)
            for slot,z in zip(slots,values):
                path=scratch/'atoms'/f'{slot:03}'/f'flip{view}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+'.tmp')
                with temporary.open('xb') as out:np.save(out,z,allow_pickle=False)
                os.replace(temporary,path);path.chmod(0o444)
                atoms.append(dict(slot=slot,view=view,path=str(path),shape=list(z.shape),sha256=sha(path)))
            write(folder/'FORWARD_JOURNAL.json',dict(at=now(),atoms=atoms,case_forwards=len(atoms),maximum=864))
            write(folder/'HEARTBEAT.json',dict(at=now(),stage='final_snapshot_validation108',case_forwards=len(atoms),maximum=864))
        del image,values
    assert len(atoms)==864 and len({(a['slot'],a['view']) for a in atoms})==864
    assert sha(h['checkpoint_path'])==h['checkpoint_sha256']
    write(folder/'ATOMS_READY.json',dict(at=now(),status='passed',run_id=h['run_id'],checkpoint_sha256=h['checkpoint_sha256'],checkpoint_path=h['checkpoint_path'],architecture=h['architecture'],construction=h['construction'],actual_new_epochs=h['actual_new_epochs'],actual_new_updates=h['actual_new_updates'],input_geometry=h['input_geometry'],case_ids=index['case_ids'],manifest_sha256=sha(ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/validation_t1ce_wt.jsonl'),index_sha256=sha(CACHE/'index.json'),atoms=atoms,case_forwards=864,seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,snapshot_freeze_sha256=sha(FREEZE)))
    del model;torch.cuda.empty_cache();fcntl.flock(lock,fcntl.LOCK_UN)
    plan=read(R/'PLAN.json');recipes=plan['recipes'][:3];selected=read(R/'SELECTED_RECIPE.json')['candidate']
    if selected['id'] not in {r['id'] for r in recipes}:recipes.append(selected)
    results={r['id']:[] for r in recipes};labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*CANVAS));lookup={(a['slot'],a['view']):a for a in atoms};cohort=rows()
    for slot,row in enumerate(cohort):
        guard();sl=slices(index['geometry'][row['case_id']]);logits=[]
        for view in range(8):
            a=lookup[slot,view];assert sha(a['path'])==a['sha256'];logits.append(np.load(a['path'],allow_pickle=False))
        outputs=decode_all(logits,np.array(images[slot][sl],np.float32),recipes);target=np.array(labels[slot][sl]>0)
        for recipe,out in zip(recipes,outputs):
            mask=out.pop('mask');tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
            results[recipe['id']].append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,configuration_id=recipe['id'],locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,tp=tp,fp=pred-tp,fn=positive-tp,automatic_point_hits_gt=bool(target[tuple(out['automatic_point'])]),**out))
    previous=read(R/'RAW_VALIDATION108.json')['per_configuration'];summary={}
    for recipe in recipes:
        records=results[recipe['id']];base=previous[recipe['id']];right={r['case_id']:r for r in base}
        summary[recipe['id']]=dict(metrics=summary_ci(records),paired_vs_same_recipe_RACE164=paired(records,base),worse_cases=sorted([dict(case_id=r['case_id'],subject_id=r['subject_id'],delta=r['locator_dsc']-right[r['case_id']]['locator_dsc']) for r in records if r['locator_dsc']<right[r['case_id']]['locator_dsc']],key=lambda r:r['delta']),patient_folds={str(fold):summary_ci([r for r in records if plan['folds'][r['subject_id']]==fold]) for fold in range(5)})
    write(folder/'RAW_VALIDATION108.json',dict(at=now(),run_id=h['run_id'],per_configuration=results,recipes=recipes,fixed_decoder_transfer=True,new_search=False,new_test_access=False,model_forwards=864))
    write(folder/'RESULT.json',dict(at=now(),state='complete',run_id=h['run_id'],checkpoint_sha256=h['checkpoint_sha256'],recipes=recipes,summary=summary,selected_transfer_id=selected['id'],neural_case_forwards=864,full_stage_training_complete=True,new_test_access=False,selection_on_this_snapshot=False,independent_confirmation=False,test_tuned_history=True))
    return folder

def watch():
    f=verify_snapshot();QUEUE.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    while True:
        guard();jobs=sorted(QUEUE.glob('*.json'));assert len(jobs)<=f['jobs_maximum']
        if jobs:
            p=jobs[0];job=read(p);folder=R/'snapshots'/job['job_id']
            if (folder/'RESULT.json').exists():return
            try:process(job)
            except BaseException as e:
                import traceback
                write(R/'SNAPSHOT108_FAILURE.json',dict(at=now(),job_path=str(p),error=repr(e),traceback=traceback.format_exc(),automatic_neural_retry=False));raise
            return
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.watch:watch()
    else:p.error('choose --freeze or --watch')
