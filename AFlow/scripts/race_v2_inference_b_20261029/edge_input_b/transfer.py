"""One globally selected guided input+decoder transferred to final RACE primary."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import read_atoms,paired,worsened
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
from scripts.race_v2_inference_b_20261029.edge_input_b.deployment import deploy
E=R/'edge_input_b';S=E/'final_snapshot'

def verify_transfer():
    verify();frozen=read(E/'TRANSFER_ENTRY_FREEZE.json')
    for path,digest in frozen['source_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(E/'DEPLOYMENT_RECIPE.json')==frozen['selected_recipe_sha256']
    assert sha(E/'INPUT_DEPLOYMENT_RECIPE.json')==frozen['selected_input_recipe_sha256']

def package(handoff_path,registration_path):
    handoff_path=Path(handoff_path);handoff=read(handoff_path);done=read(handoff_path.with_name('DONE.json'));budget=read(handoff_path.with_name('BUDGET_DECISION.json'));job=read(registration_path)
    assert job['registered_by']=='root' and job['approval_scope']=='final_completed_training_snapshot108_only'
    assert job['handoff_sha256']==sha(handoff_path) and job['run_id']==handoff['run_id']
    assert job['training_done_sha256']==sha(handoff_path.with_name('DONE.json'))
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']==budget['new_epochs']
    assert handoff['total_stage_new_epochs']==done['new_epoch'] and handoff['total_stage_new_updates']==done['new_updates']==done['new_epoch']*122
    assert handoff['strict_cpu_reload'] and handoff['source_epoch']==164 and handoff['source_checkpoint_sha256']==RACE_SHA
    assert handoff['actual_new_epochs']>0 and handoff['actual_new_updates']==handoff['actual_new_epochs']*122
    assert handoff['input_geometry']['validation_canvas']==list(CANVAS) and 'latest' not in Path(handoff['checkpoint_path']).name
    assert sha(handoff['checkpoint_path'])==handoff['checkpoint_sha256']
    assert sha(handoff_path.with_name('EXECUTION_FREEZE.json'))==handoff['execution_freeze_sha256']
    assert sha(handoff_path.with_name('RECIPE.json'))==handoff['recipe_sha256']
    recipe=read(E/'DEPLOYMENT_RECIPE.json')['candidate'];input_recipe=read(E/'INPUT_DEPLOYMENT_RECIPE.json')['input_recipe']
    assert recipe['input_spec']==input_recipe
    return handoff,dict(at=now(),handoff_sha256=sha(handoff_path),registration_sha256=sha(registration_path),
        checkpoint_sha256=handoff['checkpoint_sha256'],fixed_recipe=recipe,partner_model_id='w32e171_guided',
        partner_checkpoint_sha256=CHECKPOINT_SHA,partner_ready_path=str(E/input_recipe['id']/'ATOMS_READY.json'),
        partner_ready_sha256=sha(E/input_recipe['id']/'ATOMS_READY.json'),input_recipe=input_recipe,
        input_recipe_sha256=sha(E/'INPUT_DEPLOYMENT_RECIPE.json'),new_parameter_search=False,new_model_forwards=0)

@torch.inference_mode()
def main(handoff_path,registration_path,candidate_ready):
    verify_transfer();handoff,p=package(handoff_path,registration_path);exclusive(S/'FIXED_TRANSFER_PREACCESS.json',p)
    lock=gpu_lock();numeric_mode();cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows();digest=sha(cache/'index.json')
    sources=[read_atoms(candidate_ready,handoff['checkpoint_sha256'],digest),read_atoms(p['partner_ready_path'],CHECKPOINT_SHA,digest)]
    recipe=p['fixed_recipe'];records=[]
    for slot,row in enumerate(cohort):
        guard();sl=slices(index['geometry'][row['case_id']]);image=np.array(images[slot][sl],np.float32);loaded=[]
        for source in sources:
            views=[]
            for view in range(8):
                atom=source[slot,view];path=Path(atom['path']);assert sha(path)==atom['sha256'];value=np.load(path,allow_pickle=False)
                assert list(value.shape)==atom['shape'] and value.shape==image.shape and value.dtype==np.float32;views.append(torch.from_numpy(value).cuda())
            loaded.append(views)
        output=deploy(*loaded,image,recipe);mask=output['mask'];point=output['automatic_point']
        target=np.array(labels[slot][sl]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        path=OUT/'edge_input_b/final_snapshot/predictions'/f'{slot:03}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:np.savez_compressed(handle,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,tp=tp,fp=pred-tp,fn=positive-tp,
            locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),
            prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape)))
        write(S/'HEARTBEAT.json',dict(at=now(),completed=slot+1,total=108,configurations=1,new_model_forwards=0))
    write(S/'RAW_VALIDATION.json',dict(at=now(),per_configuration={recipe['id']:records},checkpoint_sha256=handoff['checkpoint_sha256'],partner_model_id='w32e171_guided',new_model_forwards=0,test_access=False))
    old=read(R/'RAW_VALIDATION.json')['per_configuration'];cov_recipe=read(R/'innovation_b/SELECTED_RECIPE.json')['candidate'];contra=read(R/'image_conflict_b/DEPLOYMENT_RECIPE.json')['candidate']
    references=dict(race_equal8=old['race_equal8'],old_best_fixed075=old['race0.75_logit_raw'],old_selected_covariance=read(R/'innovation_b/RAW_VALIDATION.json')['per_configuration'][cov_recipe['id']],old_selected_contralateral=read(R/'image_conflict_b/RAW_VALIDATION.json')['per_configuration'][contra['id']],old_selected_guided=read(E/'RAW_VALIDATION.json')['per_configuration'][recipe['id']])
    write(S/'RESULT.json',dict(at=now(),state='complete',checkpoint_sha256=handoff['checkpoint_sha256'],partner_model_id='w32e171_guided',recipe=recipe,
        metrics=summary_ci(records),fixed_transfer=p,comparisons={name:paired(records,value) for name,value in references.items()},all_worsened={name:worsened(records,value) for name,value in references.items()},new_model_forwards=0,new_parameter_search=False,test_access=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--handoff',required=True);parser.add_argument('--registration',required=True);parser.add_argument('--candidate-ready',required=True);a=parser.parse_args()
    try:main(a.handoff,a.registration,a.candidate_ready)
    except BaseException as error:
        import traceback
        write(S/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc()));raise
