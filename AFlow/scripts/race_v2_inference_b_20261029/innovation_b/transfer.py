"""Transfer exactly one selected covariance recipe to one final108 snapshot."""
import argparse
import numpy as np
import torch
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import metrics,paired,read_atoms,worsened
from scripts.race_v2_inference_a_20261029.statistics import summary_ci
from scripts.race_v2_inference_b_20261029.snapshot_partner import HISTORICAL,select
from scripts.race_v2_inference_b_20261029.innovation_b.core import deploy

I=R/'innovation_b';S=I/'final_snapshot'

def verify_transfer():
    verify()
    for source,digest in read(I/'TRANSFER_ENTRY_FREEZE.json')['source_sha256'].items():assert sha(ROOT/source)==digest,source
    assert sha(I/'SELECTED_RECIPE.json')==read(I/'TRANSFER_ENTRY_FREEZE.json')['selected_recipe_sha256']

def final_package(handoff_path,registration_path):
    handoff_path=Path(handoff_path);handoff=read(handoff_path);done=read(handoff_path.with_name('DONE.json'))
    budget=read(handoff_path.with_name('BUDGET_DECISION.json'));job=read(registration_path)
    assert job['registered_by']=='root' and job['approval_scope']=='final_completed_training_snapshot108_only'
    assert job['handoff_sha256']==sha(handoff_path) and job['run_id']==handoff['run_id']
    assert job['training_done_sha256']==sha(handoff_path.with_name('DONE.json'))
    assert done['state']=='complete' and done['new_epoch']==done['target_new_epochs']==budget['new_epochs']
    assert handoff['total_stage_new_epochs']==done['new_epoch'] and handoff['total_stage_new_updates']==done['new_updates']==done['new_epoch']*122
    assert handoff['strict_cpu_reload'] and handoff['source_epoch']==164 and handoff['source_checkpoint_sha256']==RACE_SHA
    assert handoff['actual_new_epochs']>0,'zero-new-best must reuse old RACE, never generate duplicate snapshot'
    assert handoff['actual_new_updates']==handoff['actual_new_epochs']*122
    assert handoff['input_geometry']['validation_canvas']==list(CANVAS)
    assert 'latest' not in Path(handoff['checkpoint_path']).name
    assert sha(handoff['checkpoint_path'])==handoff['checkpoint_sha256']
    assert sha(handoff_path.with_name('EXECUTION_FREEZE.json'))==handoff['execution_freeze_sha256']
    assert sha(handoff_path.with_name('RECIPE.json'))==handoff['recipe_sha256']
    cohort=rows();lookup={r['case_id']:r for r in handoff['validation']['per_case']}
    assert len(lookup)==108 and all(lookup[row['case_id']]['subject_id']==row['subject_id'] for row in cohort)
    current=np.array([lookup[row['case_id']]['locator_dsc'] for row in cohort]);options=[]
    for name,source in HISTORICAL.items():
        previous={r['case_id']:r for r in read(source['report'])['per_case']}
        options.append(dict(id=name,values=np.array([previous[row['case_id']]['locator_dsc'] for row in cohort]),
            checkpoint_sha256=source['checkpoint_sha256'],atoms_ready=str(source['ready']),
            atoms_ready_sha256=sha(source['ready']),report_sha256=sha(source['report'])))
    partner=select(current,options,range(108))
    return handoff,dict(at=now(),handoff_sha256=sha(handoff_path),registration_sha256=sha(registration_path),
        candidate_checkpoint_sha256=handoff['checkpoint_sha256'],**partner,
        fixed_recipe=read(I/'SELECTED_RECIPE.json')['candidate'],fixed_recipe_sha256=sha(I/'SELECTED_RECIPE.json'),
        frozen_global_partner_rule='same prior .03 performance eligibility/.01 correlation gap; one global partner',
        new_grid_search=False,new_model_forwards=0)

@torch.inference_mode()
def main(handoff_path,registration_path,candidate_ready):
    verify_transfer();handoff,package=final_package(handoff_path,registration_path)
    preaccess=S/'FIXED_TRANSFER_PREACCESS.json'
    if preaccess.exists():
        previous=read(preaccess)
        for key in ('handoff_sha256','registration_sha256','candidate_checkpoint_sha256','fixed_recipe','fixed_recipe_sha256'):
            assert previous[key]==package[key],key
        assert previous['selected_partner']==package['selected_partner']
    else:exclusive(preaccess,package)
    lock=gpu_lock();numeric_mode();cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows()
    partner=package['selected_partner'];index_sha=sha(cache/'index.json')
    sources=[read_atoms(candidate_ready,handoff['checkpoint_sha256'],index_sha),
        read_atoms(partner['atoms_ready'],partner['checkpoint_sha256'],index_sha)]
    recipe=package['fixed_recipe'];records=[]
    for slot,row in enumerate(cohort):
        guard();sl=slices(index['geometry'][row['case_id']]);image=np.array(images[slot][sl],np.float32);loaded=[]
        for source in sources:
            views=[]
            for view in range(8):
                entry=source[slot,view];path=Path(entry['path']);assert sha(path)==entry['sha256']
                array=np.load(path,allow_pickle=False);assert array.dtype==np.float32 and array.shape==image.shape and list(array.shape)==entry['shape']
                views.append(torch.from_numpy(array).cuda())
            loaded.append(views)
        output=deploy(loaded[0],loaded[1],image,recipe);mask=output['mask'];point=output['automatic_point']
        # All deployment decisions have finished before GT enters this scorer.
        target=np.array(labels[slot][sl]>0);tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        path=OUT/'innovation_b/final_snapshot/predictions'/f'{slot:03}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            saved=np.load(path,allow_pickle=False);old=np.unpackbits(saved['mask_bits'])[:mask.size].reshape(mask.shape)
            assert np.array_equal(old,mask) and saved['point'].tolist()==point
        else:
            with path.open('xb') as handle:np.savez_compressed(handle,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=row['case_id'],subject_id=row['subject_id'],source_slot=slot,
            tp=tp,fp=pred-tp,fn=positive-tp,locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.,
            automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),
            prediction_path=str(path),prediction_sha256=sha(path),prediction_shape=list(mask.shape)))
        write(S/'HEARTBEAT.json',dict(at=now(),completed=slot+1,total=108,configurations=1,new_model_forwards=0))
    write(S/'RAW_VALIDATION.json',dict(at=now(),per_configuration={recipe['id']:records},
        checkpoint_sha256=handoff['checkpoint_sha256'],partner_model_id=partner['id'],new_model_forwards=0,test_access=False))
    legacy=read(R/'RAW_VALIDATION.json')['per_configuration'];old_innovation=read(I/'RAW_VALIDATION.json')['per_configuration'][recipe['id']]
    references={'race_equal8':legacy['race_equal8'],'old_best_fusion':legacy['race0.75_logit_raw'],'old_selected_innovation':old_innovation}
    write(S/'RESULT.json',dict(at=now(),state='complete',checkpoint_sha256=handoff['checkpoint_sha256'],
        partner_model_id=partner['id'],recipe=recipe,metrics=summary_ci(records),fixed_transfer=package,
        comparisons={name:paired(records,reference) for name,reference in references.items()},
        all_worsened={name:worsened(records,reference) for name,reference in references.items()},
        configuration_count=1,new_model_forwards=0,per_case_deployment_forwards=16,test_access=False,
        interpretation='One previously selected formula transferred without12-grid rerun; partner uses all108 raw report; conditional reused-validation evidence only'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--handoff',required=True);parser.add_argument('--registration',required=True);parser.add_argument('--candidate-ready',required=True);args=parser.parse_args()
    try:main(args.handoff,args.registration,args.candidate_ready)
    except BaseException as error:
        import traceback
        write(S/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc(),new_model_forwards=0));raise
