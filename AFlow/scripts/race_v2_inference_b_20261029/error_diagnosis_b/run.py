"""GT post-hoc error geometry only: never a deployable predictor or new search."""
import numpy as np
from scripts.race_v2_inference_b_20261029.common import *
from scripts.race_v2_inference_b_20261029.analyze import read_atoms,metrics,paired
D=R/'error_diagnosis_b'

def count(mask,target):
    tp=int((mask&target).sum());fp=int((mask&~target).sum());fn=int((~mask&target).sum());den=2*tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,locator_dsc=float(np.float32(2*tp)/np.float32(den)) if den else 1.)

def unpack(record):
    path=Path(record['prediction_path']);assert sha(path)==record['prediction_sha256']
    saved=np.load(path,allow_pickle=False);shape=tuple(saved['shape'])
    return np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)

def main():
    guard();exclusive(D/'DIAGNOSTIC_ACCESS.json',dict(at=now(),scope='GT post-hoc diagnostic on validation108 full crops only',
        deployment=False,GT_oracle_not_achievable_claim=True,new_neural_forwards=0,test_access=False))
    cache=TMP/'shared_validation/native';index,images,labels=data(cache);cohort=rows();digest=sha(cache/'index.json')
    sources=[read_atoms(A/'ATOMS_READY.json',RACE_SHA,digest),read_atoms(OUT/'ATOMS_READY.json',CHECKPOINT_SHA,digest)]
    old=read(R/'RAW_VALIDATION.json')['per_configuration']['race0.75_logit_raw']
    cov_recipe=read(R/'innovation_b/SELECTED_RECIPE.json')['candidate'];cov=read(R/'innovation_b/RAW_VALIDATION.json')['per_configuration'][cov_recipe['id']]
    contra_recipe=read(R/'image_conflict_b/DEPLOYMENT_RECIPE.json')['candidate'];contra=read(R/'image_conflict_b/RAW_VALIDATION.json')['per_configuration'][contra_recipe['id']]
    records=[];oracle_records={'mean_expert_convex_hull':[],'bounded_alpha_05_095':[],'all16_view_convex_hull':[]}
    begun=time.monotonic();confidence=float(np.log(9))
    for slot,row in enumerate(cohort):
        guard();sl=slices(index['geometry'][row['case_id']]);target=np.array(labels[slot][sl]>0);image=np.array(images[slot][sl]);means=[];ranges=[]
        for source in sources:
            views=[]
            for view in range(8):
                atom=source[slot,view];path=Path(atom['path']);assert sha(path)==atom['sha256']
                array=np.load(path,allow_pickle=False);assert array.dtype==np.float32 and array.shape==target.shape
                assert np.isfinite(array).all();views.append(array)
            means.append(sum(views)/8);ranges.append((np.minimum.reduce(views),np.maximum.reduce(views)))
        lo=np.minimum(ranges[0][0],ranges[1][0]);hi=np.maximum(ranges[0][1],ranges[1][1]);a,b=means
        base=a*.75+b*.25;masks={'fixed075':base>0,'covariance':unpack(cov[slot]),'contralateral':unpack(contra[slot])}
        assert count(masks['fixed075'],target)=={k:old[slot][k] for k in ('tp','fp','fn','locator_dsc')}
        mean_lo=np.minimum(a,b);mean_hi=np.maximum(a,b)
        bounded_lo=np.minimum(.5*a+.5*b,.95*a+.05*b);bounded_hi=np.maximum(.5*a+.5*b,.95*a+.05*b)
        allwrong=np.where(target,hi<=0,lo>0);meanswrong=np.where(target,mean_hi<=0,mean_lo>0)
        boundedwrong=np.where(target,bounded_hi<=0,bounded_lo>0)
        strongwrong=np.where(target,hi<=-confidence,lo>=confidence)
        categories={
            'all16_wrong_confident09':allwrong&strongwrong,
            'all16_wrong_not_all_confident09':allwrong&~strongwrong,
            'mean_experts_wrong_but_some_view_correct':meanswrong&~allwrong,
            'mean_expert_conflict_but_alpha_bound_cannot_correct':~meanswrong&boundedwrong,
            'bounded_alpha_can_correct':~boundedwrong}
        for key,wrong in [('mean_expert_convex_hull',meanswrong),('bounded_alpha_05_095',boundedwrong),('all16_view_convex_hull',allwrong)]:
            # GT picks each voxel's best allowable side. This is only an optimistic mathematical bound.
            oracle_mask=np.where(wrong,~target,target)
            oracle_records[key].append(dict(case_id=row['case_id'],subject_id=row['subject_id'],**count(oracle_mask,target)))
        rec=dict(case_id=row['case_id'],subject_id=row['subject_id'],crop_voxels=int(target.size),GT_voxels=int(target.sum()),GT_outside_image_nonzero=int((target&(image==0)).sum()),methods={})
        for name,mask in masks.items():
            error=mask!=target;parts={kind:dict(fn=int((error&target&region).sum()),fp=int((error&~target&region).sum())) for kind,region in categories.items()}
            assert sum(v['fn'] for v in parts.values())==int((~mask&target).sum())
            assert sum(v['fp'] for v in parts.values())==int((mask&~target).sum())
            rec['methods'][name]=dict(**count(mask,target),error_partition=parts,
                FN_outside_image_nonzero=int((~mask&target&(image==0)).sum()),
                errors_within_model_mirror_sign_instability=int((error&(((ranges[0][0]<=0)&(ranges[0][1]>0))|((ranges[1][0]<=0)&(ranges[1][1]>0)))).sum()))
        base_error=masks['fixed075']!=target
        rec['changes_vs_fixed075']={name:dict(corrected=int((base_error&(mask==target)).sum()),introduced=int((~base_error&(mask!=target)).sum()),
            by_original_error_partition={kind:dict(corrected_fn=int((base_error&target&region&mask).sum()),corrected_fp=int((base_error&~target&region&~mask).sum())) for kind,region in categories.items()}) for name,mask in masks.items() if name!='fixed075'}
        records.append(rec);write(D/'HEARTBEAT.json',dict(at=now(),completed=slot+1,total=108,new_model_forwards=0,seconds=time.monotonic()-begun))
        del views,means,ranges,masks,image,target,base,lo,hi,a,b
    categories=list(records[0]['methods']['fixed075']['error_partition'])
    def aggregate(selected):
        result={}
        for method in ('fixed075','covariance','contralateral'):
            values=[r['methods'][method] for r in selected];fn=sum(r['fn'] for r in values);fp=sum(r['fp'] for r in values)
            result[method]=dict(fn=fn,fp=fp,total_error_voxels=fn+fp,FN_outside_image_nonzero=sum(r['FN_outside_image_nonzero'] for r in values),
                partition={key:dict(fn=sum(r['error_partition'][key]['fn'] for r in values),fp=sum(r['error_partition'][key]['fp'] for r in values)) for key in categories},
                errors_within_model_mirror_sign_instability=sum(r['errors_within_model_mirror_sign_instability'] for r in values))
        return result
    subjects=sorted({r['subject_id'] for r in records},key=lambda sid:hashlib.sha256(('race_v2_20261029:'+sid).encode()).hexdigest());fold={sid:i%5 for i,sid in enumerate(subjects)}
    write(D/'RAW_DIAGNOSTIC.json',dict(at=now(),per_case=records,GT_oracle_records=oracle_records,
        full_crop_FN_FP_accounting=True,never_deployable=True,new_model_forwards=0,test_access=False))
    result=dict(at=now(),state='complete',scans=108,subjects=50,aggregate=aggregate(records),
        folds=[dict(fold=f,subjects=10,aggregate=aggregate([r for r in records if fold[r['subject_id']]==f])) for f in range(5)],
        optimistic_GT_oracle_dice={key:metrics(value) if False else dict(scan_dice=float(np.mean([r['locator_dsc'] for r in value])),subject_dice=float(np.mean([np.mean([r['locator_dsc'] for r in value if r['subject_id']==sid]) for sid in subjects]))) for key,value in oracle_records.items()},
        GT_positive_total=sum(r['GT_voxels'] for r in records),GT_outside_image_nonzero=sum(r['GT_outside_image_nonzero'] for r in records),
        seconds=time.monotonic()-begun,full_crop_FN_FP_accounting=True,new_model_forwards=0,test_access=False,
        warning='GT post-hoc per-voxel oracle is only a convex-hull error bound, not a deployable algorithm, attainable estimate, model score, or patient-level GT route. No methods selected using these oracle masks.')
    write(D/'RESULT.json',result)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback
        write(D/f'FAILURE_{time.time_ns()}.json',dict(at=now(),error=repr(error),traceback=traceback.format_exc()));raise
