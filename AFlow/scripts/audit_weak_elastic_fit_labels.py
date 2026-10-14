"""One predeclared deformation per fit label, no model/GPU/development arrays."""
from datetime import datetime,timezone
import json
import resource
import time
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from sammed3d.weak_elastic import dense_field,field_grid,displacement_bound
from scripts import weak_elastic_protocol as p
from scripts.train_t1ce_wt_capg3d import write_json_atomic


def main():
    p.check_freeze();fit,_=p.check_roles()
    out=p.RECORD/'FIT_AUGMENTATION_AUDIT.json';pre=p.RECORD/'FIT_AUGMENTATION_AUDIT_PREACCESS.json'
    if out.exists() or pre.exists():raise FileExistsError('one fixed diagnostic only')
    p.checked(p.CACHE/'index.json',p.INDEX_SHA)
    index=json.loads((p.CACHE/'index.json').read_text())
    mapping={case:i for i,case in enumerate(index['case_ids'])}
    cache=json.loads((p.RECORD/'AUX_CACHE_RECEIPT.json').read_text())
    p.checked(p.CACHE/'labels.u8',cache['source_labels_sha256'])
    p.write_new(pre,{'created_utc':datetime.now(timezone.utc).isoformat(),'fit_sha256':p.FIT_SHA,'index_sha256':p.INDEX_SHA,'labels_sha256':cache['source_labels_sha256'],'files':{name:p.sha(p.ROOT/name) for name in ['scripts/audit_weak_elastic_fit_labels.py','sammed3d/weak_elastic.py','archive/research_records/2026-09-10_weak_elastic/FIT_AUGMENTATION_AUDIT_IDEA.md']},'seed_rule':'2026091041 + zero-based fit ordinal','stress_probability':1,'formal_probability_unchanged':.2,'new_model_forwards':0})
    labels=np.memmap(p.CACHE/'labels.u8',mode='r',dtype=np.uint8,shape=(972,128,128,128))
    torch.set_num_threads(2);results=[];start=time.monotonic()
    try:
        for ordinal,row in enumerate(fit):
            target=np.array(labels[mapping[row['case_id']]],dtype=np.uint8)
            assert np.isin(target,[0,1]).all() and target.sum()>0
            components,n=ndimage.label(target,structure=np.ones((3,3,3)))
            sizes=np.bincount(components.ravel(),minlength=n+1)
            coeff=torch.rand(1,3,7,7,7,generator=torch.Generator().manual_seed(2026091041+ordinal))*2-1
            field=dense_field(coeff,(128,128,128),'cpu');bound=float(displacement_bound(field).max());amplitude=float(field.abs().max())
            assert bound<=.250001 and amplitude<=2.000001
            assert all(field.select(axis,side).count_nonzero()==0 for axis in (2,3,4) for side in (0,-1))
            grid=field_grid(field)
            warped=F.grid_sample(torch.from_numpy(components.astype(np.float32))[None,None],grid,mode='nearest',padding_mode='zeros',align_corners=False)[0,0].numpy().astype(np.int32)
            counts=np.bincount(warped.ravel(),minlength=n+1)
            missing=[i for i in range(1,n+1) if counts[i]==0]
            pred=warped>0;nt=int(target.sum());npred=int(pred.sum());assert npred>0
            inter=int((pred & (target>0)).sum())
            results.append({'case_id':row['case_id'],'subject_id':row['subject_id'],'original_voxels':nt,'warped_voxels':npred,'overlap_dice_not_model_score':2*inter/(nt+npred),'relative_volume_change':(npred-nt)/nt,'components':n,'missing_component_sizes':[int(sizes[i]) for i in missing],'missing_component_original_voxels':int(sizes[missing].sum()) if missing else 0,'max_axis_displacement_voxels':amplitude,'gradient_bound':bound})
            if len(results)%32==0:
                write_json_atomic(p.RECORD/'FIT_AUGMENTATION_AUDIT_PROGRESS.json',{'completed':len(results),'total':684,'seconds':time.monotonic()-start});print(len(results),flush=True)
        missing=[size for row in results for size in row['missing_component_sizes']]
        summary={'scans':len(results),'subjects':len({r['subject_id'] for r in results}),'mean_deformed_label_overlap_not_model_dice':float(np.mean([r['overlap_dice_not_model_score'] for r in results])),'minimum_deformed_label_overlap':min(r['overlap_dice_not_model_score'] for r in results),'absolute_relative_volume_change_percentiles':dict(zip(['p50','p95','max'],np.quantile([abs(r['relative_volume_change']) for r in results],[.5,.95,1]).tolist())),'original_components':sum(r['components'] for r in results),'missing_component_count':len(missing),'missing_component_voxels':sum(missing),'largest_missing_component':max(missing,default=0),'affected_cases':sum(bool(r['missing_component_sizes']) for r in results),'original_wt_voxels':sum(r['original_voxels'] for r in results),'all_whole_wt_retained':True,'max_gradient_bound':max(r['gradient_bound'] for r in results)}
        raw={'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'summary':summary,'per_case':results,'preaccess_sha256':p.sha(pre),'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_model_forwards':0,'access':{'fit_labels':684,'fit_images':False,'development':False,'test':False,'confirmation':False,'outer':False,'external':False},'does_not_change_formal_protocol':True}
        p.write_new(out,raw);print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        p.write_new(p.RECORD/'FIT_AUGMENTATION_AUDIT_FAILURE.json',{'status':'failed','error':repr(error),'completed_rows':results,'attempted_case':fit[len(results)]['case_id'] if len(results)<684 else None,'new_model_forwards':0});raise


if __name__=='__main__':main()
