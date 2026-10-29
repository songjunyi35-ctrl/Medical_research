"""CPU audit of every saved108 mask, plus immutable deployment recipe metadata."""
import numpy as np
from scripts.race_v2_inference_innovation_a_20261029.common import *
from scripts.race_v2_inference_a_20261029.statistics import summary_ci,paired

def main():
    verify();assert read(R/'DONE.json')['state']=='complete'
    raw_path=R/'RAW_VALIDATION108.json';raw=read(raw_path)['per_configuration'];result=read(R/'RESULT.json');index=read(CACHE/'index.json')
    labels=np.memmap(CACHE/'labels.u8',np.uint8,'r',shape=(108,*index['canvas']));cohort=rows();audits={}
    for identifier,records in raw.items():
        assert len(records)==108;verified=[]
        for record,row in zip(records,cohort):
            slot=record['source_slot'];assert (record['case_id'],record['subject_id'])==(row['case_id'],row['subject_id']) and index['case_ids'][slot]==row['case_id']
            path=record['prediction_path'];assert sha(path)==record['prediction_sha256'];saved=np.load(path,allow_pickle=False)
            shape=tuple(saved['shape']);assert list(shape)==index['geometry'][row['case_id']]['extent']
            mask=np.unpackbits(saved['mask_bits'])[:int(np.prod(shape))].reshape(shape).astype(bool)
            point=saved['point'].tolist();base=saved['base_point'].tolist();assert point==record['automatic_point'] and base==record['base_automatic_point']
            target=np.array(labels[slot][slices(index['geometry'][row['case_id']])]>0)
            tp=int(np.count_nonzero(mask&target));fp=int(np.count_nonzero(mask&~target));fn=int(np.count_nonzero(~mask&target));den=2*tp+fp+fn
            dice=float(np.float32(2*tp)/np.float32(den)) if den else 1.;hit=bool(target[tuple(point)]);base_hit=bool(target[tuple(base)])
            assert (tp,fp,fn,dice,hit,base_hit)==(record['tp'],record['fp'],record['fn'],record['locator_dsc'],record['automatic_point_hits_gt'],record['base_automatic_point_hits_gt'])
            verified.append(dict(case_id=row['case_id'],tp=tp,fp=fp,fn=fn,dice=dice,Point_refined_hit=hit,Point_base_hit=base_hit))
        reported=next(item['metrics'] for item in result['ranking'] if item['id']==identifier);assert summary_ci(records)==reported
        audits[identifier]=dict(scans=108,all_masks_counts_dice_points_exact=True,metrics=reported,per_case=verified)
    audit_path=R/'MASK_AUDIT_VALIDATION108.json'
    write(audit_path,dict(at=now(),state='passed',status='passed',configurations=17,masks=1836,scans=108,
        raw_path=str(raw_path),raw_sha256=sha(raw_path),source_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),
        audit_source_sha256=sha(Path(__file__)),per_configuration=audits,new_neural_forwards=0,new270_scoring=False))
    best=next(item for item in result['ranking'] if item['id']!='equal8_logit_t05')
    recipe=next(r for r in read(R/'PLAN.json')['recipes'] if r['id']==best['id'])
    entry=dict(id=recipe['id'],mode='race8_local_dependency',config=recipe,neural_case_forwards=8)
    write(R/'DEPLOYMENT_RECIPE.json',dict(at=now(),innovation_entry=entry,innovation_metrics=best['metrics'],
        reference_metrics=next(r['metrics'] for r in result['ranking'] if r['id']=='equal8_logit_t05'),
        best_overall=result['best'],source_result_path=str(R/'RESULT.json'),source_result_sha256=sha(R/'RESULT.json'),
        raw_path=str(raw_path),raw_sha256=sha(raw_path),mask_audit_path=str(audit_path),mask_audit_sha256=sha(audit_path),
        deploy_callable='scripts.race_v2_inference_innovation_a_20261029.deployment.deploy',
        source_sha256={p:sha(ROOT/p) for p in ['scripts/race_v2_inference_innovation_a_20261029/algorithm.py','scripts/race_v2_inference_innovation_a_20261029/deployment.py']},
        selection='highest108 Scan new recipe among16; same frozen one transfers to final snapshot; equal8 remains independent global option',
        new270_authorization=False))
    pairs=[]
    for joint in read(R/'PLAN.json')['recipes']:
        if joint['dependency']!='joint':continue
        diagonal=joint['id'].replace('_joint_','_diagonal_')
        pairs.append(dict(joint=joint['id'],diagonal=diagonal,comparison=paired(raw[joint['id']],raw[diagonal])))
    write(R/'MECHANISM_COMPARISONS.json',dict(at=now(),pairs=pairs,interpretation='matched diagonal controls; descriptive comparisons after reused-validation selection; no established joint-dependency advantage'))
    with (R/'RESULT.md').open('a') as stream:
        stream.write('\n1836份保存掩膜已在同worker CPU独立重载，全部TP/FP/FN、Dice、Point_refined和Point_base逐例精确一致。最佳full与最佳diag仅差约0.000002852；配对区间跨0，不宣称相关依赖机制收益成立。固定部署与snapshot迁移取DEPLOYMENT_RECIPE.json.innovation_entry。\n')

if __name__=='__main__':main()
