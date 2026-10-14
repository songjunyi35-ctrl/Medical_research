"""Frozen test270 constants and patient-cluster summaries, no data access on import."""
from collections import defaultdict
import statistics
import torch
from scripts import weak_elastic_full864_protocol as p
from scripts.evaluate_structured_background_aux import bootstrap
from scripts.train_t1ce_wt_capg3d import dice_per_case

TEST=p.ROOT/'data/datasets/brats2024_t1ce_va_hdfs_20260812_v1/sealed_test/test_cases_t1ce.jsonl'
TEST_SHA='a5ddc0ed2c0b269431758b14061da3c6163dd0c48133b60a2ef8efd7f23aa8f9'
CACHE=p.ROOT/'workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/test270'
INDEX_SHA='41bd6cf44553b8d156de49f2c57bb424cab60cfa26c06881bdfb96ff91de39d3'
IMAGES_SHA='cf834d2b4e6cdaa414689c8723d532178980f6ad459324d13a903321f1064888'
LABELS_SHA='ecb28a45607d54cf453ece3208ef8261ea30f496e747f919242ac1172445f222'
LEADER=p.ROOT/'workspace/BraTS/sammed3d_t1ce_wt/flip8_tta_stability_20260904/FLIP8_TEST270.json'
LEADER_SHA='29efc557aa12b6f82fd079edd84cc3c8c1e5ad4536880c016b664932234f45ad'
OVERALL_SCAN=.7572696616942132
OVERALL_SUBJECT=.7412535853511771
SINGLE_SCAN=.7520031755617648
BOOTSTRAP_SEED=2026091032


def rows_from_logits(logits,labels,batch):
    dice=dice_per_case(logits,labels)
    result=[]
    for i,case in enumerate(batch['case_id']):
        pred=logits[i,0]>0;target=labels[i,0]>.5
        point=int(logits[i,0].flatten().argmax());inter=int((pred&target).sum());npred=int(pred.sum());ntarget=int(target.sum());assert ntarget>0
        result.append({'case_id':case,'subject_id':batch['subject_id'][i],'dice':float(dice[i]),'exact_count_dice':2*inter/(npred+ntarget),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':ntarget})
    return result


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['subject_id']].append(r['dice'])
    assert len(rows)==270 and len(groups)==119
    return {'scans':270,'subjects':119,'scan_dice':statistics.fmean(r['dice'] for r in rows),'subject_dice':statistics.fmean(statistics.fmean(v) for v in groups.values()),'point_hits':sum(r['point_hit'] for r in rows),**bootstrap(rows,BOOTSTRAP_SEED),'per_case':rows}


def leader_rows(report):
    rows=[{'case_id':r['case_id'],'subject_id':r['subject_id'],'dice':r['locator_dsc'],'point_hit':r['automatic_point_hits_gt']} for r in report['flip8']['per_case']]
    summary=summarize(rows)
    assert summary['scan_dice']==OVERALL_SCAN and summary['subject_dice']==OVERALL_SUBJECT and summary['point_hits']==265
    return summary


def compare(a,b):
    delta=[]
    for left,right in zip(a['per_case'],b['per_case'],strict=True):
        assert (left['case_id'],left['subject_id'])==(right['case_id'],right['subject_id'])
        delta.append({'case_id':left['case_id'],'subject_id':left['subject_id'],'dice':right['dice']-left['dice']})
    return {'scan_delta':b['scan_dice']-a['scan_dice'],'subject_delta':b['subject_dice']-a['subject_dice'],'point_delta':b['point_hits']-a['point_hits'],**bootstrap(delta,BOOTSTRAP_SEED)}


def gates(results,contrasts,roles=None):
    roles=roles or {'candidate_flip8':'full_elastic_flip8','candidate_identity':'full_elastic_identity','control_flip8':'full_control_flip8'}
    candidate=results[roles['candidate_flip8']];control=results[roles['control_flip8']];c=contrasts['candidate_flip8_vs_old_overall']
    return {'candidate_overall_scan_lead':candidate['scan_dice']>OVERALL_SCAN,
            'candidate_identity_1x_scan_lead':results[roles['candidate_identity']]['scan_dice']>SINGLE_SCAN,
            'candidate_flip8_above_matched_control':candidate['scan_dice']>control['scan_dice'],
            'subject_at_least_old_overall':candidate['subject_dice']>=OVERALL_SUBJECT,
            'point_at_least264':candidate['point_hits']>=264,
            'scan_cluster_ci_nonnegative':c['scan_cluster_95ci'][0]>=0,
            'subject_ci_nonnegative':c['subject_95ci'][0]>=0}
