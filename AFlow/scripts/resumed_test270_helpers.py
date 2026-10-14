"""Same frozen test270 metrics, prospective stricter resumed-window joint gates."""
from scripts.weak_elastic_test270_helpers import (TEST,TEST_SHA,CACHE,INDEX_SHA,IMAGES_SHA,LABELS_SHA,LEADER,LEADER_SHA,OVERALL_SCAN,OVERALL_SUBJECT,SINGLE_SCAN,BOOTSTRAP_SEED,rows_from_logits,summarize,leader_rows,compare)


def gates(results,contrasts,roles):
    candidate=results[roles['candidate_flip8']];control=results[roles['control_flip8']]
    c=contrasts['candidate_flip8_vs_old_overall']
    return {'candidate_overall_scan_lead':candidate['scan_dice']>OVERALL_SCAN,
            'candidate_identity_1x_scan_lead':results[roles['candidate_identity']]['scan_dice']>SINGLE_SCAN,
            'candidate_flip8_above_matched_control':candidate['scan_dice']>control['scan_dice'],
            'candidate_subject_above_matched_control':candidate['subject_dice']>control['subject_dice'],
            'subject_above_old_overall':candidate['subject_dice']>OVERALL_SUBJECT,
            'point_at_least265':candidate['point_hits']>=265,
            'point_at_least_matched_control':candidate['point_hits']>=control['point_hits'],
            'scan_cluster_ci_positive':c['scan_cluster_95ci'][0]>0,
            'subject_ci_positive':c['subject_95ci'][0]>0}
