import torch
from scripts.weak_elastic_test270_helpers import rows_from_logits,gates,OVERALL_SCAN,OVERALL_SUBJECT,SINGLE_SCAN


def test_test_row_keeps_historical_float32_rounding_and_exact_counts():
    x=torch.tensor([1.,1.,1.,-1.]).reshape(1,1,1,2,2)
    y=torch.tensor([1.,0.,0.,0.]).reshape_as(x)
    r=rows_from_logits(x,y,{'case_id':['a'],'subject_id':['s']})[0]
    assert r['dice']==.5 and r['exact_count_dice']==.5
    assert r['point_flat_index']==0 and r['point_hit']


def test_point_estimate_lead_does_not_imply_ci_gate():
    results={'full_elastic_flip8':{'scan_dice':OVERALL_SCAN+.001,'subject_dice':OVERALL_SUBJECT+.001,'point_hits':265},'full_control_flip8':{'scan_dice':OVERALL_SCAN},'full_elastic_identity':{'scan_dice':SINGLE_SCAN+.001}}
    contrasts={'candidate_flip8_vs_old_overall':{'scan_cluster_95ci':[-.001,.002],'subject_95ci':[-.002,.003]}}
    g=gates(results,contrasts)
    assert g['candidate_overall_scan_lead'] and g['candidate_identity_1x_scan_lead']
    assert not g['scan_cluster_ci_nonnegative'] and not g['subject_ci_nonnegative']
