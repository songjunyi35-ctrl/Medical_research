from scripts.resumed_test270_helpers import gates


def fixture():
    result={'candidate':{'scan_dice':.78,'subject_dice':.77,'point_hits':265},'identity':{'scan_dice':.76},'control':{'scan_dice':.76,'subject_dice':.75,'point_hits':264}}
    c={'candidate_flip8_vs_old_overall':{'scan_cluster_95ci':[.001,.04],'subject_95ci':[.002,.05]}}
    role={'candidate_flip8':'candidate','candidate_identity':'identity','control_flip8':'control'}
    return result,c,role


def test_strict_new_gate():
    r,c,p=fixture();assert all(gates(r,c,p).values())
    r['candidate']['point_hits']=264;assert not gates(r,c,p)['point_at_least265']


def test_no_compensating_subject_or_point_loss():
    r,c,p=fixture();r['control']['point_hits']=266
    assert not gates(r,c,p)['point_at_least_matched_control']
    r['control']['subject_dice']=.78
    assert not gates(r,c,p)['candidate_subject_above_matched_control']


def test_ci_zero_is_not_positive_support():
    r,c,p=fixture();c['candidate_flip8_vs_old_overall']['scan_cluster_95ci'][0]=0
    assert not gates(r,c,p)['scan_cluster_ci_positive']
