from scripts.publish_training_study_result import interpret


def fixture(interval):
    return {'metrics':{'seed_arm':{'scan_dice':.75}},'comparisons':{'seed_arm':{'versus_own_sampling_control':{'scan_delta':.001,'subject_delta':.002,'point_delta':-3,'scan_cluster_95ci':interval,'subject_95ci':interval}}},'gates':{'seed_arm':{'point_safety':False,'scan_positive':True}}}


def test_positive_point_estimate_not_called_interval_support():
    r=interpret(fixture([-.003,.005]))['seed_arm']
    assert '不确定性' in r['dice_interval_interpretation'] and len(r['failed_gates'])==1


def test_positive_and_negative_interval_interpretation():
    assert '均为正' in interpret(fixture([.001,.005]))['seed_arm']['dice_interval_interpretation']
    assert '整体为负' in interpret(fixture([-.005,-.001]))['seed_arm']['dice_interval_interpretation']
