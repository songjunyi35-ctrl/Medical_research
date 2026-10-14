import copy
from scripts.evaluate_fractional_occupancy import pairing,performance_gates


def reports():
    c={'initial_state_sha256':'init','sampling_weight_sha256':'weights','export_max_abs_error':0,'history':[{'steps':121,'sampler_sha256':str(i),'augmentation_rng_sha256':str(i),'first_batch':{'image_sha256':str(i),'labels_sha256':'hard'}} for i in range(168)]}
    a=copy.deepcopy(c)
    for row in a['history']:row['first_batch']['labels_sha256']='soft'
    return a,c


def test_only_labels_may_change_in_pairing():
    a,c=reports();assert all(pairing(a,c).values())
    a['history'][3]['first_batch']['image_sha256']='changed'
    assert not pairing(a,c)['same_first_batch_images']


def test_noop_target_is_not_admitted_and_point_gate_preserved():
    a,c=reports();assert not pairing(c,c)['target_labels_actually_changed']
    d={'subject_delta':.004,'scan_delta':.004,'point_delta':-1,'subject_95ci':[-.002,.01],'scan_cluster_95ci':[-.002,.01]}
    assert all(performance_gates(d).values())
    d['point_delta']=-2
    assert not all(performance_gates(d).values())
