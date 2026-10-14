from scripts.evaluate_weak_elastic import performance_gates,select_arm
from scripts import weak_elastic_protocol as p


def delta(point=0,scan=.005,subject=.005):
    return {'point_delta':point,'scan_delta':scan,'subject_delta':subject,'subject_95ci':[-.002,.01],'scan_cluster_95ci':[-.002,.01]}


def test_native_point_gate_not_averaged_away():
    assert not all(performance_gates(delta(point=-3),delta(),False).values())
    assert all(performance_gates(delta(point=-1),delta(),False).values())


def test_subject_must_improve_own_control_too():
    assert not all(performance_gates(delta(),delta(point=-1),True).values())
    assert not all(performance_gates(delta(),delta(scan=0),True).values())
    assert all(performance_gates(delta(),delta(),True).values())


def fixture():
    r={f'{s}_{a}':{'subject_dice':.75} for s in p.SEEDS for a in p.ARMS}
    c={k:{'versus_native':delta()} for k in r}
    g={k:{'pass':True} for k in r}
    return r,c,g


def test_tie_native_and_failed_seed_not_dropped():
    r,c,g=fixture();assert select_arm(r,c,g)[1]=='native_elastic'
    g['2026090711_native_elastic']['pass']=False
    assert select_arm(r,c,g)[1]=='subject_elastic'
    g['2026090712_subject_elastic']['pass']=False
    assert select_arm(r,c,g)[1] is None


def test_mean_scan_gate_and_subject_ranking():
    r,c,g=fixture()
    for s in p.SEEDS:r[f'{s}_subject_elastic']['subject_dice']=.76
    assert select_arm(r,c,g)[1]=='subject_elastic'
    for s in p.SEEDS:c[f'{s}_subject_elastic']['versus_native']['scan_delta']=.0029
    assert select_arm(r,c,g)[1]=='native_elastic'
