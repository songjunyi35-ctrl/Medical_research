from scripts.evaluate_image_blur import performance_gates,select_arm
from scripts import image_blur_protocol as p


def delta(point=0,scan=.005,subject=.005):
    return {'point_delta':point,'scan_delta':scan,'subject_delta':subject,'subject_95ci':[-.002,.01],'scan_cluster_95ci':[-.002,.01]}


def test_native_point_gate_not_averaged_away():
    assert not all(performance_gates(delta(point=-3),delta(),False).values())
    assert all(performance_gates(delta(point=-1),delta(),False).values())


def test_subject_must_improve_own_control_too():
    assert not all(performance_gates(delta(),delta(point=-1),True).values())
    assert not all(performance_gates(delta(),delta(scan=0),True).values())
    assert all(performance_gates(delta(),delta(),True).values())


def test_single_recipe_and_failed_seed_not_dropped():
    r={f'{s}_subject_blur':{'subject_dice':.75} for s in p.SEEDS}
    c={k:{'versus_native':delta()} for k in r}
    g={k:{'pass':True} for k in r}
    assert select_arm(r,c,g)[1]=='subject_blur'
    g['2026090711_subject_blur']['pass']=False
    assert select_arm(r,c,g)[1] is None


def test_average_gain_required():
    r={f'{s}_subject_blur':{'subject_dice':.75} for s in p.SEEDS}
    c={k:{'versus_native':delta(scan=.0029)} for k in r}
    g={k:{'pass':True} for k in r}
    assert select_arm(r,c,g)[1] is None
