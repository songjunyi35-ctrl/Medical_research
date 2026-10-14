from scripts.evaluate_utsw_frozen_baseline_external import bootstrap,reference_gates,summarize


def rows(dice=.7,point=True):return [{"case_id":f"BT{i:04d}","subject_id":f"BT{i:04d}","dice":dice,"point_hit":point,"predicted_voxels":10,"target_voxels":12} for i in range(362)]


def test_summary_and_bootstrap_are_deterministic():
    assert summarize(rows())["scan_dice"]==summarize(rows())["subject_dice"]==.7
    a=bootstrap(rows(.7),rows(.71));b=bootstrap(rows(.7),rows(.71));assert a==b
    lo,hi=a["scan_delta_gcsam_minus_kd_95ci"];assert abs(lo-.01)<1e-12 and abs(hi-.01)<1e-12


def test_reference_gates_use_only_frozen_primary_views():
    s={"full972_kd168":{"identity":summarize(rows(.1,False)),"flip8":summarize(rows(.7,True))},
       "gcsam184":{"identity":summarize(rows(.7,True)),"flip8":summarize(rows(.1,False))}}
    assert all(reference_gates(s).values())
