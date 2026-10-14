from scripts.evaluate_t1ce_wt_gamma_aug_stage_b_validation import promotion_gates

def s(scan,sub,hits):return {"locator_dsc_mean":scan,"locator_subject_dsc_mean":sub,"automatic_point_hit_rate":hits/108}
def c(lo=-0.002):return {"ema_minus_raw":{"scan_95ci":[lo,0.01],"subject_95ci":[lo,0.01]}}

def test_all_eleven_pass():
    g=promotion_gates(s(.74,.73,103),s(.742,.732,104),s(.741,.731,103),c(),c(),True)
    assert len(g)==11 and all(g.values())

def test_strict_point_ci_and_engineering_fail():
    g=promotion_gates(s(.74,.73,104),s(.741,.731,102),s(.741,.731,103),c(-.0031),c(-.0031),False)
    assert not g["beats_archived_scan"] and not g["beats_archived_subject"] and not g["point_not_lower_control"] and not g["point_at_least_103"] and not g["scan_ci_vs_control"] and not g["engineering_pairing"]
