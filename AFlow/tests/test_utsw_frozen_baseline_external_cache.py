from pathlib import Path


def test_builder_freezes_canonical_n4_label_domain_and_fixed_three():
    text=Path("scripts/build_utsw_frozen_baseline_external_cache.py").read_text()
    assert "use_n4=True" in text and '{0.,1.,2.,4.}' in text
    assert '("BT0001","BT0002","BT0003")' in text
    assert 'uses_gt_for_localization":False' in text
