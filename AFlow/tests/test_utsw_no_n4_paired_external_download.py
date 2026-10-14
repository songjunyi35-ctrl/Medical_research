from scripts import utsw_no_n4_paired_external_protocol as p
from scripts.download_utsw_no_n4_paired_external import expected_paths


def test_expected_download_paths_are_exact_and_collision_free():
    expected = expected_paths(p.cohort())
    assert len(expected) == 724
    assert sum(expected.values()) == 1_813_086_599
    assert all(name.endswith(("brain_t1ce.nii.gz", "rtumorseg_manual_correction.nii.gz")) for name in expected)
    assert len({name.split("/", 1)[0] for name in expected}) == 362
