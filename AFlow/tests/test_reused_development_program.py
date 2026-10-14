import pytest

from scripts.summarize_reused_development_program import weighting_reference


def test_equal_volumes_align_within_patient_reference():
    ids = list('abcd')
    subjects = {'a': 'x', 'b': 'x', 'c': 'x', 'd': 'y'}
    result = weighting_reference(ids, subjects, dict.fromkeys(ids, 100))
    assert result['reference_only_not_a_new_training_policy']
    assert result['policies']['subject_balanced']['total_variation_from_patient_and_scan_uniform'] == pytest.approx(0, abs=1e-12)
    assert result['policies']['control']['total_variation_from_patient_and_scan_uniform'] > 0


def test_unequal_within_patient_volumes_not_exact_metric_weights():
    result = weighting_reference(['a', 'b'], {'a': 'x', 'b': 'x'}, {'a': 100, 'b': 10000})
    assert result['policies']['subject_balanced']['total_variation_from_patient_and_scan_uniform'] > 0
    assert result['policies']['subject_balanced']['probability_relative_to_reference_max'] > 1
