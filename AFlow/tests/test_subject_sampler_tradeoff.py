import pytest

from scripts.audit_subject_sampler_tradeoff import describe


def test_equal_size_equal_volume_no_change():
    ids = ['a', 'b', 'c', 'd']
    result = describe(ids, {'a': 'x', 'b': 'x', 'c': 'y', 'd': 'y'}, dict.fromkeys(ids, 100))
    assert result['relative_scan_probability_min'] == pytest.approx(1)
    assert result['relative_scan_probability_max'] == pytest.approx(1)
    for group in result['quartiles']:
        assert group['policies']['native']['probability_mass'] == pytest.approx(.25)
        assert group['policies']['subject_equal']['probability_mass'] == pytest.approx(.25)


def test_unequal_subject_sizes_have_equal_candidate_mass():
    ids = ['a', 'b', 'c', 'd']
    result = describe(ids, {'a': 'x', 'b': 'x', 'c': 'x', 'd': 'y'}, dict.fromkeys(ids, 100))
    candidate = result['policies']['subject_equal']
    assert candidate['subject_probability_min'] == pytest.approx(.5)
    assert candidate['subject_probability_max'] == pytest.approx(.5)
    assert result['relative_scan_probability_min'] == pytest.approx(2 / 3)
    assert result['relative_scan_probability_max'] == pytest.approx(2)
    assert sum(q['policies']['subject_equal']['expected_draws_per968'] for q in result['quartiles']) == pytest.approx(968)


def test_grouping_volume_then_case_is_order_independent():
    ids = ['a', 'b', 'c', 'd']
    mapping = {'a': 'x', 'b': 'x', 'c': 'y', 'd': 'y'}
    volumes = {'a': 10, 'b': 100, 'c': 1000, 'd': 10000}
    forward = describe(ids, mapping, volumes)
    reverse = describe(ids[::-1], mapping, volumes)
    for left, right in zip(forward['quartiles'], reverse['quartiles'], strict=True):
        for key in ('quartile', 'scans', 'native_voxels_min', 'native_voxels_max'):
            assert left[key] == right[key]
        for policy in ('native', 'subject_equal'):
            for metric in ('probability_mass', 'expected_draws_per968'):
                assert left['policies'][policy][metric] == pytest.approx(
                    right['policies'][policy][metric], rel=1e-14, abs=1e-14)
