import pytest

from scripts.describe_subject_sampling_point_transitions import transitions


def row(case, hit, dice=.5):
    return {'case_id': case, 'subject_id': case, 'point_hit': hit, 'dice': dice,
            'target_voxels': 10, 'predicted_voxels': 10, 'point_flat_index': 0}


def test_all_four_categories_and_dice_point_divergence():
    control = [row('a', True), row('b', True), row('c', False), row('d', False)]
    candidate = [row('a', True), row('b', False, .8), row('c', True), row('d', False)]
    result = transitions(control, candidate)
    assert result['counts'] == dict.fromkeys(['kept_hit', 'lost_hit', 'gained_hit', 'kept_miss'], 1)
    assert result['net_point_delta'] == 0
    assert result['lost_point_cases_dice_improved'] == 1


def test_pair_mismatch_rejected():
    with pytest.raises(ValueError):
        transitions([row('a', True)], [row('b', False)])
