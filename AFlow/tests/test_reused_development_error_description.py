import pytest

from scripts.describe_reused_development_errors import describe_pair, summarize_counts


def row(case='a', subject='patient', target=10, pred=10, inter=5):
    return {'case_id': case, 'subject_id': subject, 'target_voxels': target,
            'predicted_voxels': pred, 'intersection': inter,
            'dice': 2 * inter / (pred + target), 'point_hit': True}


def test_empty_prediction_is_retained_without_dividing_by_prediction():
    result = summarize_counts([row(pred=0, inter=0)])
    assert result['metrics']['scan_dice'] == 0
    assert result['metrics']['mean_missed_target_fraction'] == 1
    assert result['metrics']['mean_extra_prediction_per_target'] == 0
    assert result['metrics']['empty_predictions'] == 1


def test_extra_prediction_can_exceed_one():
    result = summarize_counts([row(pred=40, inter=10)])
    assert result['metrics']['mean_extra_prediction_per_target'] == 3
    assert result['metrics']['mean_missed_target_fraction'] == 0


def test_subject_mean_not_scan_mean_for_repeated_patient():
    result = summarize_counts([row('a', 'x', inter=10), row('b', 'x', inter=10), row('c', 'y', inter=0)])
    assert result['metrics']['scan_dice'] == pytest.approx(2 / 3)
    assert result['metrics']['subject_dice'] == .5


def test_boundary_groups_and_pair_deltas_are_exhaustive():
    volumes = dict(zip('abcde', [18874, 18875, 40435, 80247, 80248]))
    controls = [row(c, c) for c in volumes]
    candidates = [row(c, c, inter=6) for c in volumes]
    result = describe_pair(controls, candidates, volumes)
    assert [result[f'native_volume_group{i}']['control']['scans'] for i in range(1, 5)] == [1, 2, 1, 1]
    delta = result['all']['candidate_minus_control']
    assert delta['scan_dice'] == pytest.approx(.1)
    assert delta['mean_missed_target_fraction'] == pytest.approx(-.1)
    assert delta['mean_extra_prediction_per_target'] == pytest.approx(-.1)
    assert result['all']['dice_changes'] == {'improved': 5, 'worse': 0, 'tied': 0}


def test_empty_group_has_no_invented_metrics():
    result = describe_pair([row()], [row()], {'a': 1})
    assert result['native_volume_group4']['control']['metrics'] is None
    assert result['native_volume_group4']['candidate_minus_control'] is None


@pytest.mark.parametrize('change', [{'intersection': 11}, {'dice': .9}, {'target_voxels': 0}])
def test_invalid_saved_counts_rejected(change):
    data = row()
    data.update(change)
    with pytest.raises(ValueError):
        summarize_counts([data])


def test_pair_identity_mismatch_rejected():
    with pytest.raises(ValueError, match='identities'):
        describe_pair([row()], [row(case='different')], {'a': 1})


def test_duplicate_cases_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        describe_pair([row(), row()], [row(), row()], {'a': 1})
