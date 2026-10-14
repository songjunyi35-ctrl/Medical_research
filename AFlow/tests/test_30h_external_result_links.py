import pytest

from scripts.audit_30h_external_result_links import recompute


def summary(with_counts=True):
    rows = []
    for case, subject, inter in [('a', 'x', 10), ('b', 'x', 10), ('c', 'y', 0)]:
        row = {'case_id': case, 'subject_id': subject, 'locator_dsc': inter / 10,
               'target_wt_voxels': 10, 'predicted_wt_voxels': 10, 'automatic_point_hits_gt': inter > 0}
        if with_counts:
            row['intersection_voxels'] = inter
        rows.append(row)
    return {'per_case': rows, 'cases': 3, 'locator_dsc_mean': 2 / 3,
            'locator_subject_dsc_mean': .5, 'automatic_point_hits': 2}


def test_patient_mean_and_available_count_scope():
    assert recompute(summary())['dice_recomputed_from_intersection_rows'] == 3
    result = recompute(summary(False))
    assert result['subject_dice'] == .5
    assert result['dice_recomputed_from_intersection_rows'] == 0


def test_tampered_intersection_rejected():
    data = summary()
    data['per_case'][0]['intersection_voxels'] = 9
    with pytest.raises(AssertionError):
        recompute(data)


def test_tampered_point_total_rejected():
    data = summary()
    data['automatic_point_hits'] = 3
    with pytest.raises(AssertionError):
        recompute(data)
