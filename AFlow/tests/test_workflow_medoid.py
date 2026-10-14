import pytest
from sammed3d.workflow_medoid import medoid


def test_majority_and_no_performance_input():
    assert medoid([64,16,16])['workflow']==16
    assert medoid([48,96,20])['workflow']==48


def test_identical():
    assert medoid([127,127,127])['views']==tuple(range(8))


@pytest.mark.parametrize('values',[[],[1],[1,2],[1,2,3,4],[1,2,128]])
def test_invalid(values):
    with pytest.raises(ValueError):medoid(values)
