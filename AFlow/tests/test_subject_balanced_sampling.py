import pytest
import torch
from sammed3d.subject_balanced_sampling import normalize_subject_weights


def test_subject_mass_equal_and_scan_ratios_preserved():
    ids=['a1','a2','b1','c1','c2','c3']
    mapping={c:c[0] for c in ids}
    original=torch.tensor([.5,2.,4.,1.,2.,3.],dtype=torch.float64)
    output=normalize_subject_weights(ids,original,mapping)
    for s in 'abc':
        slots=[i for i,c in enumerate(ids) if c[0]==s]
        torch.testing.assert_close(output[slots].sum(),torch.tensor(1.,dtype=torch.float64),rtol=0,atol=1e-12)
        torch.testing.assert_close(output[slots]/output[slots][0],original[slots]/original[slots][0],rtol=0,atol=1e-12)
    torch.testing.assert_close(output/output.sum(),output/3,rtol=0,atol=1e-12)


def test_single_scan_subjects_have_equal_weights_and_deterministic_draws():
    ids=['a','b','c'];out=normalize_subject_weights(ids,[.5,1,4],{s:s for s in ids})
    assert torch.equal(out,torch.ones(3,dtype=torch.float64))
    a=torch.multinomial(out,972,replacement=True,generator=torch.Generator().manual_seed(11))
    b=torch.multinomial(out,972,replacement=True,generator=torch.Generator().manual_seed(11))
    assert torch.equal(a,b)


@pytest.mark.parametrize('weights',[[0,1],[-1,1],[float('nan'),1],[float('inf'),1]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):normalize_subject_weights(['a','b'],weights,{'a':'a','b':'b'})


def test_missing_subject_and_duplicate_case_rejected():
    with pytest.raises(KeyError):normalize_subject_weights(['a'],[1],{})
    with pytest.raises(ValueError):normalize_subject_weights(['a','a'],[1,2],{'a':'a'})
