from scripts.describe_blur_case_tradeoffs import category


def test_exhaustive_sign_partition():
    assert category(0,0)=='count_tie'
    assert all(category(tp,fp)=='pareto_better' for tp,fp in [(1,0),(0,-1),(1,-1)])
    assert all(category(tp,fp)=='pareto_worse' for tp,fp in [(-1,0),(0,1),(-1,1)])
    assert category(1,1)=='expansion' and category(-1,-1)=='contraction'
