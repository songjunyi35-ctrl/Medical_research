import pytest
from sammed3d.automatic_workflow_search import views_for, search_controller, select_workflow, scoped_rows


def table():
    return {k:[{'case_id':'case','subject_id':'subject','dice':.75+.00001*k,'point_hit':True}]
            for k in range(128)}


def test_space_unique_and_canonical():
    assert len({views_for(k) for k in range(128)}) == 128
    assert views_for(0)==(0,) and views_for(127)==tuple(range(8))
    assert views_for(64)==(0,7) and views_for(11)==(0,1,2,4)


@pytest.mark.parametrize('name',['exhaustive','random','greedy','uct'])
def test_budget_anchors_determinism_and_selector(name):
    values=table()
    a=search_controller(name,values,33)
    assert a==search_controller(name,values,33)
    assert a['query_order'][:2]==[0,127]
    assert len(a['query_order'])==len(set(a['query_order']))
    assert a['unique_queries']==(128 if name=='exhaustive' else 16)
    assert len(a['selector_pool'])==8 and set(a['selector_pool'])<=set(a['query_order'])
    selected=select_workflow(a['selector_pool'],{k:values[k] for k in a['selector_pool']})
    assert selected['selected']==0


def test_scope_removes_other_patients():
    values=table()
    for rows in values.values():
        rows.append({'case_id':'forbidden','subject_id':'other','dice':0.,'point_hit':False})
    selected=scoped_rows(values,{'subject'})
    assert all(len(v)==1 and v[0]['case_id']=='case' for v in selected.values())


def test_no_selector_access_outside_pool():
    with pytest.raises(ValueError):
        select_workflow([0,127,1,2,3,4,5,6],table())


def test_fallback_if_point_floor_infeasible():
    pool=[0,127,1,2,3,4,5,6]
    rows={k:[{'case_id':str(i),'subject_id':str(i),'dice':.1 if k==0 else .8,
              'point_hit':k==0} for i in range(3)] for k in pool}
    out=select_workflow(pool,rows)
    assert out['fallback'] and out['selected']==127 and out['feasible']==[]


@pytest.mark.parametrize('value',[-1,128,True,1.0])
def test_bad_id(value):
    with pytest.raises(ValueError): views_for(value)
