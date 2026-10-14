"""Deterministic set-medoid among preselected fold policies, no scores accepted."""
from sammed3d.automatic_workflow_search import views_for


def medoid(workflows):
    values=list(workflows)
    if len(values)!=3:raise ValueError('exactly three fixed fold workflows required')
    sets=[set(views_for(k)) for k in values]
    candidates=sorted(set(values))
    distances={k:sum(len(set(views_for(k)) ^ s) for s in sets) for k in candidates}
    chosen=min(candidates,key=lambda k:(distances[k],len(views_for(k)),k))
    return {'workflow':chosen,'views':views_for(chosen),'fold_workflows':values,
            'symmetric_difference_totals':distances,'tie_break':'view_count_then_workflow_id'}
