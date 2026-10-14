import numpy as np
import torch
from sammed3d.automatic_fixed_workflow import fuse_atoms
from sammed3d.automatic_workflow_search import views_for
from sammed3d.workflow_score_audit import numpy_workflow_score
from scripts.collect_automatic_workflow_space import score


def test_independent_numpy_all128_matches_torch():
    rng=np.random.default_rng(21)
    atoms=rng.normal(size=(8,4,5,6)).astype(np.float32)
    truth=rng.integers(0,2,size=(4,5,6)).astype(bool)
    tensors=[torch.from_numpy(a)[None,None] for a in atoms]
    for k in range(128):
        expected=numpy_workflow_score(atoms,truth,k)
        actual=score(fuse_atoms(tensors,views_for(k)),torch.from_numpy(truth)[None,None],
                     {'case_id':['a'],'subject_id':['s']})[0]
        assert expected=={name:actual[name] for name in expected}
