"""Equal subject mass with original relative scan weights inside each subject."""
from collections import defaultdict
import torch


def normalize_subject_weights(case_ids,weights,case_to_subject):
    weights=torch.as_tensor(weights,dtype=torch.float64,device='cpu')
    if weights.ndim!=1 or len(case_ids)!=len(weights) or not len(weights):
        raise ValueError('nonempty one-dimensional aligned weights required')
    if not torch.isfinite(weights).all() or not (weights>0).all():
        raise ValueError('weights must be finite and positive')
    if len(set(case_ids))!=len(case_ids):raise ValueError('duplicate eligible case IDs')
    subjects=[case_to_subject[c] for c in case_ids]
    totals=defaultdict(float)
    for subject,weight in zip(subjects,weights.tolist()):totals[subject]+=weight
    result=torch.tensor([w/totals[s] for s,w in zip(subjects,weights.tolist())],dtype=torch.float64)
    if not torch.isfinite(result).all() or not (result>0).all():raise ValueError('normalization invalid')
    return result
