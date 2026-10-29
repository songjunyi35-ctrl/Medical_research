"""Normalized deployment entry including explicit equal8 reference fallback."""
import torch
from scripts.race_v2_inference_innovation_a_20261029 import algorithm

@torch.inference_mode()
def deploy(logits,image,recipe):
    prototype=logits[0];device=prototype.device if torch.is_tensor(prototype) else 'cpu'
    values=[torch.as_tensor(z,dtype=torch.float32,device=device) for z in logits]
    assert len(values)==8 and all(tuple(z.shape)==tuple(image.shape) for z in values)
    if recipe.get('mode')=='equal' or recipe.get('kind')=='reference_equal8':
        score=algorithm.equal8(values);point=list(map(int,torch.unravel_index(score.argmax(),score.shape)))
        return dict(mask=score>0,logits=score,automatic_point=point,base_automatic_point=point,diagnostic={})
    if recipe.get('mode')=='race8_local_dependency':recipe=recipe['config']
    assert recipe['kind']=='race_same_model8_local_dependency'
    return algorithm.deploy(values,image,recipe)
