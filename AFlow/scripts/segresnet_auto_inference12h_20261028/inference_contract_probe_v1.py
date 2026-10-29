"""Actual selected-model state invariance on one synthetic image, no GT."""
import hashlib,inspect
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.deploy_group_v1 import FrozenGroupPredictor,export_recipe


def digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str((tensor.dtype,tuple(tensor.shape))).encode());h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def main():
    guard();inc=read(R/'INCUMBENT.json');folder=ROOT/Path(inc['result_path']).parent;recipe=R/'CONTRACT_PROBE_RECIPE.json';export_recipe(folder,recipe)
    torch.cuda.set_per_process_memory_fraction(.8);predictor=FrozenGroupPredictor(recipe);model=predictor.model
    frozen=all(not p.requires_grad for p in model.parameters());evaluation=all(not m.training for m in model.modules());assert frozen and evaluation
    before=digest(model);x=torch.ones((1,1,*CANVAS));values=predictor.predict(x,[dict(canvas_offset=[0,0,0],extent=list(CANVAS))]);after=digest(model)
    assert before==after and len(values)==1 and values[0]['mask'].dtype==bool and values[0]['mask'].shape==CANVAS
    names=list(inspect.signature(type(predictor).predict).parameters);assert names==['self','images','geometries']
    write(R/'INFERENCE_CONTRACT_PROBE.json',dict(at=now(),state='passed',candidate=folder.name,source_result_sha256=sha(folder/'RESULT.json'),checkpoint_sha256=CHECKPOINT_SHA,parameters_require_grad_false=frozen,all_modules_evaluation_mode=evaluation,model_state_including_buffers_sha256_before=before,model_state_including_buffers_sha256_after=after,model_parameters_and_buffers_unchanged=True,API_input_parameters=names,GT_or_case_ID_input=False,patient_access=False,synthetic_images=1,training=False,source_sha256=sha(Path(__file__)),scope='Selected inference implementation state invariance on synthetic input plus prior real-cohort equivalence; not independent clinical validation'))
    print(dict(state='passed',candidate=folder.name,parameters_and_buffers_unchanged=True,patient_access=False))

if __name__=='__main__':main()
