"""Compare portable group orchestration to existing scoring formulas, no network."""
from types import SimpleNamespace
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.deploy_group_v1 import FrozenGroupPredictor
from scripts.segresnet_auto_inference12h_20261028.render_v1 import forward_view
from scripts.segresnet_auto_inference12h_20261028.intensity_ensemble_v1 import probability,BASE,core_prepare,core_probability
from scripts.segresnet_auto_inference12h_20261028.phase_fusions_fast_v1 import probability as single_probability
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import single_cut


@torch.inference_mode()
def main():
    numeric_mode()
    class Identity:
        def __call__(self,x):return SimpleNamespace(segmentation_logits=x)
    shape=(16,24,32);x=torch.linspace(-2,2,int(np.prod(shape)),device='cuda').reshape(1,1,*shape);x[:,:,:2]=0;geometry=dict(canvas_offset=[0,0,0],extent=list(shape));model=Identity()
    specs={'a':dict(batch=8,gamma=.85,gain=.85,offset=.15),'b':dict(batch=8,gamma=1.15,gain=1.,offset=0.)};means={}
    for gid,spec in specs.items():
        total=torch.zeros_like(x)
        for flip in range(8):total.add_(forward_view(model,x,spec,flip).sigmoid(),alpha=.125)
        means[gid]=total[0,0]
    raw=[forward_view(model,x,{},flip)[0,0] for flip in range(8)];base=core_probability(core_prepare(raw),BASE);valid=supports(x[0,0].cpu().numpy())[0];params=dict(post_temperature=1.125,bias=.375,beta=.5,volume_scale=1.)
    probes=[]
    single=dict(kind='group_mix',group='a',group_spec=specs['a'],group_views=list(range(8)),base='base7',base_views=[0,1,3,4,5,6,7],base_view_temperature=4.,group_temperature=1.,alpha=.35)
    base7=torch.zeros_like(base)
    for flip in single['base_views']:base7.add_((raw[flip]/4).sigmoid(),alpha=1/7)
    cases=[(single,single_probability(dict(base7=base7,mean_probability=means['a']),single),15)]
    for mode in ['mean_probability','mean_logit_group_probability','case_probability','case_logit_group_probability','vote_probability','mean_plus_std']:
        recipe=dict(kind='group_mix',multi_group_fusion=True,groups=['a','b'],group_specs=specs,group_views=list(range(8)),family='probe',mode=mode,case_exponent=8.,group_temperature=1.,alpha=.35,brain_support_margin=0)
        expected=probability(dict(base=base,probs=means,valid=torch.from_numpy(valid).cuda(),aggregates={}),recipe);cases.append((recipe,expected,24))
    for recipe,expected,cost in cases:
        predictor=FrozenGroupPredictor.__new__(FrozenGroupPredictor);predictor.device=torch.device('cuda');predictor.model=model;predictor.config=dict(aggregation=recipe,decoder=params,decoder_kind='brain_domain_volume',postprocessing=[],neural_case_forwards=cost)
        result=predictor.predict(x,[geometry])[0];codes=(expected.clamp(0,1)*65535+.5).long().cpu().numpy();cut=single_cut(np.bincount(codes[valid],minlength=65536),params);mask=(codes>=cut)&valid;point=list(map(int,np.unravel_index(int(expected.argmax()),expected.shape)))
        assert np.array_equal(result['mask'],mask) and result['automatic_point']==point and predictor.last_case_forwards==cost
        probes.append(dict(mode=recipe.get('mode','single'),mask_different_voxels=0,point_equal=True,logical_forward_count=cost))
    write(R/'GROUP_PORTABLE_SYNTHETIC_GATE.json',dict(at=now(),state='passed',probes=probes,source_sha256=sha(Path(__file__)),predictor_sha256=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/deploy_group_v1.py'),real_network_forward=False,patient_forward=False,GT_access=False,meaning='synthetic orchestration/math equivalence only; not checkpoint/neural verification'))
    print(dict(state='passed',probes=len(probes),real_network_forwards=0))

if __name__=='__main__':main()
