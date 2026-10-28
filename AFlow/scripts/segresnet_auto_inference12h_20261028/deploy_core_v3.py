"""Portable frozen-weight predictor for core-view recipes; no GT or case-ID input."""
import argparse,copy,hashlib,importlib,json
from pathlib import Path
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import CHECKPOINT_SHA,CHECKPOINT,ROOT
from scripts.segresnet_inference_search_20261028.common import numeric_mode
from scripts.segresnet_auto_inference12h_20261028.promote_v6 import single_cut,adaptive_single_cut


def export_recipe(candidate,output):
    folder=Path(candidate);recipe_path=folder/'RECIPE.json';posts=[]
    while True:
        recipe=json.loads(recipe_path.read_text())
        if 'postprocessing' in recipe:posts.append(recipe['postprocessing'])
        if 'base_recipe_path' not in recipe:break
        recipe_path=ROOT/recipe['base_recipe_path']
    assert 'aggregation' in recipe
    assert recipe['aggregation']['kind']!='group_mix','use a view-group predictor for this recipe'
    config=dict(schema='aflow_frozen_core_inference_v1',checkpoint_sha256=CHECKPOINT_SHA,
       aggregation=recipe['aggregation'],decoder=recipe['decoder'],decoder_kind=recipe['decoder_kind'],
       point_rule=recipe['point_rule'],postprocessing=list(reversed(posts)),selected_on_reused_test270=True,
       input_contract='same native1mm T1ce normalization/canvas; geometry removes padding',
       source_result_sha256=hashlib.sha256((folder/'RESULT.json').read_bytes()).hexdigest())
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    return config


def crop(geometry):return tuple(slice(int(a),int(a)+int(n)) for a,n in zip(geometry['canvas_offset'],geometry['extent']))


def aggregate(logits,recipe,image):
    kind=recipe['kind'];views=recipe['views'];prototype=next(iter(logits.values()))
    all_logits=[logits[i] if i in logits else torch.zeros_like(prototype) for i in range(8)]
    if kind=='brain_domain':return aggregate(logits,recipe['base'],image)
    if kind=='probability_mean':
        p=torch.zeros_like(prototype)
        for i in views:p.add_((logits[i]/recipe['view_temperature']).sigmoid(),alpha=1/len(views))
        return p,p
    if kind=='logit_mean':
        p=torch.zeros_like(prototype)
        for i in views:p.add_(logits[i],alpha=1/len(views))
        return p.sigmoid(),p.sigmoid()
    if kind=='consensus':
        from scripts.segresnet_auto_inference12h_20261028.consensus_v1 import probability
        stack=torch.stack([z.sigmoid() for z in all_logits]);p=probability(dict(logits=all_logits,p=stack),recipe)
        return p,stack.mean(0)
    if kind in ['case_consistency','nonlinear_weighted_fusion','joint_weighted_fusion']:
        module_name={'case_consistency':'case_consistency_v1','nonlinear_weighted_fusion':'nonlinear_fusions_v1','joint_weighted_fusion':'joint_fusions_v1'}[kind]
        module=importlib.import_module('scripts.segresnet_auto_inference12h_20261028.'+module_name)
        p=module.probability(module.prepare(all_logits),recipe);return p,p
    if kind=='local_bilateral_mrf':
        from scripts.segresnet_auto_inference12h_20261028.local_mrf_v1 import refine
        base=torch.zeros_like(prototype)
        for i in [2,3,4,6,7]:base.add_(logits[i].sigmoid(),alpha=.2)
        return refine(base,image,recipe),base
    if kind=='group_mix':raise ValueError('group/ROI/phase recipe needs its view-group predictor; this entry supports core recipes')
    from scripts.segresnet_auto_inference12h_20261028.core_fusions_v3 import prepare,probability
    p=probability(prepare(all_logits),recipe);return p,p


def decode(p,parameters,kind):
    if kind=='rankseg_v1':
        from scripts.segresnet_auto_inference12h_20261028.rankseg_v1 import decode as rank_decode,vendor
        return rank_decode(p,parameters,vendor()).cpu().numpy()
    codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy()
    if 'threshold' in parameters:cut=int(np.floor(parameters['threshold']*65535))+1
    else:
        counts=np.bincount(codes.ravel(),minlength=65536)
        cut=adaptive_single_cut(counts,parameters) if 'temperature_base' in parameters else single_cut(counts,parameters)
    return codes>=cut


class FrozenCorePredictor:
    def __init__(self,recipe,checkpoint=CHECKPOINT,device='cuda'):
        self.config=json.loads(Path(recipe).read_text()) if isinstance(recipe,(str,Path)) else copy.deepcopy(recipe)
        self.device=torch.device(device);assert self.device.type=='cuda'
        h=hashlib.sha256()
        with Path(checkpoint).open('rb') as f:
            for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
        assert h.hexdigest()==self.config['checkpoint_sha256']==CHECKPOINT_SHA
        numeric_mode()
        from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
        state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['epoch']==171
        self.model=AutomaticPromptLocator(center_aware=False,init_filters=32,dropout=.1).to(self.device)
        self.model.load_state_dict(state['model_state_dict'],strict=True);self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def predict(self,images,geometries):
        x=torch.as_tensor(images,dtype=torch.float32,device=self.device).contiguous()
        assert x.ndim==5 and x.shape[1]==1 and len(x)==len(geometries) and all(n%8==0 for n in x.shape[-3:])
        assert torch.isfinite(x).all()
        for geometry in geometries:
            assert all(0<=int(a) and 0<int(n) and int(a)+int(n)<=int(s) for a,n,s in zip(geometry['canvas_offset'],geometry['extent'],x.shape[-3:]))
        recipe=self.config['aggregation'];views=set(recipe['views'])
        if recipe['kind'] in ['consensus','case_consistency']:views=set(range(8))
        if recipe['kind']=='local_bilateral_mrf':views.update([2,3,4,6,7])
        if any(p.get('source') in ['union','intersection','distance'] for p in self.config['postprocessing']):views.add(0)
        logits={}
        for flip in sorted(views):
            dims=tuple(d+2 for d in range(3) if flip&(1<<d));view=torch.flip(x,dims) if dims else x
            with torch.autocast('cuda',dtype=torch.bfloat16):z=self.model(view).segmentation_logits
            z=z.float();logits[flip]=torch.flip(z,dims) if dims else z
        results=[]
        for k,geometry in enumerate(geometries):
            sl=crop(geometry)
            cropped={i:z[(k,0)+sl].contiguous() for i,z in logits.items()}
            p,point_p=aggregate(cropped,recipe,x[(k,0)+sl].contiguous())
            point=list(map(int,np.unravel_index(int(point_p.argmax()),point_p.shape)))
            if recipe['kind']=='brain_domain':
                from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
                valid=supports(x[(k,0)+sl].cpu().numpy())[recipe['margin']]
                codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy()
                counts=np.bincount(codes[valid],minlength=65536)
                cut=single_cut(counts,self.config['decoder']);mask=(codes>=cut)&valid
            else:mask=decode(p,self.config['decoder'],self.config['decoder_kind'])
            original=(cropped[0].sigmoid()>.5).cpu().numpy() if 0 in cropped else mask
            for rule in self.config['postprocessing']:
                if rule.get('kind') in ['minimum','relative','largest']:
                    from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import apply
                    mask=apply(mask,rule)
                else:
                    from scripts.segresnet_auto_inference12h_20261028.postprocess_sweep_v1 import apply_global
                    mask=apply_global(mask,original,rule,{'selected':mask})
            results.append(dict(mask=mask,automatic_point=point))
        return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export-candidate',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();export_recipe(args.export_candidate,args.output)
