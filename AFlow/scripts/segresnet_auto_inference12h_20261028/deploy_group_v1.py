"""Portable single-checkpoint photometric group predictor; no GT or case IDs."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from scripts.segresnet_auto_inference12h_20261028.common import ROOT,CHECKPOINT,CHECKPOINT_SHA,sha
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import FrozenCorePredictor,crop,decode,export_recipe as export_core
from scripts.segresnet_auto_inference12h_20261028.render_v1 import forward_view
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import single_cut
from scripts.segresnet_auto_inference12h_20261028.intensity_ensemble_v1 import probability as ensemble_probability,BASE,core_prepare,core_probability


def photo_spec(spec):
    assert set(spec)<=set(['batch','gain','offset','gamma','gaussian_sigma']), 'this predictor supports deterministic photometric groups only'


def export_recipe(candidate,output):
    folder=ROOT/candidate if not isinstance(candidate,Path) else candidate
    recipe_path=folder/'RECIPE.json';posts=[]
    while True:
        recipe=json.loads(recipe_path.read_text())
        if 'postprocessing' in recipe:posts.append(recipe['postprocessing'])
        if 'base_recipe_path' not in recipe:break
        recipe_path=ROOT/recipe['base_recipe_path']
    aggregation=recipe['aggregation']
    if aggregation['kind']!='group_mix':return export_core(folder,output)
    if aggregation.get('multi_group_fusion'):
        assert set(aggregation['groups'])==set(aggregation['group_specs'])
        for spec in aggregation['group_specs'].values():photo_spec(spec)
        mode='photometric_ensemble'
    else:photo_spec(aggregation['group_spec']);assert aggregation['group_views']==list(range(8));mode='single_photometric_group'
    config=dict(schema='aflow_frozen_photometric_inference_v1',predictor_kind=mode,checkpoint_sha256=CHECKPOINT_SHA,aggregation=aggregation,decoder=recipe['decoder'],decoder_kind=recipe['decoder_kind'],postprocessing=list(reversed(posts)),selected_on_reused_test270=True,input_contract='same native1mm T1ce normalization/canvas and image-derived unpadding geometry',source_result_sha256=sha(folder/'RESULT.json'),neural_case_forwards=json.loads((folder/'RESULT.json').read_text())['locator_forwards'])
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n');return config


class FrozenGroupPredictor(FrozenCorePredictor):
    @torch.inference_mode()
    def predict(self,images,geometries):
        recipe=self.config['aggregation']
        if recipe['kind']!='group_mix':return super().predict(images,geometries)
        x=torch.as_tensor(images,dtype=torch.float32,device=self.device).contiguous();assert x.ndim==5 and x.shape[1]==1 and len(x)==len(geometries) and all(n%8==0 for n in x.shape[-3:]);assert torch.isfinite(x).all()
        for g in geometries:assert all(0<=int(a) and 0<int(n) and int(a)+int(n)<=int(s) for a,n,s in zip(g['canvas_offset'],g['extent'],x.shape[-3:]))
        multi=recipe.get('multi_group_fusion',False);specs=recipe['group_specs'] if multi else {recipe['group']:recipe['group_spec']}
        for spec in specs.values():photo_spec(spec)
        if recipe['alpha']<1:views=set(range(8)) if multi else set(recipe['base_views'])
        else:views=set()
        needs_original=any(rule.get('source') in ['union','intersection','distance'] for rule in self.config['postprocessing'])
        if needs_original:views.add(0)
        logits={flip:forward_view(self.model,x,{},flip) for flip in sorted(views)}
        means={}
        if recipe['alpha']!=0:
            for gid,spec in specs.items():
                value=torch.zeros_like(x)
                for flip in range(8):value.add_(forward_view(self.model,x,spec,flip).sigmoid(),alpha=1/8)
                means[gid]=value
        self.last_case_forwards=len(views)+8*len(means);assert self.last_case_forwards==self.config['neural_case_forwards']
        results=[]
        for k,geometry in enumerate(geometries):
            sl=crop(geometry);image=x[(k,0)+sl].contiguous();cropped={i:z[(k,0)+sl].contiguous() for i,z in logits.items()}
            valid=supports(image.cpu().numpy())[0]
            if multi:
                base=core_probability(core_prepare([cropped[i] for i in range(8)]),BASE) if recipe['alpha']<1 else None
                cache=dict(base=base,probs={gid:z[(k,0)+sl].contiguous() for gid,z in means.items()},valid=torch.from_numpy(valid).to(self.device),aggregates={});p=ensemble_probability(cache,recipe)
            else:
                value=means[recipe['group']][(k,0)+sl].contiguous();p=(torch.logit(value.clamp(1e-7,1-1e-7))/recipe['group_temperature']).sigmoid()
                if recipe['alpha']<1:
                    base=torch.zeros_like(p);base_views=recipe['base_views'];temperature=recipe['base_view_temperature']
                    for flip in sorted(base_views):base.add_((cropped[flip]/temperature).sigmoid(),alpha=1/len(base_views))
                    p=base*(1-recipe['alpha'])+p*recipe['alpha']
            point=list(map(int,np.unravel_index(int(p.argmax()),p.shape)))
            if 'brain_support_margin' in recipe or self.config['decoder_kind']=='brain_domain_volume':
                codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy();cut=single_cut(np.bincount(codes[valid],minlength=65536),self.config['decoder']);mask=(codes>=cut)&valid
            else:mask=decode(p,self.config['decoder'],self.config['decoder_kind'])
            original=(cropped[0].sigmoid()>.5).cpu().numpy() if 0 in cropped else mask
            for rule in self.config['postprocessing']:
                if rule.get('kind') in ['minimum','relative','largest']:
                    from scripts.segresnet_auto_inference12h_20261028.component_sweep_v1 import apply
                    mask=apply(mask,rule)
                else:
                    from scripts.segresnet_auto_inference12h_20261028.postprocess_sweep_v3 import apply_global
                    mask=apply_global(mask,original,rule,{'selected':mask})
            results.append(dict(mask=mask,automatic_point=point))
        return results


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--export-candidate',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();export_recipe(Path(args.export_candidate).resolve(),args.output)
