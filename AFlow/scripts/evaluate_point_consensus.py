"""Fixed point-vote aggregation on the already evaluated patient-sampling models."""
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import REFLECTION_AXES
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_structured_background_aux import summarize
from scripts import structured_background_protocol as p

ROOT = p.ROOT
RECORD = ROOT / "archive/research_records/2026-09-11_point_consensus"
SOURCE = ROOT / "archive/research_records/2026-09-10_subject_fixed_flip8"


def point_vote(logits):
    # logits: (8, D,H,W), where each view has already been inverse-reflected.
    points = [int(x.flatten().argmax()) for x in logits]
    unique = sorted(set(points))
    scored=[]
    for point in unique:
        support = int((logits.reshape(8,-1)[:,point] > 0).sum())
        mean_logit = float(logits.reshape(8,-1)[:,point].mean())
        scored.append((support, mean_logit, -point, point))
    return max(scored)[-1], points, max(scored)[0]


@torch.inference_mode()
def main():
    out=RECORD/'RAW_DEVELOPMENT.json'
    if out.exists(): raise FileExistsError(out)
    source=json.loads((SOURCE/'RAW_DEVELOPMENT.json').read_text())
    p.checked(SOURCE/'RAW_DEVELOPMENT.json', json.loads((SOURCE/'VERIFICATION_RECEIPT.json').read_text())['raw_sha256'])
    fit,dev=p.check_roles()
    models={}
    for seed in p.SEEDS:
        path=ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed{seed}_subject_balanced/TRAIN_DONE.json'
        d=json.loads(path.read_text()); p.checked(d['checkpoint']['path'],d['checkpoint']['sha256']); models[str(seed)]=d
    files=['scripts/evaluate_point_consensus.py','scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py','scripts/evaluate_structured_background_aux.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','archive/research_records/2026-09-11_point_consensus/IDEA.md','archive/research_records/2026-09-11_point_consensus/PREREG.md']
    p.write_new(RECORD/'DEV_PREACCESS.json',{'status':'fixed_point_vote_before_new_dev_forwards','created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(ROOT/n) for n in files},'source_raw_sha256':p.sha(SOURCE/'RAW_DEVELOPMENT.json'),'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'views':[list(a) for a in REFLECTION_AXES],'mask_rule':'unchanged Flip8 mean raw logits >0','point_rule':'candidate argmax support count > mean-logit > lowest index','threshold':0,'expected_forwards':2880,'new_forwards':0,'test_access':False})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,shuffle=False,num_workers=0)
    device=torch.device('cuda:0'); torch.cuda.set_device(device); torch.set_num_threads(1)
    results={}; comparisons={}; gates={}; forwards=0
    for idx,seed in enumerate(p.SEEDS):
        model=AutomaticPromptLocator(center_aware=False).to(device).eval(); ck=torch.load(models[str(seed)]['checkpoint']['path'],map_location='cpu',weights_only=False); model.load_state_dict(ck['model_state_dict'],strict=True)
        rows=[]; transitions=[]
        for batch in loader:
            image=batch['image'].to(device); truth=batch['label'].to(device)>.5
            views=[]
            for axes in REFLECTION_AXES:
                transformed=torch.flip(image,dims=axes) if axes else image
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16): logits=model(transformed).segmentation_logits[:,0].float()
                restored=torch.flip(logits,dims=axes) if axes else logits
                views.append(restored)
            stack=torch.stack(views); mean=stack.mean(0); forwards+=len(image)*8
            for j,case in enumerate(batch['case_id']):
                point,raw_points,support=point_vote(stack[:,j]); pred=mean[j]>0; target=truth[j,0]; inter=int((pred&target).sum()); npred=int(pred.sum()); nt=int(target.sum())
                identity=source['results'][str(seed)]['identity']['per_case'][len(rows)]
                flip=source['results'][str(seed)]['flip8']['per_case'][len(rows)]
                rows.append({'case_id':case,'subject_id':batch['subject_id'][j],'dice':2*inter/(npred+nt),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':nt,'vote_support':support,'identity_point':identity['point_flat_index'],'flip8_point':flip['point_flat_index']})
                transitions.append({'case_id':case,'identity_hit':identity['point_hit'],'flip8_hit':flip['point_hit'],'vote_hit':bool(target.flatten()[point]),'support':support})
        summary=summarize(rows,2026091121+idx); results[str(seed)]={'vote':summary,'flip8':source['results'][str(seed)]['flip8'],'identity':source['results'][str(seed)]['identity'],'transitions':transitions}
        def delta(a,b):
            return {'scan_delta':b['scan_dice']-a['scan_dice'],'subject_delta':b['subject_dice']-a['subject_dice'],'point_delta':b['point_hits']-a['point_hits']}
        comparisons[str(seed)]={'vote_vs_flip8':delta(results[str(seed)]['flip8'],summary),'vote_vs_identity':delta(results[str(seed)]['identity'],summary)}
        gates[str(seed)]={'point_at_least_flip8':comparisons[str(seed)]['vote_vs_flip8']['point_delta']>=0,'point_at_least_identity':summary['point_hits']>=results[str(seed)]['identity']['point_hits'],'mask_dice_unchanged_from_flip8':abs(summary['scan_dice']-results[str(seed)]['flip8']['scan_dice'])<1e-12}
        del model,ck; torch.cuda.empty_cache()
    assert forwards==2880
    all_pass=all(all(g.values()) for g in gates.values())
    raw={'schema':'aflow.point_consensus.reused_development','status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':comparisons,'gates':gates,'all_gates_passed':all_pass,'new_forwards':forwards,'dev_preaccess_sha256':p.sha(RECORD/'DEV_PREACCESS.json'),'access':{'reused_development180':True,'test':False,'confirmation':False,'outer':False,'external':False},'decision':'candidate_requires_new_joint_dice_point_gate' if all_pass else 'close_point_vote_rule_no_neighborhood_or_weight_search'}
    p.write_new(out,raw); print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2),flush=True)


if __name__=='__main__': main()
