"""Fixed centroid-projected point from the existing Flip8 mask; Dice must be identical."""
import json
from datetime import datetime,timezone
import statistics
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.evaluate_t1ce_wt_flip8_tta_replication_inner import flip8_logits
from scripts.train_t1ce_wt_capg3d import BrainCropDataset
from scripts.evaluate_structured_background_aux import summarize
from scripts.evaluate_data_recipe_fusion import score
from scripts import structured_background_protocol as p

RECORD=p.ROOT/'archive/research_records/2026-09-11_mask_centroid_point'
SOURCE=p.ROOT/'archive/research_records/2026-09-10_subject_fixed_flip8'


def centroid_point(mask):
    coords=torch.nonzero(mask,as_tuple=False)
    if coords.numel()==0:return None
    center=coords.float().mean(0); dist=(coords.float()-center).square().sum(1); best=dist.min(); candidates=coords[dist==best]
    linear=candidates[:,0]*mask.shape[1]*mask.shape[2]+candidates[:,1]*mask.shape[2]+candidates[:,2]
    return int(linear.argmin().item()), int(candidates[linear.argmin()].tolist()[0]*mask.shape[1]*mask.shape[2]+candidates[linear.argmin()].tolist()[1]*mask.shape[2]+candidates[linear.argmin()].tolist()[2])


@torch.inference_mode()
def main():
    out=RECORD/'RAW_DEVELOPMENT.json'
    if out.exists():raise FileExistsError(out)
    source=json.loads((SOURCE/'RAW_DEVELOPMENT.json').read_text());verification=json.loads((SOURCE/'VERIFICATION_RECEIPT.json').read_text());p.checked(SOURCE/'RAW_DEVELOPMENT.json',verification['raw_sha256'])
    fit,dev=p.check_roles();models={}
    for seed in p.SEEDS:
        path=p.ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/subject_balanced_sampling_20260907/seed{seed}_subject_balanced/TRAIN_DONE.json';d=json.loads(path.read_text());p.checked(d['checkpoint']['path'],d['checkpoint']['sha256']);models[str(seed)]=d
    files=['scripts/evaluate_mask_centroid_point.py','scripts/evaluate_t1ce_wt_flip8_tta_replication_inner.py','scripts/evaluate_data_recipe_fusion.py','scripts/train_t1ce_wt_capg3d.py','sammed3d/t1ce_wt_capg3d.py','archive/research_records/2026-09-11_mask_centroid_point/IDEA.md','archive/research_records/2026-09-11_mask_centroid_point/PREREG.md']
    p.write_new(RECORD/'DEV_PREACCESS.json',{'status':'fixed_centroid_point_before_new_dev_forwards','created_utc':datetime.now(timezone.utc).isoformat(),'files':{n:p.sha(p.ROOT/n) for n in files},'source_raw_sha256':p.sha(SOURCE/'RAW_DEVELOPMENT.json'),'fit_sha256':p.FIT_SHA,'dev_sha256':p.DEV_SHA,'views':'fixed_flip8','point_rule':'prediction foreground centroid projected to nearest foreground; tie smallest flat index','mask_rule':'unchanged flip8 mean logit>0','expected_forwards':2880,'new_forwards':0,'test_access':False})
    loader=DataLoader(BrainCropDataset(p.CACHE,expected_split='train',case_ids=[r['case_id'] for r in dev]),batch_size=8,num_workers=0,shuffle=False); device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(1)
    results={};gates={};comparisons={};forwards=0
    for idx,seed in enumerate(p.SEEDS):
        model=AutomaticPromptLocator(center_aware=False).to(device).eval();ck=torch.load(models[str(seed)]['checkpoint']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict'],strict=True);rows=[]
        for batch in loader:
            image=batch['image'].to(device);truth=batch['label'].to(device)>.5;identity,flip8=flip8_logits(model,image);forwards+=len(image)*8
            for j,case in enumerate(batch['case_id']):
                pred=flip8[j,0]>0;target=truth[j,0];inter=int((pred&target).sum());npred=int(pred.sum());nt=int(target.sum());chosen=centroid_point(pred);point=chosen[1] if chosen else int(flip8[j,0].flatten().argmax());rows.append({'case_id':case,'subject_id':batch['subject_id'][j],'dice':2*inter/(npred+nt),'point_hit':bool(target.flatten()[point]),'point_flat_index':point,'intersection':inter,'predicted_voxels':npred,'target_voxels':nt,'centroid_defined':chosen is not None})
        summary=summarize(rows,2026091131+idx);flip=source['results'][str(seed)]['flip8'];ident=source['results'][str(seed)]['identity'];assert all(a['dice']==b['dice'] for a,b in zip(rows,flip['per_case'],strict=True));results[str(seed)]={'centroid':summary,'flip8':flip,'identity':ident};comparisons[str(seed)]={'vs_flip8_point_delta':summary['point_hits']-flip['point_hits'],'vs_identity_point_delta':summary['point_hits']-ident['point_hits']};gates[str(seed)]={'dice_exact_flip8':True,'point_at_least_flip8':comparisons[str(seed)]['vs_flip8_point_delta']>=0,'point_at_least_identity':comparisons[str(seed)]['vs_identity_point_delta']>=0,'native_point_floor':summary['point_hits']>=[169,162][idx]}
        del model,ck,image,truth,identity,flip8;torch.cuda.empty_cache()
    assert forwards==2880;all_pass=all(all(v.values()) for v in gates.values());raw={'schema':'aflow.mask_centroid_point.reused_development','status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'comparisons':comparisons,'gates':gates,'all_gates_passed':all_pass,'new_forwards':forwards,'dev_preaccess_sha256':p.sha(RECORD/'DEV_PREACCESS.json'),'access':{'reused_development180':True,'test':False,'confirmation':False,'outer':False,'external':False},'decision':'candidate_requires_fixed_test_only_if_all_gates' if all_pass else 'close_centroid_point_rule_no_radius_or_component_search'};p.write_new(out,raw);print(json.dumps({k:v for k,v in raw.items() if k!='results'},indent=2))


if __name__=='__main__':main()
