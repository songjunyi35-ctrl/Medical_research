"""Frozen validation statistics and atomic CPU-verified inference handoff."""
import numpy as np
import torch
from scripts.race_v2_20261029.common import *
from scripts.race_v2_20261029.model import RaceBenefitSeg
from scripts.race_v2_20261029.runner import checkpoint
from scripts import segresnet_pretrained_native_common as native

def main():
    verify_sources();disk();done=read(R/'DONE.json');best_epoch=done['best_new_epoch']
    diagnosis=read(R/'DIAGNOSIS108.json')
    base_cases=[dict(case_id=c['case_id'],subject_id=c['subject_id'],**{k:v for k,v in c['modes']['normal'].items() if k not in ('gate','logit_correction','probability_correction')}) for c in diagnosis['per_case']]
    report=(read(R/f'validation_e{best_epoch:03}.json') if best_epoch else
            dict(metrics=diagnosis['metrics']['normal'],per_case=base_cases))
    comparison=native.paired(base_cases,report['per_case'])
    promoted=(comparison['scan_delta']>=.003 and comparison['scan_delta_ci95'][0]>0 and
              comparison['subject_delta']>=0 and comparison['point_hits_delta']>=0 and best_epoch>0)
    portable={}
    for name in ('best','latest'):
        saved=torch.load(TMP/'run'/f'{name}.pth',map_location='cpu',weights_only=False)
        model=RaceBenefitSeg(width=32,dropout=.1,dim=16,enabled=True)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        assert all(bool(v.isfinite().all()) for v in model.state_dict().values())
        path=R/'checkpoints'/f'{name}.pth';disk(extra=80<<20)
        state={k:v for k,v in saved.items() if k!='optimizer_state_dict'}
        checkpoint(path,state)
        reload=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(reload['model_state_dict'],strict=True)
        assert all(torch.equal(v,reload['model_state_dict'][k]) for k,v in state['model_state_dict'].items())
        portable[name]=dict(path=str(path),sha256=sha(path),new_epoch=saved['new_epoch'],new_updates=saved['new_updates'],strict_cpu_reload=True)
    fixed=R/'checkpoints'/f'race_v2_fixed_e{best_epoch:03}.pth'
    if not fixed.exists():os.link(portable['best']['path'],fixed)
    assert sha(fixed)==portable['best']['sha256']
    recipe=read(R/'RECIPE.json')
    handoff=dict(at=now(),run_id=recipe['run_id'],architecture='scripts.race_v2_20261029.model.RaceBenefitSeg',
       construction=dict(width=32,dropout=.1,dim=16,enabled=True),state_dict_compatible_with='sammed3d.race_seg.RaceSeg',
       checkpoint_path=str(fixed),checkpoint_sha256=sha(fixed),strict_cpu_reload=True,
       actual_new_epochs=best_epoch,actual_new_updates=best_epoch*122,total_stage_new_epochs=done['new_epoch'],
       total_stage_new_updates=done['new_updates'],source_epoch=164,source_checkpoint_sha256=SOURCE_SHA,
       optimizer_new=True,source_best_has_no_optimizer=True,execution_freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),
       recipe_sha256=sha(R/'RECIPE.json'),validation=report,promoted_by_frozen_validation_gate=promoted,
       input_geometry=dict(validation_canvas=list(CANVAS),spacing_mm=[1,1,1],crop='historical image-only nonzero brain crop; native voxels',
           normalization='historical N4/nonzero zscore',padding='centered; excluded from scoring',
           reference='image-derived LR reflection in input-voxel homogeneous coordinates; low feature centers input8*j',
           mirror='R_view=T@R@T for self-inverse mirror T; flip both image and valid support; invert logits to canonical view before crop',
           test_canvas='use split index canvas dynamically, do not hardcode validation160 or test168 across splits'),
       deployment_labels_or_case_lookup=False,test_pixels=False)
    write(R/'HANDOFF.json',handoff)
    baseline={c['case_id']:c for c in base_cases}
    worse=sorted([dict(case_id=c['case_id'],subject_id=c['subject_id'],delta=c['locator_dsc']-baseline[c['case_id']]['locator_dsc'],
           baseline=baseline[c['case_id']]['locator_dsc'],candidate=c['locator_dsc']) for c in report['per_case'] if c['locator_dsc']<baseline[c['case_id']]['locator_dsc']],key=lambda c:c['delta'])
    result=dict(at=now(),state='complete',run_id=recipe['run_id'],best_new_epoch=best_epoch,metrics=report['metrics'],
        comparison=comparison,promoted=promoted,worse_cases=worse,portable=portable,actual_new_epochs=done['new_epoch'],
        actual_new_updates=done['new_updates'],test_pixels=False,test_tuned_development_history=True,
        interpretation='new-optimizer finetuning plus benefit objective; no matched control causal attribution')
    write(R/'RESULT.json',result)
    (R/'RESULT.md').write_text(f'# RACE-v2参照收益校准训练结果\n\n状态complete；新增{done["new_epoch"]}轮/{done["new_updates"]}更新，新AdamW，不是e164无缝续训。\n\n'
        f'选新增{best_epoch}轮，108 Scan {report["metrics"]["scan_dice"]:.9f}、Subject {report["metrics"]["subject_dice"]:.9f}、Point {report["metrics"]["point_hits"]}/108。'
        f'相对同口径e164 Scan差{comparison["scan_delta"]:+.9f}，患者整簇配对95%区间{comparison["scan_delta_ci95"]}；冻结晋级门{promoted}。\n\n'
        '原972/108、单T1ce、原生1mm固定；训练agent未访问270像素或模型评测。全部继承test-tuned开发历史，不称独立泛化或匹配因果消融。固定CPU严格重载权重及几何约定见[HANDOFF.json](HANDOFF.json)。\n')

if __name__=='__main__':main()
