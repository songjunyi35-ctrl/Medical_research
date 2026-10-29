"""Reconstruct full-crop masks for a domain decoder; outside GT never disappears."""
import argparse,copy,numpy as np,torch
from scripts.segresnet_auto_inference12h_20261028.intensity_ensemble_v1 import prepare,probability,BASE
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_cached_v1 import guard
from scripts.segresnet_auto_inference12h_20261028.brain_domain_v1 import supports
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v2 import aggregate
from scripts.segresnet_auto_inference12h_20261028.promote_v7 import single_cut


@torch.inference_mode()
def main(candidate):
    from scripts.followup_test270_common import summary_ci,paired,validate_rows
    analysis='g2_1000_intensity_ensemble_calibration';best=read(R/'analyses'/analysis/'RANKING.json')['best'];recipe=best['recipe'];params=best['parameters'];source_job=read(R/'analysis_queue_aux_v1/g2_0000_intensity_ensemble.json')
    folder=R/'candidates'/candidate;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists()
    write(folder/'AUDIT_FREEZE.json',dict(at=now(),source_sha256=sha(Path(__file__)),best=best,analysis=analysis,whole_original_GT_domain=True))
    rows,geometry,images,labels=data();records=[];files=[]
    for slot,row in enumerate(rows):
        guard();cid=row['case_id'];sl=slices(geometry[cid]);image=np.array(images[slot][sl],np.float32)
        p=probability(prepare(cid,source_job),recipe);point_p=p;point=list(map(int,np.unravel_index(int(point_p.argmax()),point_p.shape)))
        valid=supports(image)[0];codes=(p.clamp(0,1)*65535+.5).long().cpu().numpy();counts=np.bincount(codes[valid],minlength=65536)
        cut=single_cut(counts,params);mask=(codes>=cut)&valid;target=np.array(labels[slot][sl]>0)
        tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive;score=float(np.float32(2*tp)/np.float32(den)) if den else 1.
        expected=float(np.load(TMP/'analyses'/analysis/'cases'/str(best['recipe_index'])/(cid+'.npz'))['dice'][best['parameter_index']]);assert score==expected,(cid,score,expected)
        path=folder/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=cid,subject_id=row['subject_id'],locator_dsc=score,automatic_point=point,automatic_point_hits_gt=bool(target[tuple(point)]),tp=tp,fp=pred-tp,fn=positive-tp,source_slot=slot,GT_outside_support=int((target&~valid).sum())))
        files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),case_id=cid))
    validate_rows(records);metrics=summary_ci(records);metrics['point_hits']=sum(r['automatic_point_hits_gt'] for r in records)
    assert abs(metrics['scan_dice']-best['scan_dice'])<1e-12 and metrics['point_hits']==best['point_hits']
    reference=read(OLD/'RAW_TEST270.json')['per_case'];contrast=paired(records,reference);contrast['bootstrap']='10000 patient-cluster paired seed20260918; post-selection descriptive, no multiplicity correction'
    write(folder/'RAW_TEST270.json',dict(at=now(),per_case=records,predictions=files,test_tuned=True,current_test_retuning=True,model_forwards_added=0))
    exported=copy.deepcopy(recipe)
    if recipe['alpha']==0:exported=read(R/'candidates/0020_brain_domain/RECIPE.json')['aggregation']
    else:
        specs={g['render']:g['spec'] for g in read(R/'INTENSITY_REGISTRATION.json')['groups']}
        exported.update(group_specs={gid:specs[gid] for gid in recipe['groups']},group_views=list(range(8)),base_recipe=BASE,brain_support_margin=0)
    write(folder/'RECIPE.json',dict(at=now(),aggregation=exported,source_job=source_job,decoder=params,decoder_kind='brain_domain_volume',point_rule='unchanged base aggregation point',checkpoint_sha256=CHECKPOINT_SHA,GT_or_case_ID_input=False,evaluation_domain_unchanged=True))
    method='segresnet_w32_e171_auto12h_'+candidate+'_testtuned_20261028';result=dict(at=now(),state='audited_candidate',method=method,metrics=metrics,comparison_to_previous_incumbent=contrast,best_configuration=best,analysis=analysis,locator_forwards=recipe['neural_case_forwards'],all270_masks_independently_verified=True,GT_domain_unchanged=True,raw_sha256=sha(folder/'RAW_TEST270.json'),recipe_sha256=sha(folder/'RECIPE.json'),training=False,test_tuned=True,current_test_retuning=True,checkpoint_sha256=CHECKPOINT_SHA)
    write(folder/'RESULT.json',result)
    (folder/'RESULT.md').write_text(f'''# 已审计强度增强组合与脑区解码 `{candidate}`\n\n同w32第171轮/T1ce/native1mm，原270/119。Scan **{metrics['scan_dice']:.9f}**，Subject **{metrics['subject_dice']:.9f}**，Point **{metrics['point_hits']}/270**，{recipe['neural_case_forwards']}次单权重前向。GT只评分与全局超参选择；支撑仅图像非零填孔/膨胀，阈值只读支撑内预测计数。\n\n全部原crop GT含支撑外仍计FN，没有缩小评估域。270mask计数与搜索矩阵逐例精确一致。对研究前5视图差{contrast['scan_delta']:+.9f}，描述性95%{contrast['scan_delta_ci95']}，直接test-tuned选择后区间未校正搜索，非独立泛化。训练0，审计神经前向0，正负完整保留。\n\n[逐例](RAW_TEST270.json) · [规则](RECIPE.json) · [结果](RESULT.json)\n''')
    if metrics['scan_dice']>read(R/'INCUMBENT.json')['scan_dice']:
        write(R/'INCUMBENT.json',dict(at=now(),state='interim_audited_test_tuned_incumbent',method=method,scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],raw_path=str((folder/'RAW_TEST270.json').relative_to(ROOT)),result_path=str((folder/'RESULT.json').relative_to(ROOT)),recipe_path=str((folder/'RECIPE.json').relative_to(ROOT)),test_tuned=True,research_continues=True))
    print(dict(method=method,metrics=metrics))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args();main(a.candidate)
