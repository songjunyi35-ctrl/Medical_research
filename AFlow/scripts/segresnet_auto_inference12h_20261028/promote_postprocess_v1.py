"""Audit one globally selected component rule composed with an audited base."""
import argparse
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.guard_v2 import guard
from scripts.segresnet_auto_inference12h_20261028.postprocess_sweep_v1 import apply_global
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import unpack,LABEL_DIR


def main(analysis,candidate):
    from scripts.followup_test270_common import summary_ci,paired,validate_rows
    source=R/'analyses'/analysis;ranking=read(source/'RANKING.json');best=ranking['best'];base=ROOT/ranking['base_folder'];base_result=read(base/'RESULT.json')
    folder=R/'candidates'/candidate;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'RESULT.json').exists()
    write(folder/'AUDIT_FREEZE.json',dict(at=now(),analysis=analysis,ranking_sha256=sha(source/'RANKING.json'),source_sha256=sha(Path(__file__)),base_raw_sha256=sha(base/'RAW_TEST270.json'),postprocessing_source_sha256=sha(ROOT/'scripts/segresnet_auto_inference12h_20261028/postprocess_sweep_v1.py'),best=best,test_tuned=True))
    index=read(LABEL_DIR/'index.json');labels=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))
    previous=read(base/'RAW_TEST270.json')['per_case'];start_reference=read(OLD/'RAW_TEST270.json')['per_case'];matrix=np.load(source/'ALL_PER_CASE_DICE.npy')
    records=[];files=[]
    for slot,row in enumerate(previous):
        guard();cid=row['case_id'];assert cid==index['case_ids'][slot]
        selected=unpack(base/'predictions'/(cid+'.npz'));original=unpack(R/'control_from_core_flip0/predictions'/(cid+'.npz'))
        mask=apply_global(selected,original,best['recipe'],{'selected':selected});target=np.array(labels[slot][slices(index['geometry'][cid])]>0)
        tp=int((mask&target).sum());pred=int(mask.sum());positive=int(target.sum());den=pred+positive
        score=float(np.float32(2*tp)/np.float32(den)) if den else 1.;assert score==float(matrix[best['configuration_index'],slot])
        point=row['automatic_point'];assert bool(target[tuple(point)])==row['automatic_point_hits_gt']
        path=folder/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        records.append(dict(case_id=cid,subject_id=row['subject_id'],locator_dsc=score,automatic_point=point,automatic_point_hits_gt=row['automatic_point_hits_gt'],tp=tp,fp=pred-tp,fn=positive-tp,source_slot=slot))
        files.append(dict(case_id=cid,path=str(path.relative_to(ROOT)),sha256=sha(path)))
    validate_rows(records);metrics=summary_ci(records);metrics['point_hits']=sum(r['automatic_point_hits_gt'] for r in records)
    assert abs(metrics['scan_dice']-best['scan_dice'])<1e-12
    def contrast(reference):
        x=paired(records,reference);x['bootstrap']='10000 patient-cluster paired seed20260918;95% descriptive,97.5% sensitivity only; not multiplicity correction';x['interpretation']='post-selection test-tuned descriptive; not independent confirmation';return x
    versus_base=contrast(previous);versus_start=contrast(start_reference)
    write(folder/'RAW_TEST270.json',dict(at=now(),per_case=records,predictions=files,test_tuned=True,current_test_retuning=True,training=False,model_forwards_added=0))
    write(folder/'RECIPE.json',dict(at=now(),base_recipe_path=str((base/'RECIPE.json').relative_to(ROOT)),base_recipe_sha256=sha(base/'RECIPE.json'),postprocessing=best['recipe'],checkpoint_sha256=CHECKPOINT_SHA,GT_input_to_inference=False,selected_on_test270=True,point_source='unchanged base probability point',locator_forwards=best['neural_case_forwards']))
    method='segresnet_w32_e171_auto12h_'+candidate+'_testtuned_20261028'
    result=dict(at=now(),state='audited_candidate',method=method,metrics=metrics,comparison_to_previous_incumbent=versus_start,comparison_to_search_base=versus_base,
          checkpoint_sha256=CHECKPOINT_SHA,analysis=analysis,best_configuration=best,locator_forwards=best['neural_case_forwards'],training=False,test_tuned=True,current_test_retuning=True,all270_masks_independently_verified=True,model_forwards_added=0,raw_sha256=sha(folder/'RAW_TEST270.json'),recipe_sha256=sha(folder/'RECIPE.json'))
    write(folder/'RESULT.json',result)
    (folder/'RESULT.md').write_text(f'''# 本轮已审计组合 `{candidate}`\n\n同一w32第171轮、单T1ce、原270/119，固定已审计基线上选择全局后处理规则`{best['recipe']['id']}`。直接test-tuned选择，非独立确认。\n\nScan **{metrics['scan_dice']:.9f}**，Subject **{metrics['subject_dice']:.9f}**，Point **{metrics['point_hits']}/270**。对组合前基线差{versus_base['scan_delta']:+.9f}、描述性95%区间{versus_base['scan_delta_ci95']}；对研究前五镜像配方差{versus_start['scan_delta']:+.9f}、区间{versus_start['scan_delta_ci95']}。选择后的区间不证明独立泛化或校正大规模搜索。\n\n全部270掩膜TP/FP/FN与搜索逐例矩阵精确一致，定位点保持基线概率点；后处理只读预测掩膜及同权重已存在的原视图，无GT输入或病例路由。部署仍{best['neural_case_forwards']}次单权重前向，后处理新增网络前向0。该规则只评价当前组合与当前复用队列。12小时研究继续。\n\n[逐例结果](RAW_TEST270.json) · [配方](RECIPE.json) · [机器结果](RESULT.json)\n''')
    if metrics['scan_dice']>read(R/'INCUMBENT.json')['scan_dice']:
        write(R/'INCUMBENT.json',dict(at=now(),state='interim_audited_test_tuned_incumbent',method=method,scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],raw_path=str((folder/'RAW_TEST270.json').relative_to(ROOT)),result_path=str((folder/'RESULT.json').relative_to(ROOT)),recipe_path=str((folder/'RECIPE.json').relative_to(ROOT)),test_tuned=True,research_continues=True))
    print(dict(method=method,scan_dice=metrics['scan_dice'],point_hits=metrics['point_hits'],delta_start=versus_start['scan_delta']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--candidate',required=True);a=p.parse_args();main(a.analysis,a.candidate)
