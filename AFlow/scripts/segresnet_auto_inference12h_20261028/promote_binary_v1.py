"""Independent complete-mask audit of one global CPU-search winner."""
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.cpu_binary_v1 import unpack,apply,LABEL_DIR,FOLDER


def main():
    from scripts.followup_test270_common import summary_ci,paired,validate_rows
    result=read(FOLDER/'RANKING.json');best=result['best'];recipe=best['recipe']
    candidate=R/'candidates/0001_cpu_binary';candidate.mkdir(parents=True,exist_ok=True)
    index=read(LABEL_DIR/'index.json');labels=np.memmap(LABEL_DIR/'labels.u8',np.uint8,'r',shape=(270,*CANVAS))
    previous=read(OLD/'RAW_TEST270.json')['per_case'];lookup={d['case_id']:d for d in previous}
    original_dir=ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270'
    recorded=np.load(FOLDER/'ALL_PER_CASE_DICE.npy')
    cases=[];files=[]
    write(candidate/'RECIPE.json',dict(at=now(),base_recipe=str((OLD/'SELECTED_RECIPE.json').relative_to(ROOT)),
          base_recipe_sha256=sha(OLD/'SELECTED_RECIPE.json'),postprocessing=recipe,
          selected_on_test270=True,test_tuned=True,GT_input_to_inference=False,
          checkpoint_sha256=CHECKPOINT_SHA,forwards_per_case=best['neural_case_forwards'],point_source='unchanged five-view pre-postprocessing argmax'))
    for slot,cid in enumerate(index['case_ids']):
        guard();selected=unpack(OLD/'predictions'/(cid+'.npz'));original=unpack(original_dir/'predictions'/(cid+'.npz'))
        mask=apply(selected,original,recipe,{'selected':selected})
        target=np.array(labels[slot][slices(index['geometry'][cid])]>0)
        tp,fp,fn=[int(a.sum()) for a in [mask&target,mask&~target,~mask&target]]
        den=2*tp+fp+fn;dice=float(np.float32(2*tp)/np.float32(den)) if den else 1.
        assert dice==float(recorded[best['configuration_index'],slot])
        point=lookup[cid]['automatic_point'];assert bool(target[tuple(point)])==lookup[cid]['automatic_point_hits_gt']
        path=candidate/'predictions'/(cid+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,mask_bits=np.packbits(mask.ravel()),shape=np.array(mask.shape),point=np.array(point))
        cases.append(dict(case_id=cid,subject_id=lookup[cid]['subject_id'],locator_dsc=dice,automatic_point=point,
              automatic_point_hits_gt=lookup[cid]['automatic_point_hits_gt'],tp=tp,fp=fp,fn=fn,source_slot=slot))
        files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),case_id=cid))
    validate_rows(cases);metrics=summary_ci(cases);metrics['point_hits']=sum(d['automatic_point_hits_gt'] for d in cases)
    contrast=paired(cases,previous);contrast['bootstrap']='10000 patient-cluster paired seed20260918; post-selection/test-tuned descriptive only'
    write(candidate/'RAW_TEST270.json',dict(at=now(),per_case=cases,predictions=files,test_tuned=True,
          current_test_retuning=True,model_forwards_added=0,training=False))
    output=dict(at=now(),state='audited_candidate',method='segresnet_w32_e171_auto12h_component1000_testtuned_20261028',
          metrics=metrics,comparison_to_previous_incumbent=contrast,recipe=recipe,
          checkpoint_sha256=CHECKPOINT_SHA,model_forwards_added=0,training=False,test_tuned=True,
          current_test_retuning=True,original_test_result_unchanged=True,all270_masks_independently_verified=True,
          raw_sha256=sha(candidate/'RAW_TEST270.json'),recipe_sha256=sha(candidate/'RECIPE.json'))
    write(candidate/'RESULT.json',output)
    write(R/'INCUMBENT.json',dict(at=now(),state='interim_audited_test_tuned_incumbent',
          method=output['method'],scan_dice=metrics['scan_dice'],subject_dice=metrics['subject_dice'],point_hits=metrics['point_hits'],
          raw_path=str((candidate/'RAW_TEST270.json').relative_to(ROOT)),result_path=str((candidate/'RESULT.json').relative_to(ROOT)),
          recipe_path=str((candidate/'RECIPE.json').relative_to(ROOT)),test_tuned=True,research_continues=True))
    (candidate/'RESULT.md').write_text(f'''# 中间候选：已存掩膜的全局小连通域过滤

原270/119、同一w32第171轮权重。对旧五镜像＋体积解码掩膜，按本轮270均值从179配置选择规则`{recipe['id']}`。这是直接test-tuned选择；不是独立确认，也不是全轮研究终态。

Scan **{metrics['scan_dice']:.9f}**，Subject **{metrics['subject_dice']:.9f}**，Point **{metrics['point_hits']}/270**。对前一0.780995配方差{contrast['scan_delta']:+.9f}，患者配对描述性95%区间{contrast['scan_delta_ci95']}；{contrast['improved_scans']}例改善、{contrast['worse_scans']}例下降、{contrast['tied_scans']}例相同。选择后的区间不证明独立泛化，少量提升也不保证可迁移。

只根据预测掩膜的26连通域体积过滤，不读GT；保留>=1000体素区域。小区域既可能是假阳性也可能是真实残余灶，不能当临床通用清理规则。未添加网络前向，部署仍原单权重5视图加CPU后处理；Point定义保持原概率argmax，不从处理后的掩膜重新选点。

全部270保存掩膜的TP/FP/FN与搜索逐例矩阵精确一致，定位点也一致。无训练/outer/confirmation。本轮12小时研究继续，GPU资源仍按独立恢复记录核验。

[原始逐例结果](RAW_TEST270.json) · [配方](RECIPE.json) · [机器结果](RESULT.json)
''')
    print(output)


if __name__=='__main__':main()
