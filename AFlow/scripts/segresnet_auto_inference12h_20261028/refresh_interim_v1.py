"""Refresh only this study's documentation markers from audited current data."""
import re
from scripts.segresnet_auto_inference12h_20261028.common import *
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import export_recipe
from scripts.segresnet_auto_inference12h_20261028.research_snapshot_v1 import main as snapshot


def main():
    from scripts.followup_test270_common import paired
    snapshot();inc=read(R/'INCUMBENT.json');folder=ROOT/Path(inc['result_path']).parent
    result=read(folder/'RESULT.json');metrics=result['metrics'];candidate=folder.name
    records=read(folder/'RAW_TEST270.json')['per_case']
    assert len(records)==270 and result['all270_masks_independently_verified']
    reference=read(ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270/RAW_TEST270.json')['summary']['per_case']
    prior=read(OLD/'RAW_TEST270.json')['per_case']
    contrasts={}
    for name,rows in [('original_w32',reference),('study_start_mirror5',prior)]:
        contrasts[name]=paired(records,rows)
        contrasts[name]['bootstrap']='10000 patient-cluster paired seed20260918; post-selection descriptive, no multiplicity correction'
    write(folder/'PAIRWISE_REFERENCES.json',dict(at=now(),comparisons=contrasts,test_tuned=True,model_forwards_added=0))
    config=export_recipe(folder,R/'PORTABLE_RECIPE_CURRENT.json')
    live=R/'live_verification'/candidate/'DONE.json';fresh=read(live) if live.exists() else None
    alternative=read(R/'candidates/0018_adaptive_holes/RESULT.json')['metrics']
    evidence=read(R/'COMPUTE_SNAPSHOT.json');n=evidence['completed_configuration_tuples'];cost=result['locator_forwards']
    delta=contrasts['original_w32']['scan_delta'];ci=contrasts['original_w32']['scan_delta_ci95'];used=read(R/'RECOVERY_COUNT.json')['used']
    brief=f"当前已审计Scan **{metrics['scan_dice']:.9f}**、Subject **{metrics['subject_dice']:.9f}**、Point **{metrics['point_hits']}/270**，部署{cost}次同权重前向；相对原普通推理0.773228040差 **{delta:+.9f}**，患者配对描述性95%{ci}。"
    audit='全部270掩膜TP/FP/FN、定位点与搜索矩阵独立核对。'
    fresh_text=(f"当前最高独立API重新前向270通过，mask差{fresh['mask_different_voxels']}体素，所有point相同，Dice差0。" if fresh else '当前最高的独立API重新前向270尚未完成；旧0008的fresh270通过仅支持旧配方，不能替代当前最高核验。')
    limits='所有全局规则直接由复用270选择，全部test-tuned；选择后区间未校正大规模搜索，不称独立泛化。无GT/病例ID单例路由，GT只评分和全局超参选择；脑区支撑限制预测时，仍保留整个原crop GT分母。'
    running=f"完整完成{n:,}个有限全局参数元组（非独立方法或唯一掩膜），保留全部负结果。双A10080的flip0/flip7各512整数相位共享前缀及后续融合/校准继续；当前不是结题。恢复{used}/6。"
    link=f'archive/research_records/{R.name}/INTERIM_RESULT.md'
    report=f'''# 12小时自动推理研究：中间快照，尚未结题

更新时间{now()}；研究窗口北京时间2026-10-28 20:41:16至29日08:41:16，不提前结束。固定单T1ce、SegResNet32通道171轮、原972/108/270及119测试患者，模型权重和规范状态训练0；不干预独立RACE。

{brief} {audit} {limits}

当前固定全局配方见[便携JSON](PORTABLE_RECIPE_CURRENT.json)，聚合为`{config['aggregation']['kind']}`，解码为`{config['decoder_kind']}`。配置只包含算法、模型SHA和原native归一化/几何输入契约，没有病例ID、GT或270个realized cuts查表。后处理新增CNN0。

{fresh_text} [部署接口及状态](DEPLOYMENT_INTERIM.md)。

7次前向替代0018仍保留Scan{alternative['scan_dice']:.9f}、Subject{alternative['subject_dice']:.9f}、Point{alternative['point_hits']}/270，供成本取舍，全部test-tuned。

{running}

已覆盖核心镜像子集/温度/阈值、对数与概率融合、预测一致性及STAPLE控制、非线性逐像素权重、MC-dropout、轴置换、整数相位、近尺度重采样、预测ROI放大、噪声平均、全局/不确定性体积解码、官方RankSEG、局部MRF、原生3D DenseCRF、脑区支撑、signed-distance形状融合及组件/形态组合。DenseCRF有限网格没有边项收益，最高是unary控制；跨相位有alpha0控制，不能把其收益归相位；未改变权重的候选不冒称新训练算法。论文/官方来源与范围见[联网调研](WEB_REVIEW.md)，各轮冻结见ROUND001–026。

计算只据记录报告：[配置与计算快照](COMPUTE_SNAPSHOT.json)、[访问记录](ACCESS_LEDGER.jsonl)、[实时任务](RESEARCH_STATUS.json)。阶段前缀只按新增前向计数，不重复相加；旧节点失联/失败未提交计算和未完任务不伪造确数。GPU利用率按新worker20秒采样区间统计，不将某次100%扩称全程100%。

持久新增<=2GiB/余量>=20GiB、scratch<=300GiB/余量>=70GiB。仅清理本轮已归档且无读者的可重建中间文件，保留输入/权重/core8逐视图logits/所有score-cut矩阵与审计掩膜。工具拒绝一次跨主机派生mask转移后未执行或重试，已在原worker安全完成相关分析，无需该传输。

[当前候选](candidates/{candidate}/RESULT.md) · [配方链](candidates/{candidate}/RECIPE.json) · [配对参考](candidates/{candidate}/PAIRWISE_REFERENCES.json) · [7视图替代](candidates/0018_adaptive_holes/RESULT.md) · [接续记录](RUNBOOK.md) · [结题要求](FINALIZATION_CHECKLIST.md)。

科研仓库只同步源码和文本证据，MRI/标签、权重、预测与数值数组留授权环境。最终全部任务、接口、文档、预算及发布核验后另行归档推送。
'''
    (R/'INTERIM_RESULT.md').write_text(report)
    deployment=f'''# 当前无训练核心视图部署接口

当前审计最高`{candidate}`，Scan{metrics['scan_dice']:.9f}/Point{metrics['point_hits']}，{cost}次前向。{fresh_text}

```python
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import FrozenCorePredictor
predictor = FrozenCorePredictor("archive/research_records/{R.name}/PORTABLE_RECIPE_CURRENT.json")
# images: float32 [B,1,X,Y,Z], original native1mm T1ce normalization; dimensions divisible by8
# geometry: image-derived canvas_offset/extent, removes padding
predictions = predictor.predict(images, geometries)
# each result: boolean mask and automatic_point in original native crop
```

API不接GT、病例ID或逐例阈值表；从图像/自身多视图预测计算支撑、权重与切点，严格核验唯一171轮checkpoint SHA，eval/bf16、无梯度。脑区支撑需输入中至少存在非零脑体素，合成显存门使用ones。

v3支持核心镜像、共识/case/非线性/joint、图像脑区支撑、局部MRF及组件/形态组合；group/ROI/phase若成为最高须额外对应预测器，不得强行套核心视图接口。输入必须沿用工程原预处理，不宣称任意外部MRI可直接代入。

旧0008独立fresh270核验：[2160次前向/全部mask0差](live_verification/0008_consensus_cc/RESULT.json)。当前候选是否fresh完成以其live_verification/DONE为准。独立复现只是工程一致性，所有规则仍test-tuned，不是独立泛化。模型与输入不上传科研库。
'''
    (R/'DEPLOYMENT_INTERIM.md').write_text(deployment)
    common=f"{brief} {audit} {limits}\n\n{running} [中间快照]({link}) · [接续](archive/research_records/{R.name}/RUNBOOK.md)。"
    blocks={
       'AGENTS.md':f"## 2026-10-28新授权：12小时无训练自动研究｜持续运行\n\n用户授权上网调研、实现和全局组合搜索，窗口至29日08:41:16，不提前结题；固定T1ce/w32第171轮/972108270，训练0，无新outer/confirmation，独立RACE不干预。worker1153287双A10080，原native三载荷SHA一致。\n\n{common}\n\n持久新增<=2GiB/空闲>=20GiB，全部scratch<=300GiB/空闲>=70GiB；只unlink本轮已归档无读者派生文件、保留目录骨架，原数据/权重/core8logits/所有分数切点掩膜保留。已冻结源码不改，CPU性能修复有数值等价门；不以旧fresh结果替代新候选。",
       'AFLOW_BRATS_HANDOFF_CURRENT.md':f"**12小时自动推理研究持续至29日08:41:16。**\n\n{common}\n\n{fresh_text} 当前接续以WORKING_STATE和直连任务为准；只改本研究marker，保护独立RACE。",
       'AFLOW_BRATS_METHOD_CATALOG.md':f"## 2026-10-28：12小时无训练推理研究｜running\n\n**一句话：** 固定模型，检验多视图与概率/体积解码及形态组合。\n**出发点：** 在五镜像0.780995基础上继续比较预测信息与解码方法。\n**具体做法：** 联网调研、冻结有限网格、核几何/数值门、完整270全局选择及掩膜审计。\n**结果：** {brief}\n**是否支持直觉：** {limits}\n**代价/下一步：** {running} {fresh_text} [完整记录]({link})。",
       'AFLOW_BRATS_RESULTS.md':f"## 2026-10-29：固定171权重的无训练搜索｜中间结果\n\n{common}",
       'AFLOW_BRATS_WEEKLY_REPORT.md':f"## 2026-10-29：无训练自动研究进展\n\n{common}"}
    marker='AUTO_INFERENCE12H_20261028';pattern=rf'<!-- {marker} -->.*?<!-- END_{marker} -->'
    for name,body in blocks.items():
        path=ROOT/name;old=path.read_text();matches=list(re.finditer(pattern,old,re.S));assert len(matches)==1,name
        block=f'<!-- {marker} -->\n{body}\n<!-- END_{marker} -->'
        new=re.sub(pattern,lambda _:block,old,flags=re.S)
        assert re.sub(pattern,'',old,flags=re.S)==re.sub(pattern,'',new,flags=re.S)
        path.write_text(new)
    print(dict(candidate=candidate,scan_dice=metrics['scan_dice'],configs=n,root_markers_updated=5,fresh270=bool(fresh)))

if __name__=='__main__':main()
