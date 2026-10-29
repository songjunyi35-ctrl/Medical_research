# 新增推理与结题链：独立CPU审查

未修改A/B或训练源码，未读正在覆盖的latest，未执行模型前向、未读患者数组、未占GPU。只读实际worker的两个小READY副本和index元数据；其余为CPU合成及已有逐例整数报告。代码SHA、可重复检查和事实见CPU_REVIEW_CHECKS.json、FINDINGS.json。

## 算法、统计与缓存

A的other7多数票分歧、局部8×8二阶矩、ridge求解后截负归一、物理stride4格点展开、分歧/低置信区域的有界更新与16冻结项一致。三轴有效box平均等同声明的3D平均，CPU最大误差1.19e-7；置换差2.38e-7。该代理不是GT错误概率，截负归一也不是带非负约束QP的精确解；代码和文档没有作这些过强声称。

B对sigmoid(z/4)中心化扰动的方差/交叉协方差、同一非负窗口平均、lambda锚点正则与闭式alpha推导一致；alpha界、零方差/奇异回退、连续冲突门均吻合12配置。其闭式式子最小化所声明的二次代理而非Dice；奇异回退是额外明示规则。CPU全12项通过，闭式目标与101个替代alpha比较通过，配对打乱改变协方差权重但不改变两个模型均值。

两路在两个不同奇数crop上的返回Point均精确为最终score argmax；A的equal8 Point仅辅助另列，没有择GT更优点。部署不接GT/ID。image当前用于shape合同，两个算法没有额外脑掩膜裁剪；这与完整原生crop评分一致，不排除脑支撑外GT。输入网络的RACE几何/support变换已经由既有对齐atom产生，本次算法不再改变几何。

独立整数审计覆盖A17组和B12新组，108例顺序/患者/TP/FP/FN/Dice均合法；A equal8、B RACE equal8、w32 equal8、old fixed.75四参考共432行的整数、Dice与Point全部精确相同。全部排名的Scan/Subject/Point重算一致，患者折号与总控STATISTICS_FREEZE完全相同。GT仅进入评分与全局/四折配置选择，没有流入局部alpha或矩计算。固定.75锚点已在全108选过，分折只是条件稳定性，双方已披露。

读取的A/B TMP READY副本与冻结ROOT副本逐字节SHA相同；实际index SHA也与A冻结PLAN一致。B原runtime没有显式绑定这两个副本SHA的审计缺口已向B报告，B已补ACTUAL_MANIFEST_PARITY_AUDIT，当前结果没有因此错误。所有原始atom均只读，GPU融合计算使用新tensor，不回写源cache；统计一次在有限recipe间复用。A已有完整native显存门约13.02GiB低于16GiB；B按冻结38GiB门通过。本审查不伪称重测了CUDA。

## 发现与处置

1. **A负结果部署分支（已由新增wrapper解决）。** 旧algorithm.deploy收到reference_equal8标记会访问缺失box_mm，CPU确实复现KeyError；本次真实赢家joint-k17未触发。A保持冻结算法不改，新deployment.deploy支持equal/reference和内嵌config归一化，CPU复核通过；新的DEPLOYMENT_RECIPE与root selector约定吻合。
2. **root掩膜收据绑定（已报root）。** final_report的ready/complete校验receipt SHA、selection和passed，但未把receipt.raw_sha256与实际canonical raw、wrapper.canonical_raw_sha256及其逐例内容绑定。同一selection下的过期收据理论上可被配到另一报告；建议最终冻结前补这一独立绑定。现有worker包装器自身按正确顺序生成审计，并未发现当前错误掩膜或错误统计。
3. **root历史源码发布缺项（已报root）。** publish_innovation沿用publisher的archive扩展名过滤，仅上传.md/.json/.jsonl，training/engineering内为了保留旧freeze而存的.py快照不会上传；远端source-archive映射将缺代码字节。应在新发布入口纳入这些Python文本快照，继续排除pth、npz和影像；不改旧publisher或训练冻结。

## 新selector与结题合同

当前合法合同下没有发现必然阻断的selector错误。四类单模/四类融合以完整108选项比较；best0只保留旧模型选项并等待总控NO_NEW_MODEL标记，跳过新snapshot。best>0须等待四个终态108结果，并检查候选checkpoint SHA、配方ID、伙伴一致。model工厂只允许RACE/SegResNet三种固定身份并逐SHA核验；最终恰单模1和融合2模型，各8/16前向。stage_guard额外计入新A记录和脚本目录，未漏该阶段持久增量。既有两个原始模型缓存共享，root最终runtime按model_id去重计新增网络成本。

建议冻结前加紧当前已吻合但未全部显式assert的合同：A config等于实际108 recipe、B candidate等于RESULT.best.recipe，HANDOFF总轮/更新/actual范围与终态80一致，旧global两组选项的reported metrics也从对应RAW ID比对。它们是完整性硬化，不是已发现当前数值错误。A/B已被提醒终态若best仍0应写显式skip，不等待不存在的新模型cache，也不重复108/270网络。

结题链仍将单模与纯SegResNet融合分别报告，保留270逐mask重载审计、整数重算、患者整簇区间及变差病例；新result包含实际算法及108结果SHA，新增模型前向按去重模型分别2160，复用RACE为0。报告应等待真实发布回执后才声称代码已经推送。

审查期间训练仍原PID/原80，无任何改动。17:17已完整41轮且best仍新增0轮（初始化Scan0.763266263），当前尚未胜初始化；gate约.07不是收益证据。该事实不是终态失败结论，不触发停训、换参或第二训练。终态若仍e0，报告应明确80训练未选出新权重，最终推理使用既有权重而非暗示新微调模型已获益。

## 后续V2复核

总控保留旧冻结入口，新建published_raw、final_report_innovation_v2/watch_v2及publish_innovation_v2，已解决上列掩膜收据绑定和历史Python源码发布缺项。独立复跑test_published_raw四项通过，覆盖同selection错收据、wrapper篡改拒绝、legacyB和新A配置审计两种schema。V2快审未发现阻断，源码SHA及复核结论见RESOLUTIONS_V2.json。没有修改或重新启动训练。
