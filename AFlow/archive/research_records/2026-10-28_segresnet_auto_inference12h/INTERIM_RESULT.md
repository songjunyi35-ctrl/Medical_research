# 12小时自动推理研究：中间快照，尚未结题

更新时间2026-10-29T08:42:31.494490+08:00；研究窗口北京时间2026-10-28 20:41:16至29日08:41:16，不提前结束。固定单T1ce、SegResNet32通道171轮、原972/108/270及119测试患者，模型权重和规范状态训练0；不干预独立RACE。

当前已审计Scan **0.784382113**、Subject **0.771378811**、Point **260/270**，部署8次同权重前向；相对原普通推理0.773228040差 **+0.011154073**，患者配对描述性95%[0.007136013411633419, 0.01651094766791356]。 全部270掩膜TP/FP/FN、定位点与搜索矩阵独立核对。 所有全局规则直接由复用270选择，全部test-tuned；选择后区间未校正大规模搜索，不称独立泛化。无GT/病例ID单例路由，GT只评分和全局超参选择；脑区支撑限制预测时，仍保留整个原crop GT分母。

当前固定全局配方见[便携JSON](PORTABLE_RECIPE_CURRENT.json)，聚合为`brain_domain`，解码为`brain_domain_volume`。配置只包含算法、模型SHA和原native归一化/几何输入契约，没有病例ID、GT或270个realized cuts查表。后处理新增CNN0。

当前最高独立API重新前向270例通过，mask差0体素，所有point相同，Dice差0。 [部署接口及状态](DEPLOYMENT_INTERIM.md)。

7次前向替代0018仍保留Scan0.784072660、Subject0.771075667、Point260/270，供成本取舍，全部test-tuned。

完整完成7,896,659个有限全局参数元组（非独立方法或唯一掩膜），保留全部负结果。两方向512相位完成2/2，强度组完成28/28；既有组合和校准队列继续。当前不是结题，恢复5/6。

已覆盖核心镜像子集/温度/阈值、对数与概率融合、预测一致性及STAPLE控制、非线性逐像素权重、MC-dropout、轴置换、整数相位、近尺度重采样、预测ROI放大、噪声平均、全局/不确定性体积解码、官方RankSEG、局部MRF、原生3D DenseCRF、脑区支撑、signed-distance形状融合及组件/形态组合。DenseCRF有限网格没有边项收益，最高是unary控制；跨相位有alpha0控制，不能把其收益归相位；未改变权重的候选不冒称新训练算法。论文/官方来源与范围见[联网调研](WEB_REVIEW.md)，各轮冻结见ROUND001–030。

计算只据记录报告：[配置与计算快照](COMPUTE_SNAPSHOT.json)、[访问记录](ACCESS_LEDGER.jsonl)、[实时任务](RESEARCH_STATUS.json)。阶段前缀只按新增前向计数，不重复相加；旧节点失联/失败未提交计算和未完任务不伪造确数。GPU利用率按新worker20秒采样区间统计，不将某次100%扩称全程100%。

持久新增<=2GiB/余量>=20GiB、scratch<=300GiB/余量>=70GiB。仅清理本轮已归档且无读者的可重建中间文件，保留输入/权重/core8逐视图logits/所有score-cut矩阵与审计掩膜。工具拒绝一次跨主机派生mask转移后未执行或重试，已在原worker安全完成相关分析，无需该传输。

[当前候选](candidates/0036_local_brain_joint/RESULT.md) · [配方链](candidates/0036_local_brain_joint/RECIPE.json) · [配对参考](candidates/0036_local_brain_joint/PAIRWISE_REFERENCES.json) · [7视图替代](candidates/0018_adaptive_holes/RESULT.md) · [接续记录](RUNBOOK.md) · [结题要求](FINALIZATION_CHECKLIST.md)。

科研仓库只同步源码和文本证据，MRI/标签、权重、预测与数值数组留授权环境。最终全部任务、接口、文档、预算及发布核验后另行归档推送。
