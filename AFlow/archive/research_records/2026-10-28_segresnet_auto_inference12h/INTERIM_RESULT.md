# 12小时自动推理研究：中间快照，尚未结题

时间2026-10-29T00:50:48.981529+08:00；研究窗口2026-10-28 20:41:16至29日08:41:16。用户固定单T1ce/SegResNet32通道171轮，同原270扫描/119患者，允许直接用270反馈选择全局规则。全部test-tuned，无网络权重训练/outer/confirmation，不干预独立RACE。

当前已独立核验最高：Scan **0.784139146**，Subject **0.771156128**，Point **258/270**。相对原同权重普通推理0.773228040，差+0.010911106，患者配对描述性95%区间[0.006975465464098059, 0.0161204336599086]；对研究前五镜像0.780995198差+0.003143948。大规模选择后区间不是独立确认，也未作多重搜索校正。

当前配方链：`[{"path": "archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0008_consensus_cc/RECIPE.json", "rule": {"id": "c1_min512_protectFalse", "connectivity": 1, "kind": "minimum", "value": 512, "protect_largest": false}, "decoder": null}, {"path": "archive/research_records/2026-10-28_segresnet_auto_inference12h/candidates/0007_consensus_calibration/RECIPE.json", "rule": {"id": "agreement_t0.2_logit_w16.0", "kind": "consensus", "mode": "agreement_weight", "vote_threshold": 0.2, "representation": "logit", "exponent": 16.0, "views": [0, 1, 2, 3, 4, 5, 6, 7], "neural_case_forwards": 8}, "decoder": {"post_temperature": 4.0, "bias": -1.0, "beta": 0.7, "volume_scale": 1.0}}]`。预测不读GT或病例ID路由；GT仅评分和全局参数选择。部署8次同权重前向，后处理新增神经前向0。全部270掩膜TP/FP/FN与搜索矩阵独立精确对齐。点定位规则见链条中基础配方。

7视图替代方案保留Scan0.783849666、Subject0.770681114、Point260/270；当前8视图一致性方案虽提高Scan到0.784139146，Point为258，完整保留代价/指标取舍。


当前756694种有限配置完成，保留所有负结果。1464216种温度/子集/置信度加权组合正在GPU0完整270评分；GPU1继续变换组并将运行完整8³=512整数平移相位，1/64/128/256/512共享前缀，CPU16进程校准。新worker1153287双A10080，原native三项SHA和旧五镜像全部270掩膜/点/分数精确匹配。原节点失联与两次预算遍历竞态有独立记录，当前恢复计数3/6，不能把重算称每例只访问一次。

[当前候选](candidates/0008_consensus_cc/RESULT.md) · [配方链](candidates/0008_consensus_cc/RECIPE.json) · [配对参照](candidates/0008_consensus_cc/PAIRWISE_REFERENCES.json) · [进度](RESEARCH_STATUS.json) · [初始七问](IDEA.md) · [大组合预登记](ROUND014.md) · [继续接手](RUNBOOK.md)。

只同步源码和文本证据；MRI、标签、权重、预测数组与大中间缓存留在授权环境。清理仅本轮新生成、已归档且无读者的可重建数据；输入/权重/核心逐视图logits及所有评分保留。研究仍继续，当前最高不是最终结论。
