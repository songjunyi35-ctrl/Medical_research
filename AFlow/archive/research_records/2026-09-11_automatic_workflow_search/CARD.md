## 自动工作流搜索与剪枝：开发—执行闭环完成，主门通过

一句话：以预先允许的小幅Dice损失减少重复镜像。只改视图子集；模型、权重、128网格、融合、阈值0和argmax固定。两个native模型、复用dev180/78患者三组轮换搜索/选择/评估，穷举主参照及随机/贪心/UCT等预算对照。

Seed 2026090711：OOF Scan/Subject=0.750404732/0.736766902，Point172/180，平均2.0×。相对full8 Scan差-0.001125、Subject差-0.001284、Point+1，实测阶段加速3.961倍。

Seed 2026090712：OOF Scan/Subject=0.747436134/0.733170501，Point166/180，平均3.0×。相对full8 Scan差-0.001428、Subject差-0.002003、Point-1，实测阶段加速2.652倍。

支持受限精度—成本权衡，不是无损或显著不降；部分full8差值CI整体为负。随机seed11和UCT seed12点门失败，贪心两seed通过，不证明树搜索优越。原定主方法两seed全部精度/点/成本门与在线门通过，不换主方法救结果。

完整46080候选病例评分由独立NumPy重算exact，16项所选在线/原子parity exact；零训练，开发2880前向、fit smoke496、在线1675前向，/tmp原子22.5GiB保留。无test/保护outer/confirmation/固定validation/外部访问；没有单一最终部署工作流或新盲测确认。下一步补搜索随机性敏感度，不恢复相似增强训练。

[七问](archive/research_records/2026-09-11_automatic_workflow_search/IDEA.md)、[完整结果与CI](archive/research_records/2026-09-11_automatic_workflow_search/RESULT.md)、[机器结果](archive/research_records/2026-09-11_automatic_workflow_search/RESULT.json)、[独立验证](archive/research_records/2026-09-11_automatic_workflow_search/VERIFICATION.json)。
