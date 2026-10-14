## 搜索随机性敏感度：20种子已完成，主实验保持不变

动机：单个搜索种子不够判断算法。仅改变random/UCT随机种子，固定模型、128子集、患者三组、16查询/8候选与联合门；同78患者重复分析，不是新盲测。

random：模型11/12各通过12/20、12/20，双模型同时通过8/20。

uct：模型11/12各通过17/20、12/20，双模型同时通过10/20。

完整80结果/240折轨迹均保留，不择最好seed，不改穷举主实验；频次不是显著性检验，不支持稳定树搜索优越性。0训练/GPU/模型前向、新影像标签或保护队列访问，无新大cache。下一步论文主张保持精度—成本权衡，最终workflow/确认另冻，不能冒称全论文已完成。

[七问](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/IDEA.md)、[结果](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/RESULT.md)、[机器汇总](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/RESULT.json)、[核验](archive/research_records/2026-09-11_workflow_search_seed_sensitivity/VERIFICATION.json)。
