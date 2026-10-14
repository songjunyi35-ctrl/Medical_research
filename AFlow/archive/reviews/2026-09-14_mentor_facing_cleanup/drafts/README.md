# AFlow：单通道T1ce脑肿瘤分割研究

输入仅为T1ce磁共振图像；目标为整体肿瘤，包含标签1/2/3，不包含切除腔标签4。

## 我要向导师汇报，先看哪里？

1. **[导师汇报](AFLOW_BRATS_WEEKLY_REPORT.md)**：起点是什么、做了什么优化、Dice达到多少、哪些结论成立。
2. **[从8月13日起的日期流水账](AFLOW_BRATS_RESULTS.md)**：按日期连续读完整研究过程。
3. **[方法成败简表](AFLOW_BRATS_METHOD_CATALOG.md)**：每个主要尝试的直觉、结果和下一决定。

## 当前成绩

同一复用内部270扫描／119患者：全自动总体Scan Dice最高 **0.757270**（一个模型、八次前向）；单次前向Scan最高 **0.752003**。前者点命中265/270，后者252/270。不是官方隐藏榜或新的盲测。

师兄路线基础模型复现为 **0.594226**，但采用真实标签辅助定位和交互点击；与全自动成绩不是同条件对照。

## 我要核对证据或继续实验

- [当前接续状态](AFLOW_BRATS_HANDOFF_CURRENT.md)
- [分队列排行榜与置信区间](AFLOW_BRATS_LEADERBOARD.md)
- [原始研究证据索引](archive/reviews/2026-09-14_mentor_facing_cleanup/EVIDENCE.md)
- [执行与文档规范](AGENTS.md)
- [新方法通俗预注册模板](EXPERIMENT_IDEA_TEMPLATE.md)

技术路径、完整哈希、运行日志和历史阶段状态放在资料层；导师汇报正文不再复制这些内容。
