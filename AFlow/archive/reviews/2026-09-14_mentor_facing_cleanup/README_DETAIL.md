# 清理前技术资料：README.md

> 2026-09-14整理前的完整技术快照。所有“当前最好”“运行中”“下一步”均属当时状态，不可据此恢复实验；当前阅读入口请看[导师汇报](../../../AFLOW_BRATS_WEEKLY_REPORT.md)和[日期流水账](../../../AFLOW_BRATS_RESULTS.md)。原字节另有压缩备份；这里仅调整相对链接以便归档后阅读。

# AFlow BraTS T1ce/WT research workspace

本仓库当前聚焦BraTS 2024 post-treatment GLI：单通道T1ce、全自动二分类WT=`{1,2,3}`分割。

## 新会话从这里开始

按顺序阅读：

1. [AGENTS.md](../../../AGENTS.md) — Codex必须遵守的科研、文档、test与Leaderboard流程。
2. [AFLOW_BRATS_HANDOFF_CURRENT.md](../../../AFLOW_BRATS_HANDOFF_CURRENT.md) — 当前状态、最佳模型、已关闭方向和下一候选。
3. [AFLOW_BRATS_LEADERBOARD.md](../../../AFLOW_BRATS_LEADERBOARD.md) — 同口径Scan/Subject/Point排行榜。
4. [AFLOW_BRATS_METHOD_CATALOG.md](../../../AFLOW_BRATS_METHOD_CATALOG.md) — 小白版方法百科：为什么做、怎么做、结果和结论。
5. [EXPERIMENT_IDEA_TEMPLATE.md](../../../EXPERIMENT_IDEA_TEMPLATE.md) — 新方法开跑前必须填写的通俗idea模板。

## 当前结论

- 内部test270总体最高点估计：固定`0.75 history168 + 0.25 EMA168` logit ensemble，Scan Dice `0.745054`，2× locator推理。
- 当前最高单模型点估计：2025/2026 voxel-KD global epoch200，Scan Dice `0.742976`，1× locator推理。
- internal test270已多次揭示；这些是项目内部固定读数，不是pristine blind test、外部确认或公开SOTA。

## 详细历史

- 精简结果账本：[AFLOW_BRATS_RESULTS.md](../../../AFLOW_BRATS_RESULTS.md)
- 历史周报：[AFLOW_BRATS_WEEKLY_REPORT.md](../../../AFLOW_BRATS_WEEKLY_REPORT.md)
- 2026-08-13长交接：[AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md](../../../AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)
- 2026-08-19阶段handoff/idea/旧plan：[archive/handoffs_20260819/](../../handoffs_20260819)
- 更早handoff：[archive/handoffs_pre_20260813/](../../handoffs_pre_20260813)
- 自包含实验记录：[archive/research_records/](../../research_records)
- 其他研究文档：[archive/research_documents/](../../research_documents)
- 顶层旧`.orig`备份：[archive/legacy_root_backups_20260819/](../../legacy_root_backups_20260819)

## 强制文档校验

```bash
PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python scripts/verify_aflow_brats_leaderboard.py
PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python scripts/verify_aflow_brats_method_catalog.py
```

test测评完成但Leaderboard未更新，或新方法缺少通俗解释时，任务不算完成。
