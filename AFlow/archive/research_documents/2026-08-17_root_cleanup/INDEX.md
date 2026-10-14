# AFlow 根目录 Markdown 归档（2026-08-17）

## 目的

本次整理把 AFlow 根目录从 36 份 Markdown 缩减为 4 个关键入口：

1. `README.md`：仓库入口；
2. `AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`：权威科研交接；
3. `AFLOW_BRATS_RESULTS.md`：精简结果账本；
4. `AFLOW_BRATS_WEEKLY_REPORT.md`：周报。

其余 32 份文档仅移动到本目录，没有删除或改写科研内容。原路径和新路径的精确映射见
[`MOVE_MAP.tsv`](MOVE_MAP.tsv)。本目录中的四个同名符号链接指回根目录关键入口，用于保持
历史文档之间的相对链接。

## 分类

| 分类 | 数量 | 内容 |
|---|---:|---|
| 2026-08-17 当前自动主线 | 5 | plain locator confirmation/test、intensity residual 预注册与结果 |
| 2026-08-14 架构与推理实验 | 16 | CAPG、SAGE、ELPGR、safe/boundary adapter、router、fixed-budget |
| 2026-08-13 MCTS | 5 | workflow、atom parity、效率贡献预注册与结果 |
| 历史交接与维护 | 6 | robust refiner 说明、旧 T1CE mainline、迁移和清理审计 |

## 当前科研状态

- benchmark 仍是 epoch99 direct plain locator：已揭示 270-case internal test scan
  `0.725820`、subject `0.709413`；
- intensity-aware residual 在新 subject-clean inner split 正式判负，没有打开 selector、
  confirmation 或 test；
- 权威状态以根目录 handoff 和结果账本为准，本目录只用于追溯。

## 恢复

所有移动都是可逆的。需要恢复单份文档时，根据 `MOVE_MAP.tsv` 将对应 `archived_path`
移动回 `original_path`；不要批量恢复四个指向关键入口的符号链接或本索引。
