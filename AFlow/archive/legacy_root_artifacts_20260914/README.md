# 从根目录移入的旧技术材料

这些是已结束的2026年8月实验候选、清理收据与旧交接，不是当前待办。整理不修改模型、数据、实验结果或原始JSON内容。

## 文件用途

| 原根文件 | 用途 | 现在的位置 |
|---|---|---|
| FIXED1_DEPLOYMENT_CANDIDATE_20260814.json | 旧修正器单次推理候选 | [原文件](FIXED1_DEPLOYMENT_CANDIDATE_20260814.json) |
| MCTS_DEPLOYMENT_CANDIDATE_20260813.json | 旧修正器八视图候选 | [原文件](MCTS_DEPLOYMENT_CANDIDATE_20260813.json) |
| HANDOFF_CLEANUP_20260813.json | 8月13日文档整理收据 | [原文件](HANDOFF_CLEANUP_20260813.json) |
| CODE_ARCHIVE_CANDIDATES_20260813.sha256 | 当时的源码哈希清单，不是删除清单 | [原文件](CODE_ARCHIVE_CANDIDATES_20260813.sha256) |
| AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md | 实际追加至8月18日的历史长交接 | [可读全文](AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md) |

根目录历史交接文件仍存在，但只保留短导航，不再与当前交接争当执行入口。历史长交接的原始字节以gzip备份保存，可读全文仅调整相对链接并添加历史提示。

## 历史脚本兼容边界

没有修改以下已冻结脚本，也没有留下根目录软链接。它们原先默认读取的候选JSON根路径已迁移，**旧脚本的无参默认查找不再可用**。历史复现必须明确指定归档位置，不能把这次整理视为恢复实验的授权：

- `scripts/evaluate_t1ce_wt_adaptive_router.py`：`--deployment-candidate`指定本目录的MCTS候选；`--prereg`另指定`archive/research_documents/2026-08-17_root_cleanup/ADAPTIVE_ROUTER_PREREG_20260814.md`。
- `scripts/summarize_t1ce_wt_fixed1_benchmark.py`：`--candidate`指定本目录的FIXED1候选；`--prereg`另指定`archive/research_documents/2026-08-17_root_cleanup/FIXED1_BENCHMARK_PREREG_20260814.md`。
- 旧搜索脚本曾把根历史交接的SHA写入收据。短导航页不是当时冻结的原文，不能替代旧SHA；原文和迁移前SHA见[迁移收据](MIGRATION_RECEIPT.json)。
- `.sha256`清单中的相对路径仍以仓库根目录为基准。JSON里的旧路径、历史状态和哈希保持原样，不改写成今日状态。

这两份候选属于旧真实标签辅助裁剪修正器，不是当前自动模型的部署配置。当前工作入口为[当前交接](../../AFLOW_BRATS_HANDOFF_CURRENT.md)，向导师汇报看[导师报告](../../AFLOW_BRATS_WEEKLY_REPORT.md)。

## 恢复

四份技术文件是原字节移动，可按迁移映射移回，但不要覆盖后续新增文件。历史交接原文可从`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md.original.gz`解压恢复。全部源/目标SHA及未改动的旧脚本SHA见[迁移收据](MIGRATION_RECEIPT.json)。
