# 2026-08-17 intensity-aware residual / inner split 科研归档

## 归档结论

- 当前 intensity-aware residual 假设正式判负，不晋级 outer selector。
- intensity 相对参数匹配 control：scan `-0.002638`、subject `-0.003331`、point hit
  `-0.370370`。
- 5 个预注册门槛失败 3 个；selector、confirmation 和 270-case test 均未访问。
- 当前 benchmark checkpoint 仍是既有 epoch99 direct plain locator。

本目录是逻辑归档索引。checkpoint、训练收据和逐病例 JSON 保留在原路径，不复制、不移动；
核心文件校验值见 [`MANIFEST.sha256`](MANIFEST.sha256)。

## A. 权威叙述

| 作用 | 活跃文件 |
|---|---|
| 完整结果、失败机制和证据边界 | [`INTENSITY_RESIDUAL_RESULT_20260817.md`](../../research_documents/2026-08-17_root_cleanup/INTENSITY_RESIDUAL_RESULT_20260817.md) |
| 实验前冻结协议 | [`INTENSITY_RESIDUAL_PREREG_20260817.md`](../../research_documents/2026-08-17_root_cleanup/INTENSITY_RESIDUAL_PREREG_20260817.md) |
| handoff 第23节 | [`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`](../../../AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md) |
| 精简结果账本 | [`AFLOW_BRATS_RESULTS.md`](../../../AFLOW_BRATS_RESULTS.md) |

## B. 数据与隔离

| 类别 | 路径 | 关键信息 |
|---|---|---|
| subject-clean split | `data/datasets/brats2024_t1ce_intensity_inner_seed20260820_v1/` | 648 fit / 108 tune；case、subject overlap 0 |
| persistent train cache | `workspace/BraTS/cache/t1ce_wt_n4_braincrop_128_20260817/train/` | 只使用历史 train role |

inner 决策收据明确记录 `outer_selector_access=false`、`confirmation_access=false`、
`test_access=false`。

## C. 模型与输出

根目录：
`workspace/BraTS/sammed3d_t1ce_wt/intensity_residual_inner_20260817/seed20260821/`

| arm | best epoch | checkpoint 子目录 | scan/subject |
|---|---:|---|---:|
| plain | 79 | `plain/` | `0.728322/0.713027` |
| residual control | 93 | `residual_control/` | `0.729545/0.717166` |
| intensity residual | 98 | `intensity_residual/` | `0.726907/0.713835` |

核心推理收据：`INNER_RELOAD_plain.json`、`INNER_RELOAD_residual_control.json`、
`INNER_RELOAD_intensity_residual.json`；最终门槛收据：`INNER_DECISION.json`。三臂重载最大
绝对差均为 0。

## D. 代码入口

| 作用 | 文件 |
|---|---|
| 模型与灰度特征 | `sammed3d/t1ce_wt_intensity_residual.py` |
| 新 inner split | `scripts/build_t1ce_wt_intensity_inner_split.py` |
| 三臂训练 | `scripts/train_t1ce_wt_intensity_residual.py` |
| 角色隔离重载 | `scripts/evaluate_t1ce_wt_intensity_residual.py` |
| 配对 bootstrap 与 gates | `scripts/analyze_t1ce_wt_intensity_residual.py` |

## 后续限制

- 不得在本轮 108 inner-tune cases 上继续搜索强度 cut-off、kernel、residual scale 或输出阈值。
- 因 gates 失败，不得打开旧 108 selector；confirmation 和 270 test 也不得用于本假设。
- 若研究 residual-only control 或多亚区强度建模，必须另做预注册并使用新的 subject-clean
  split 或外部 cohort。
