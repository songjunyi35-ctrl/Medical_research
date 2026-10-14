# MedTAWS x SAM-Med3D (BraTS2024) 历史适配器

> **2026-08-10 最终覆盖：本文件是历史账本，不是执行入口。当前唯一主线是
> T1ce -> WT={1,2,3}；请以 `T1CE_MAINLINE.md`、
> `AFLOW_BRATS_HANDOFF_20260807.md` 和
> `AFLOW_BRATS_HANDOFF_20260810_ADDENDUM.md` 为准。**

> **2026-07-20 状态警告：本页以下内容是旧单模态、GT corrective-click、binary-target
> 适配器的历史说明，不是当前科研入口，也不得继续运行来生成论文主结果。** Work #1 的
> 搜索算法保留并更名为 MedTAWS，但医学任务接入正按四模态联合输入、无 GT prompt、自动
> 五类输出和 BraTS lesion-wise 指标重建。当前执行口径见 `AFLOW_BRATS_HANDOFF.md` 第 0 节
> 与最新追加节；旧 `round_1` 至 `round_31` 仅作 provenance。

历史实现曾用 AFlow 的 stochastic UCB-guided workflow tree search 搜索 SAM-Med3D 推理
工作流。它没有 rollout、真实 `N/W/Q` 或 ancestor backup，因此不能称为标准 MCTS。

## 思路与映射

AFlow 本是优化「LLM 智能体工作流」的。这里把它的**思想**接到分割上：
- **优化器 LLM**（opt model）= 每轮改写工作流代码的「大脑」（唯一真正用到的 LLM）。
- **工作流节点** = 分割 operator（不调 LLM，包 SAM-Med3D 推理）：
  - `SamPredict`：迭代点击 3D 推理 → 概率图（≈ Review/Revise，点数/采点法/seed 可调）
  - `MaskEnsemble`：多次推理概率图软投票平均（≈ ScEnsemble，降方差，对小结构最有潜力）
  - `PostProcess`：阈值 + 最大连通域
- **打分** = 预测 mask 对金标准的 **Dice**（`benchmarks/brats.py`）。

`round_1` 种子工作流 = **单次 5 点击预测**，即 AFlow 框架内的 baseline；它和后续轮次在**同一验证集**上可直接对比。

## 评测对齐边界

必须区分两种 Dice：

- **SAM-Med3D 官方 validation.py 指标**：同一 case 内记录每次 click 后的 Dice，最后取 `max(dice_list)`。
- **AFlow 搜索指标**：只对 workflow 最终返回的单个 mask 算 Dice。

当前 `round_1` 是 final-click mask，不是官方 max-over-clicks 指标。已加
`scripts/archive/audit_brats_eval_alignment.py`，用于显式检查两者差异：

```bash
MPLCONFIGDIR=/tmp/matplotlib-aflow python scripts/archive/audit_brats_eval_alignment.py --split validate
```

当前 15 例 validation split 上：

- final-click mean Dice: `0.31247`
- SAM-Med3D official max-click mean Dice: `0.33024`
- 差值: `+0.01777`

所以论文或汇报里不要把 AFlow 的 workflow Dice 直接写成 SAM-Med3D 官方
validation Dice；如果要对标官方结果，必须用 max-over-clicks 口径，并注明
当前 split 只有 ET 的 45 例子集，不是官方本地结果里的 145 例全集。

## 新增/改动的文件

| 文件 | 作用 |
|---|---|
| `data/make_brats_split.py` | 生成 `data/datasets/brats_{validate,test}.jsonl` |
| `sammed3d/engine.py` | SAM-Med3D 推理封装（模型单例、单 case 加载、预测、集成、后处理、Dice） |
| `scripts/archive/audit_brats_eval_alignment.py` | 已归档诊断脚本：检查 final-click 与官方 max-click 指标是否混淆 |
| `benchmarks/brats.py` | `BraTSBenchmark`（Dice 打分，GPU 串行 max_concurrent=1） |
| `scripts/evaluator.py` | 注册 `BraTS` |
| `run.py` | `EXPERIMENT_CONFIGS["BraTS"]`，并对 BraTS 跳过远程下载 |
| `workspace/BraTS/workflows/template/{operator.py,operator.json}` | 分割算子 + 给优化器的描述 |
| `workspace/BraTS/workflows/round_1/{graph.py,prompt.py}` | 种子工作流（baseline） |
| `config/config2.yaml` | 内部 LLM 代理配置（**待填**） |
| `scripts/archive/smoke_brats.py` | 已归档 smoke 脚本：无需 LLM 的 GPU 自检 |

## 运行步骤（小规模验证）

```bash
cd /mlx_devbox/users/songjunyi.666/playground/AFlow
source ../sammed3d-venv/bin/activate     # 已装好 SAM-Med3D + AFlow 全部依赖

# 1) 数据 split（已生成 ET：val=15 / test=30；换目标改 --targets）
python data/make_brats_split.py --targets tumor_ET --n_val 15 --n_test 30

# 2) 填 config/config2.yaml：把两个条目的 KEY 改成代理上真实模型名，
#    填 base_url / api_key。注意 AFlow 用 YAML 的 key 当模型名。

# 3) GPU 自检（确认分割 pipeline 通；不需要 LLM）
python scripts/archive/smoke_brats.py --n 3
#    期望输出每例 baseline/ensemble Dice，并打印 SMOKE_OK

# 4) 启动搜索（小规模）
python run.py --dataset BraTS \
  --opt_model_name  <OPT_MODEL_NAME>  \
  --exec_model_name <EXEC_MODEL_NAME> \
  --max_rounds 10 --validation_rounds 1 --sample 4 --check_convergence true
```

### 看结果
- 每轮验证集 Dice：`workspace/BraTS/workflows/results.json`（`round_1` = baseline）。
- 各轮生成的工作流代码：`workspace/BraTS/workflows/round_N/graph.py`。
- 在 **test 集**上评测最佳轮：把 `run.py` 末尾的 `optimizer.optimize("Test")` 取消注释，并在 `scripts/optimizer.py::Optimizer.test()` 里把 `rounds=[1]` 改成最佳轮号。

## 成本与注意

- 每轮 ≈ 验证集(15) × 工作流内推理次数（集成会成倍）次 GPU 前向；`max_rounds=10` 大概几小时（单卡）。
- GPU 推理串行（`max_concurrent_tasks=1`），模型只加载一次，避免 OOM。
- 真正的提升来自 operator（集成 / 迭代点击）；AFlow 的价值是自动搜出它们的最佳组合与参数。
- 公平对比 = 同一 split 上 `round_1`(baseline) vs 最佳轮，**不要**直接和全量 145 例的历史 Dice 比。
