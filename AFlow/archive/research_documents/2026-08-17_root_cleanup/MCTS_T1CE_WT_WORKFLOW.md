# BraTS T1ce/WT MCTS 推理 Workflow 搜索

## 范围

这条新实验固定当前三个 no-gray robust residual-refiner checkpoint，只搜索推理时的
组合策略。它不训练权重，不读取已经揭示的 270 例 test，也不改变历史 result、prereg、
freeze 或 handoff。输入仍是 GT-centered、N4、固定 click-3 coarse logit，因此结果只能写成
oracle validation workflow optimization，不能写成自动定位或独立 test。

当前 `0.7004189626` 是 locked reference：3 seeds × 8 reflections 的 float32 logit mean，
raw-logit threshold `-1.0`，再删除 26-connectivity 下小于 500 voxels 的组件。它已在同一
108 例 validation 上参与选择，所以 MCTS 的 nested split 只能约束新增 workflow 选择，
不能把 validation 变成 pristine evidence。

## 与旧 AFlow 搜索的区别

旧 AFlow 优化器从 top workflows 做 uniform/softmax 混合抽样，每轮只生成一个子流程；
它没有节点 `N/W/Q`、rollout 或 ancestor backup，准确名称只能是 UCB-guided workflow
tree search / MCTS variant。新实现位于 `sammed3d/t1ce_wt_mcts.py`，包含：

- typed、canonical、可哈希的 `WorkflowSpec`，不执行 LLM 生成的任意 Python；
- 标准 UCT selection、stochastic rollout、progressive widening 和祖先 reward backup；
- 每个节点的 visits/value sum/mean value，以及确定性的 RNG/state resume；
- 3 个 outer subject folds；每折内部再拆成 MCTS search 和 inner selector；
- workflow 只接收 logits，label 只进入独立 scorer；
- checkpoint/cache/source/action-space SHA 和逐 simulation ledger。

## 搜索空间

固定策略可选择 1–3 个 robust models，以及 identity、identity + 三个 single flips 或全部
8 reflections。自适应策略先运行三个 identity models，用 prediction-only disagreement
决定在单病例上停止，或继续到 12 / 24 个 member-view forwards。

随后搜索：

- float32 logit mean、probability mean 或 logit median；
- raw-logit threshold；
- voxel disagreement 调整后的 threshold；
- 26-connectivity minimum-component filter。

reward 同时保留 scan-macro、subject-macro、相对 locked reference 的 subject-paired
标准误、平均 forwards 和 workflow complexity。标量 reward 用于树搜索，但最终报告必须
同时看无惩罚 Dice、paired delta 和计算量。

locked `0.700419` workflow 是每折 mandatory anchor：它计入 128 个 unique workflow
预算，但不伪造任何 tree visit。inner selector 因此总能选择“保留 locked baseline”，不会
被迫从全部劣化的随机 rollout 中选一个。

## 第一步：GPU 上一次性物化 atoms

默认 float32 cache 约 20.25 GiB。当前卷剩余空间应先用 `df -h` 核对。不要把唯一 cache
放在易失 `/tmp`。命令可断点续建；完成前没有可供搜索加载的 `index.json`。

```bash
bash scripts/run_worker_research.sh cache-mcts-atoms-t1ce-wt \
  --output-dir workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32 \
  --dtype float32 \
  --device cuda:0
```

runner 会验证当前 validation cache SHA、三个 checkpoint SHA，按固定顺序生成 24 个
inverse-reflected logits，并重新执行 locked reference。scan-macro、subject-macro 和逐病例
Dice 都通过 parity gate 后才发布 cache。同一 A100/BF16 workflow 独立重放可在边界 voxel
产生微小差异，因此 formal gate 按
[`MCTS_ATOM_PARITY_PREREG_20260813.md`](MCTS_ATOM_PARITY_PREREG_20260813.md)
冻结 aggregate `1e-5`、per-case `5e-4` 两级容差；原先更严 gate 的失败收据保留，不会
改写。若已有 prereg 中记录的完整失败 cache，运行：

```bash
bash scripts/run_worker_research.sh promote-mcts-atoms-t1ce-wt
```

promotion 不接受 tolerance 参数，并会重新核所有 SHA 和逐病例结果。float16 只可作为另行
预注册的 compact proxy，正式 MCTS 默认拒绝。

## 第二步：CPU 上运行三折 MCTS

```bash
bash scripts/run_worker_research.sh search-mcts-workflow-t1ce-wt \
  --atom-cache workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32 \
  --output-dir workspace/BraTS/sammed3d_t1ce_wt/mcts_workflow_20260813_v1 \
  --outer-fold all \
  --unique-workflows 128 \
  --max-simulations 512 \
  --selector-top-k 8 \
  --memory-cache-gib 8
```

若任务中断，用完全相同参数加 `--resume`。每折输出：

- `subject_roles.json`：subject-disjoint 三个角色及 SHA；
- `mcts_state.json`：完整树、N/W、RNG、已评估 workflow，可恢复；
- `events.jsonl`：每次 selection/expansion/rollout/cache/backup；
- `fold_report.json`：search top-k、inner selector winner、一次 outer evaluation。

自适应 gate 的 prediction-only 指标是成员概率图之间的 squared soft-Dice disagreement；
相同但未饱和的预测仍定义为零分歧，不把置信度本身混进成员差异。

根目录 `final_report.json` 汇总 cross-fitted out-of-fold 开发指标和三个 fold winner 的
canonical medoid，后者只作为待冻结 deployment candidate，不在全 validation 上再次选择。

## 研究判定

MCTS 搜出的 workflow 若只提高惩罚 reward、但没有稳定的 outer subject-paired Dice 增益，
不能宣称分割性能创新。建议后续用相同 unique-workflow/case budget 增加 random search 和
greedy coordinate search 对照。若自适应 workflow 保持 Dice 且显著降低平均 forwards，
创新点应表述为 uncertainty-gated adaptive robust inference；MCTS 是可审计的 workflow
optimizer。完整协议冻结后，最终泛化结论必须来自新的 external confirmation set。

首轮正式三折结果及其负结果边界见
[`MCTS_T1CE_WT_RESULT_20260813.md`](MCTS_T1CE_WT_RESULT_20260813.md)。
