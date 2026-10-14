# 硕士论文贡献：精度约束的三维分割推理 Workflow 自动优化

## 最终判断

这部分可以作为硕士论文的一个独立贡献，但不应写成“提出 MCTS 并提高了 Dice”，也不能写成
“MCTS 优于随机或贪心搜索”。完成等预算对照后，证据支持的准确定位是：

> 针对三维医学分割中多模型、多视图集成推理成本高的问题，构建了一套精度约束、可审计的
> 推理 workflow 自动优化框架。该框架用强类型 grammar 组合模型子集、测试时视图、融合、
> uncertainty threshold 和连通域后处理，用 subject-disjoint nested validation 控制新增选择
> 偏差，并用 MCTS、random 与 greedy 控制器搜索 Dice—计算量 Pareto 解。在当前 BraTS
> T1ce/WT oracle validation 上，搜索过程可将 residual-refiner member-view forwards 从 24
> 降至约 6.7–8，Dice 仅有很小变化；冻结的 8-forward 工程候选在双 A100 交叉实测中获得
> 2.779× refiner-stage 加速。

这是“方法和系统贡献”，而不是“新的 MCTS 理论”或“新的 Dice SOTA”。对于不冲击顶会的硕士
论文，这一贡献有完整的问题、方法、消融、真实工程收益和负结果边界，成立。

## 为什么原始 MCTS 结果还不够

原始三折 MCTS 的 OOF subject Dice 为 `0.695037`，locked reference 为 `0.695373`，配对差
`-0.000336`，95% CI `[-0.001989, +0.001365]`；member-view forwards 从 24 降为 7。
这只能证明搜索到了精度—成本 Pareto workflow，不能证明 Dice 提高。

复核还发现两点：

1. 三个 winner 都是 fixed workflow，不是自适应 `3→12→24` early exit。因此当前收益主要
   来自裁剪冗余模型和翻转视图，而不是按病例动态分配算力。
2. locked anchor 虽进入 search evaluation cache，但旧 runner 只把 search reward 前 8 名送入
   selector。anchor 在三折 search 中排名 39、89、6，前两折实际没有进入 selector。旧结果和
   SHA 均未改写；新效率协议已把“anchor 必须占一个 selector 名额”写成代码和测试。

此外，旧 runner 设置 `max_simulations=512`，但达到 128 个 unique workflows 后就停止；三折
实际只有约 128–129 次 root visits。当前实现是真实 UCT MCTS，但没有充分利用“多次树模拟、
有限次昂贵评价”的潜力。这是后续算法版本应改的地方。

## 本轮完成的优化

### 1. 精度约束和 Pareto 选择

新增独立模块 `sammed3d/t1ce_wt_mcts_efficiency.py`，不修改已被旧 cache 和报告 SHA 绑定的
`sammed3d/t1ce_wt_mcts.py`。新增内容包括：

- locked anchor 强制进入 selector；
- Dice—member-forward nondominated frontier；
- `0.000 / 0.001 / 0.002 / 0.005` 四个预注册精度容差档位；
- 在达到相对 locked accuracy floor 后选择最省算力 workflow；
- 对应单元测试 4/4 通过。

容差是开发选择规则，不等同于统计学 non-inferiority；outer OOF 仍单独做 subject-paired
bootstrap。

### 2. 等预算控制器消融

三个控制器使用完全相同的：

- 3 个 subject-disjoint outer folds；
- 每折 36-case search、36-case selector、36-case outer；
- 强类型 workflow grammar；
- 128 unique workflows；
- 8 个 selector candidates；
- mandatory locked anchor；
- exact float32 24-atom cache；
- 50-subject、10,000 次 paired bootstrap；
- 270-case test 不可达。

严格 `epsilon=0` 的 OOF 结果：

| Controller | OOF subject Dice | 相对 locked 配对差 | 95% CI | 平均 forwards | 减少 |
|---|---:|---:|---:|---:|---:|
| MCTS | `0.695037` | `-0.000336` | `[-0.001989, +0.001365]` | `7.00` | `70.8%` |
| Uniform random | `0.695746` | `+0.000373` | `[-0.001139, +0.001907]` | `8.00` | `66.7%` |
| Greedy best-first | `0.695016` | `-0.000357` | `[-0.002569, +0.001715]` | `6.67` | `72.2%` |
| Locked reference | `0.695373` | — | — | `24.00` | — |

控制器间配对结果：

- MCTS minus random：`-0.000709`，95% CI `[-0.002046, +0.000611]`；
- MCTS minus greedy：`+0.000021`，95% CI `[-0.001375, +0.001778]`。

所以 MCTS 没有显著优于两个简单控制器。反过来，这个消融强化了“完整 24 路推理高度冗余”
这一应用结论，也避免把搜索器本身过度包装。

其他容差档位可见最终 JSON。值得注意的点：

- MCTS 在 `epsilon=0.001` 时降到平均 `4.083` forwards，subject Dice `0.694378`；
- MCTS 在 `epsilon=0.005` 时降到 `2.0` forwards，subject Dice `0.693758`；
- random 在 `epsilon=0.001` 时为 `6.667` forwards、subject Dice `0.695909`；
- 控制器差异较小且随 fold 变化，当前不应选择性报告单折 winner。

### 3. 真正按需的在线推理

旧 MCTS 搜索先生成全部 24 个 atoms，因此搜索运行本身不节省 GPU。新增在线 runner 只加载
workflow 选中的 checkpoint，只执行选中的 views；不会先生成 24 个结果再丢弃。

冻结的工程 benchmark candidate 为历史三个 fold winner 的 canonical medoid：seed
`20260815` 单模型、8 reflections、float32 logit mean、threshold `-0.25`、uncertainty beta
`-0.5`、26-connectivity CC `250`。它只为无标签工程 benchmark 冻结，不是已经外部确认的
最终精度模型。

双 A100-SXM4-80GB 交叉设计：

- run 1：candidate 在 GPU 0，locked 在 GPU 1；
- run 2：candidate 在 GPU 1，locked 在 GPU 0；
- 每个 receipt 3 例 warm-up，24 例 × 2 repeats，共 48 个测量；
- 两个策略使用独立进程；
- 不打开 label，不计算 Dice；
- 口径包含 image/coarse H2D、refiner forwards、融合、D2H 和连通域；
- 不包含上游 SAM-Med3D coarse prediction 和预处理。

| 指标 | 8-forward candidate | 24-forward locked | 变化 |
|---|---:|---:|---:|
| 平均延迟 | `0.3871 s/case` | `1.0757 s/case` | 减少 `64.0%` |
| 吞吐 | `2.584 cases/s` | `0.930 cases/s` | `2.779×` speedup |
| 模型加载后 allocated | `18,818,560 B` | `56,455,680 B` | 减少 `66.7%` |
| 推理 peak allocated | `780,058,112 B` | `952,568,320 B` | 减少 `18.1%` |

两个交叉配对 speedup 分别为 `2.763×` 和 `2.796×`。candidate 与 locked 各抽取 8 例进行
online-vs-exact-atom parity，16/16 均逐体素完全一致，minimum mask Dice `1.0`、最大差异体素
`0`。因此加速不是由数值或后处理口径改变造成的。

## 论文里能写和不能写的话

可以写：

> 本文提出一套面向三维医学分割集成推理的精度约束 workflow 自动优化框架，并以可恢复的
> UCT MCTS 作为主要搜索器。框架在 subject-disjoint nested validation 中搜索模型、视图、
> 融合和后处理组合，在几乎不改变 OOF Dice 的情况下显著减少 member-view forwards；在线
> A100 benchmark 验证了 2.779× refiner-stage 加速。等预算 random/greedy 消融表明主要收益
> 来自 workflow 空间中的推理冗余裁剪，而非 MCTS 对其他控制器的显著优势。

不能写：

- “MCTS 将 Dice 从 0.695 提高到 0.70”；
- “MCTS 显著优于 random 或 greedy”；
- “端到端推理加速 2.779×”；当前只量了 refiner stage；
- “已经证明 non-inferiority”；CI 跨 0 不等于通过未预注册的非劣界值；
- “test 结果”或“自动定位结果”；全部仍是 GT-centered oracle validation；
- “adaptive early exit 已验证”；当前 winner 都是 fixed。

## 答辩时把它算成一个贡献需要哪些组成

建议把它作为一个总贡献，而不是拆成多个小点：

1. 问题：3D 多模型 × 多视图推理昂贵；
2. 方法：typed grammar + 可审计 MCTS/Pareto workflow search；
3. 科研协议：subject-clean nested roles、mandatory anchor、test isolation、完整 SHA 证据链；
4. 实验：MCTS/random/greedy 等预算消融、四个精度容差档位；
5. 工程：真正按需执行、双 A100 交叉 latency/memory benchmark、逐体素 parity；
6. 边界：oracle geometry、validation-only、MCTS 不支配控制器。

多目标 MCTS 本身已有经典工作，因此论文不应声称发明了 multi-objective MCTS；可以把创新落在
医学 3D 推理 workflow 的结构化搜索、严格证据协议和可部署加速闭环。相关方法背景可参考
[Multi-objective Monte-Carlo Tree Search (PMLR 2012)](https://proceedings.mlr.press/v25/wang12b.html)。
基于预测不确定性动态分配视觉推理预算也已有相近研究方向，例如
[SA-TTS (CVPRW 2026)](https://openaccess.thecvf.com/content/CVPR2026W/Viscale/html/Yang_SA-TTS_Stress-Aware_Test-Time_Scaling_for_Vision_Models_CVPRW_2026_paper.html)，
因此后续若主打 adaptive early exit，需要在 3D 医学场景、prompt/coarse-mask 条件和
subject-level验证上做出清晰差异。

## 下一版算法建议

当前数据不支持继续调当前 outer folds 后宣称新泛化。代码层面的下一版应先冻结，再在新的
external confirmation set 验证：

1. 用 Pareto/hypervolume backup 替代单一手调 scalar reward；
2. 把 acquisition/model/view 分支做 stratified progressive widening，避免根节点大分支只被
   随机覆盖一部分；
3. 让 512 tree simulations 真正运行，但最多只新增 128 次昂贵 workflow evaluation，其余
   simulation 复用已有叶子估计；
4. 每个控制器运行多个 seed，报告 search variance，而不是各跑一个 seed；
5. 单独训练或校准 prediction-only difficulty proxy，真正验证 `3→12→24` per-case routing；
6. 冻结一个单一 deployment workflow 后，在新数据上一次性做 Dice 与端到端 latency 确认；
7. 自动定位替代 GT-centered oracle 是独立且更重要的后续问题。

## 权威证据

- 方法预注册：`MCTS_EFFICIENCY_CONTRIBUTION_PREREG_20260813.md`  
  SHA-256 `8272f77393e8523206eade1e5affcf6d6f774685ef0602ede99bc0443cd75316`
- 控制器 OOF：`workspace/BraTS/sammed3d_t1ce_wt/mcts_efficiency_contribution_20260813/FINAL_CONTROLLER_COMPARISON.json`  
  SHA-256 `757e6a82fd8be59b925010a4c692b75ad9acd6c72d0e892e46a4b7cc7cf277ff`
- 在线 benchmark：`workspace/BraTS/sammed3d_t1ce_wt/mcts_efficiency_contribution_20260813/FINAL_DEPLOYMENT_BENCHMARK.json`  
  SHA-256 `ad8cffddc4382c20af67b84674a18c707909df13cfc2cb048dd3eb5b2d471cf7`
- 冻结工程候选：`MCTS_DEPLOYMENT_CANDIDATE_20260813.json`  
  SHA-256 `89ae9e10992f9989ca99a86d96bb5b153eb91050b45096940ea2a871297a986b`
- 旧 MCTS 正式报告保持不变：`mcts_workflow_20260813_v2/final_report.json`  
  SHA-256 `b3cabb91642e8f0b711b1a9acfd2567fa305ba57850d49d3ec5247024ed22009`

本轮没有修改旧 handoff、旧 result、旧 freeze 或旧 MCTS evidence chain，没有读取或使用
270-case test，也没有启动新的模型训练。
