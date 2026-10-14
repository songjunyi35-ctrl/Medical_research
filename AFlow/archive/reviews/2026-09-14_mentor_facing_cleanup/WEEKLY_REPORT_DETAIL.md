# 清理前技术资料：AFLOW_BRATS_WEEKLY_REPORT.md

> 2026-09-14整理前的完整技术快照。所有“当前最好”“运行中”“下一步”均属当时状态，不可据此恢复实验；当前阅读入口请看[导师汇报](../../../AFLOW_BRATS_WEEKLY_REPORT.md)和[日期流水账](../../../AFLOW_BRATS_RESULTS.md)。原字节另有压缩备份；这里仅调整相对链接以便归档后阅读。

# BraTS T1ce/WT 周报

报告日期：2026-08-17

## 2026-08-17 最终补充：direct plain locator刷新项目记录

按新的756 train / 108 selector / 108 untouched internal confirmation subject-clean角色
从头训练4,700,897参数plain locator，selector预先选中epoch99。固定raw logit `>0`、
无CAPG、SAM、TTA或后处理，selector/confirmation分别达到scan
`0.720419/0.716505`。用户明确授权并在test前冻结协议后，同一270-case internal test
达到scan-macro **`0.725820`**、subject-balanced **`0.709413`**、automatic point
hit `0.900000`。

这是项目内最高自动和最高数值test记录。最强直接机制证据是旧development上locator
direct mask `0.716125`，接旧冻结SAM point+dense后反降到`0.599324`；新系统保留
任务对齐的dense locator输出，避开已测得的decoder/prompt分布失配。旧受控brain-crop
几何增益为`+0.054250`，也为当前输入表征提供支持；其他架构、sampling和augmentation
因素没有逐项消融。

270 test早已被其他arm揭示，因此这不是pristine blind test、外部确认或公开BraTS SOTA。
完整结果和证据分级见
[`archive/research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md`](../../research_documents/2026-08-17_root_cleanup/PLAIN_LOCATOR_TEST270_RESULT_20260817.md)，
归档见
[`archive/research_records/2026-08-17_plain_locator_test270/`](../../research_records/2026-08-17_plain_locator_test270)。

## 2026-08-13 最终补充：robust residual refiner

在周报较早结论之后，已完成 frozen coarse 失配诊断与 robust-shift 实验。train
frozen coarse Dice 约 `0.829776`，validation 仅 `0.636588`。有效改动是在训练时
只随机平移 coarse logit（每轴最多 4 voxel），图像和标签不移动；最终方法不含
灰度 loss。

- 三 seed raw validation：`0.695636`、`0.696351`、`0.699043`；
- 三 seed × 8 reflection TTA raw：`0.698235`；
- validation 冻结 threshold `-1` + 去除小于 500 voxel 连通域：
  scan-macro **`0.7004189626`**，subject-balanced `0.6953719491`；
- 相对 epoch58 A100 replay `0.669548`，提升 `+0.030871`，96/108 例改善；
- gray loss 配对效应约 `-0.00096/-0.00175/+0.00155`，不稳定，未进入最终方案。

`0.700419` 是 GT-centered、validation-selected 开发结果，不是 test 或自动分割。
旧 epoch58 的 oracle test `0.703798` 是另一套 checkpoint/协议，不能当作 robust
三 seed 的 test。新的唯一权威交接为
[`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`](../../../AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md)。

## 本周结果

vanilla SAM-Med3D 基线阶段已基本完成。实验使用单通道 T1ce、WT `{1,2,3}`、
subject-clean 的 972/108/270 划分、官方 1.5 mm 预处理、严格 upstream 11-pass
trainer、完全解冻和五次循环 mask 点击。选中的 epoch 21 得到：

- 验证集五次点击取最佳 Dice `0.571306`，第 5 次点击 Dice `0.559185`；
- 固定测试集五次点击取最佳 Dice **`0.594226`**，第 5 次点击 Dice
  **`0.585754`**；
- subject-balanced 测试集五次点击取最佳 Dice `0.580970`。

师兄/MSCL 的 vanilla 结果约为 0.601，因此这可以视为本地划分和 prompt 协议下
合理、可审计的基础复现。该实验没有灰度/强度阈值损失、课程学习、BraTS 2022
或模拟病灶。

## 成绩总览

下表按 Dice 数值从低到高排列，但不是一个严格同口径 leaderboard。必须同时看
split、定位方式、预处理和点击协议。

| 结果性质 | Split / 指标 | 定位与协议 | Dice |
|---|---|---|---:|
| official15 vanilla baseline | 270 test，scan-macro best-of-5 | GT 中心；1.5 mm；无 N4；upstream 五点击 | **0.594226** |
| 师兄/MSCL vanilla 参考值 | 论文 BraTS 2024；精确划分/prompt 未披露 | 外部参考，不是本项目 run | **0.601** |
| 自动 brain-crop 冻结结果 | 270 test，scan-macro C3 | 图像非零脑区自动裁剪；N4；旧三点击 | **0.625359** |
| GT-centered residual refiner | 108 validation，scan-macro C3 | oracle GT crop；N4；旧三点击 | **0.668399** |
| GT-centered residual refiner | 270 test，scan-macro C3 | oracle GT crop；N4；旧三点击 | **0.703798** |
| **direct plain locator** | 270 test，scan-macro fixed logit>0 | T1ce-only；无GT定位/点击 | **0.725820** |

所以，当前最高的**自动定位 test** 仍是 `0.625359`；当前最高的数值结果是
oracle GT-centered refiner test `0.703798`。后者不能写成自动端到端成绩，也不能
替换 official15 vanilla baseline。

## 预处理进展

较早的自动预处理分支先裁剪图像非零脑区 bounding box，再 resize 到 128^3。
在与它匹配的旧三点击/N4 协议下，固定测试 Dice 从 whole-head 对照的
`0.571109` 提升到 **`0.625359`**，提升 **`+0.054250`**；270 例中
232 例改善，配对 95% CI 为 `[+0.0455,+0.0629]`。

这说明减少空背景 token 浪费很可能有效，可以把它报告为该旧协议内已经完成的
预处理改进。但目前不能把它解释成相对最新官方基线的单因素提升，因为两套实验还有
多项差异。因此下一项干净实验应当是“官方流程匹配”的验证集对照：只把空间几何
替换为图像非零 brain-crop + resize，其他条件全部固定。

## GT-centered residual refiner：validation 与 test

正式记录中最好的开发集指标是**验证 Dice `0.668399`**，不是 0.68。它来自
全分辨率 residual SegResNet refiner，输入为 T1ce 和固定 SAM 第 3 次点击的
raw logit，并工作在 oracle GT 中心的 stage-2 crop 上。epoch 58 将同一 H100
coarse 基线从 `0.635949` 提高到 `0.668399`，93/108 例改善。0.6800 是
预注册成功门槛，最终没有达到。

2026-08-13 经用户明确授权，在写入一次性 freeze 后，epoch-58 checkpoint 首次
在同一 270 例固定 test 上完成评测。两张 A100 同时生成 frozen SAM coarse 和
refined 结果：

- scan-macro：`0.666658 -> 0.703798`，配对提升 **`+0.037140`**；
- subject-balanced：`0.648864 -> 0.686637`，提升 **`+0.037773`**；
- 244/270 例改善，26/270 例下降；
- WT crop 体积 `<8000` voxels 的 10 例总体没有改善（`-0.000469`），其余三个
  体积分层分别提升 `+0.037582`、`+0.042922` 和 `+0.036501`。

test refined Dice 高于 validation，主要因为同次 coarse test 本身也更高
（`0.666658` 对 `0.635949`）；真正可归因给 refiner 的配对增益是 test
`+0.037140`，与 validation `+0.032450` 接近。该结果仍是 oracle GT-centered、
N4、旧三点击协议，不是 official15 或自动分割成绩。

后续 prompt-local 64^3 refiner 只把 `0.669548` 改善到 `0.669738`
（`+0.000190`），因此该分支记为正式负结果，不应原样继续投入。

## 当前结论

- 基础复现已经足以作为下一阶段的锚点。
- brain-crop 是最有希望的自动预处理方向，但需要官方协议下的单因素验证实验。
- residual refiner 在 validation 和固定 test 上都给出约 `+0.03` 至 `+0.04`
  的配对增益，但依赖 oracle 定位；它是机制证据，不是自动主线终点。
- 270 例测试集已经揭示，禁止用于任何新方法或超参数选择。

## 下周计划

1. 只为一个实验写预注册：灰度/强度损失，或官方流程匹配的 brain-crop 几何。
2. 如果复现灰度损失，先确定 soft/hard prediction、系数、强度域、空 mask 行为、
   reduction 和应用的交互步。
3. 运行预处理一致性/单元测试和短时双卡 GPU smoke。
4. 只在 972/108 上训练与选择，并报告 scan-macro、subject-balanced 和病灶
   体积分层验证指标。
5. 在两个单因素结果明确前，不要把灰度损失与 brain-crop 混合。

精确路径、哈希和安全边界见 [`archive/research_documents/2026-08-17_root_cleanup/T1CE_MAINLINE.md`](../../research_documents/2026-08-17_root_cleanup/T1CE_MAINLINE.md)，精简数字
账本见 [`AFLOW_BRATS_RESULTS.md`](../../../AFLOW_BRATS_RESULTS.md)。

本次 refiner 最终绑定为
`workspace/BraTS/sammed3d_t1ce_wt/FINAL_TEST_RESULT_RESIDUAL_REFINER_20260813.json`
（SHA-256 `149fc9a89d5877de9efa6e52f1a8a459d7c918eb2aabf06a3872846bd20fdcdd`）；
逐病例报告 SHA-256 为
`f6ef9f1965f29502e1d190747f0363b89ecb42828ac58213dcd3e0fb674f8f44`。

## 2026-08-14 补充：MCTS 推理 workflow 效率优化

本周把 robust refiner 的固定 3 checkpoint x 8 reflection TTA 从人工固定流程，扩展成
了可搜索、可恢复、可审计的推理 workflow。真正的 UCT MCTS 包含 N/W/Q、selection、
rollout、progressive widening、祖先回传和断点恢复；workflow grammar 同时表达模型/
视图选择、fixed 或 adaptive 推理、融合、uncertainty threshold 与 26-connectivity
后处理。搜索采用 subject-disjoint nested 三折，只能使用 108 例 validation atom，
270 例 test 在搜索入口不可达。

### 核心结果

| 指标 | MCTS | locked reference |
|---|---:|---:|
| OOF scan Dice | `0.699821` | `0.700422` |
| OOF subject Dice | `0.695037` | `0.695373` |
| subject 配对差 | `-0.000336` | — |
| 配对 95% CI | `[-0.001989,+0.001365]` | — |
| 平均 member-view forwards | `7` | `24` |

这说明本轮 MCTS 没有带来 Dice 指标收益，但以约 `70.8%` 的平均 forward 降低换取了
很小的精度变化。原因是 24 个 member-view prediction 高度冗余；不同视图/成员的二值
mask 平均两两 Dice 约 `0.946–0.949`。三个 fold winner 均为 fixed workflow，已实现的
adaptive 3→12→24 没有胜出。

### 必须同时报告的控制实验

相同三折、相同每 fold 128 unique workflow 预算下，MCTS/random/greedy 的 OOF subject
Dice 分别为 `0.695037/0.695746/0.695016`，平均 forwards 为 `7.0/8.0/6.667`。
MCTS 对 random 的差为 `-0.000709`，对 greedy 的差为 `+0.000021`，CI 均跨 0。
因此不能宣称 MCTS 优于简单搜索器；硕士论文中应把贡献表述为“精度约束、可审计的
三维医学分割推理 workflow 自动优化框架和效率验证”，而非 Dice SOTA 或 MCTS
优越性。

### 真实部署收益

冻结的工程候选为单 checkpoint x 8 views。双 A100 交叉换卡在线 benchmark 中，
refiner-stage 平均延迟从 `1.0757` 降至 `0.3871 s/case`，加速 `2.779x`；模型加载后
显存下降 `66.7%`，peak allocated 下降 `18.1%`。在线 candidate/locked 各 8 例与
exact atom cache 的输出 mask 完全一致。此计时包含 refiner 的传输、forward、融合和
后处理，但不含上游 SAM coarse 与预处理，不能写成端到端加速。

完整贡献文档为 `archive/research_documents/2026-08-17_root_cleanup/MCTS_EFFICIENCY_CONTRIBUTION_RESULT_20260813.md`；controller 对照
JSON SHA-256 为 `757e6a82fd8be59b925010a4c692b75ad9acd6c72d0e892e46a4b7cc7cf277ff`，
在线 benchmark JSON SHA-256 为
`ad8cffddc4382c20af67b84674a18c707909df13cfc2cb048dd3eb5b2d471cf7`。
新增实现和验证完成后，全仓测试为 `223 passed`。

## 2026-08-14 架构创新补充

本周新增两轮预注册架构实验。第一轮 ELPGR-A2 在 residual refiner 中加入 coarse
uncertainty、三维边界提示和显式 correction gate，但两 seed mean 为 `0.695917`，低于
matched robust control `0.697697`，正式判负。机制分析显示 gate 虽学到 error map，
却把平均 residual scale 压到 `0.18–0.27`，破坏已有修正。

第二轮改用 SAGE-3D：冻结 `0.699043` robust base，外挂零初始化、约 1.18M 可训练参数的
uncertainty adapter。plain/gated 分别达到 `0.701991/0.702090`，subject Dice 为
`0.697102/0.697140`；相对 base 的 subject 配对 CI 下界均大于 0。adapter 达到预注册
成功线，但 gated 相对 plain 仅 `+0.000099`，所以正收益归因于 safe adapter，不归因于
gate。结果仍是 GT-centered oracle development validation，未访问 270 test。

外部会话使正式训练停在 epoch 10、未生成 `TRAIN_DONE.json`；epoch-3 best checkpoint
均已完整 reload 且指标严格一致。详细限制、哈希和下一步见
`archive/research_documents/2026-08-17_root_cleanup/SAFE_ADAPTER_PAIR_RESULT_20260814.md`。独立 reload evaluator 和本轮架构单测加入后，
全仓回归为 `236 passed, 24 warnings`。

### Seed17 全量复现与 boundary 消融

随后使用两张 A100 完成 full 972/108 配对：GPU0 复现 plain safe adapter，GPU1 只增加
Dice-only 三维 boundary auxiliary。seed17 control/plain 达到 scan/subject
`0.702191/0.697218`；结合 seed16，plain 两 seed mean 为 `0.702091/0.697160`，确认
safe adapter 的正收益方向可复现。

boundary arm 为 `0.702632/0.697579`，相对 matched control 只高
`+0.000440/+0.000362`，subject CI `[-0.000626,+0.001339]`，没有达到预注册的
`+0.001`。boundary head 虽学到 Dice `0.343`，但不能替代主指标失败；不继续在当前 108
validation 搜 boundary 权重。两个正式任务和独立 reload 均正常完成，全仓测试更新为
`240 passed, 24 warnings`。完整证据见 `archive/research_documents/2026-08-17_root_cleanup/BOUNDARY_SAFE_ADAPTER_RESULT_20260814.md`。

## 2026-08-14 追加：Adaptive router 与固定推理剪枝

在 MCTS 搜索发现 24-way ensemble/TTA 冗余后，本周继续实现并预注册了 prediction-only
的 1→8→24 per-case router。strict-margin 三折 OOF subject Dice 为 `0.695260`，locked
为 `0.695373`，平均 forwards `2.556`。但是路由 gain correlation 只有
`0.0806/-0.2298`；等 stop-count 的 10,000 次随机路由中有 `13.74%` 达到或超过学习
路由。因此 adaptive routing 按预注册记为负结果，不能作为有效 early-exit 贡献。

真正有用的结果是固定推理剪枝：

| forwards | scan Dice | subject Dice | 相对 locked subject |
|---:|---:|---:|---:|
| 1 | `0.700106` | `0.695016` | `-0.000358` |
| 4 | `0.701150` | `0.696179` | `+0.000806` |
| 8 | **`0.701571`** | **`0.696271`** | `+0.000897` |
| 24 locked | `0.700422` | `0.695373` | `0` |

4/8-forward 的 Dice 点估计高于 locked，但 CI 跨 0，只能报告为开发集点估计。1-forward
在 Dice 点变化很小的同时，双 A100 交叉 benchmark 将 refiner-stage latency 从
`1.075861` 降到 `0.078786 s/case`，实测 `13.6555x` 加速；模型加载显存下降 `66.67%`，
peak allocated 下降 `26.92%`。candidate/locked online 与 exact cache parity 全通过。

完整方法、负结果、Pareto 曲线、论文口径和证据哈希见
`archive/research_documents/2026-08-17_root_cleanup/INFERENCE_PRUNING_RESULT_20260814.md`。本轮结束后全仓测试为
`228 passed, 26 warnings`。下一步冻结 fixed1/fixed4/fixed8/locked24，在新外部确认集
一次性确认，禁止继续在当前 108 validation 上选择组合。

## 2026-08-14 补充：DPD启发的Domain-SAGE配对实验

本周进一步参考DPD-SAM的latent-domain conditioning，但没有修改冻结的SAM-Med3D主干，
而是在plain SAGE adapter上加入fit-only K=4 domain registry、两层domain MLP和四级
zero-init low-rank FiLM。GPU0/1以相同seed `20260818`并行训练完整972/108。

plain/Domain scan Dice分别为`0.701897/0.701765`，subject Dice分别为
`0.696970/0.696724`。Domain−plain scan/subject差为`-0.000132/-0.000245`，subject CI
`[-0.000849,+0.000352]`，未过预注册`+0.001`。虽然domain accuracy最终达到`0.9537`，
分割best仍在epoch1并低于plain，说明域辅助任务可学习但没有转化为Dice收益。本分支
正式判负，不继续在当前validation调参。

plain seed18同时构成第三次SAGE复现；seed16/17/18 scan mean为`0.702027`。两臂均正常
early-stop并完成独立reload。完整结论和哈希见`archive/research_documents/2026-08-17_root_cleanup/DOMAIN_SAGE_RESULT_20260814.md`；仍为
GT-centered oracle development，270 test未访问。新增实现后全仓测试为
`245 passed, 26 warnings`。

## 2026-08-14 补充：全自动locator与CAPG-3D

旧brain-crop实验仍使用GT前景首点，本周补齐了真正无GT的自动链路。使用图像非零bbox、
N4和128³ resize训练4.70M参数的3D locator；重建cache四个数据SHA与旧cache完全一致。
plain/CAPG均完整跑满100 epoch，270 test未访问。

plain locator达到scan/subject `0.716125/0.706215`；带中心热图辅助头的CAPG为
`0.719090/0.710459`，差`+0.002965/+0.004243`，subject CI
`[-0.003736,+0.013624]`，不足以证明架构收益。更重要的是CAPG中心点hit仅`0.638889`，
plain分割argmax为`0.935185`。

冻结SAM的预注册point+dense主指标反而是plain `0.599324`、CAPG `0.597910`，差
`-0.001414`，subject CI `[-0.004178,+0.002713]`。所以CAPG正式判负，旧SAM也不是
强locator的有效postprocessor。可保留的新进展是完全自动plain locator约`0.7161`；
需在新外部确认集验证后才能写成泛化结果。完整证据见`archive/research_documents/2026-08-17_root_cleanup/CAPG3D_RESULT_20260814.md`。
本轮结束后全仓回归为`252 passed, 26 warnings`。

## 2026-08-17 补充：灰度/强度 residual 没有晋级

在新的 subject-clean 648/108 inner split 上完成 plain、参数匹配 zero-intensity control
和 intensity-aware residual 三臂。intensity 相对 control 的 scan/subject 差为
`-0.002638/-0.003331`，point hit 从 `0.888889` 降到 `0.518519`；预注册门槛
失败 3/5。所有 checkpoint 重载差为 0，outer selector、confirmation、270 test 均未访问。

本结果否定的是当前“归一化 T1ce + 局部对比 + 梯度”不确定区 residual，不代表所有强度
建模无效；但禁止在当前 inner split 继续搜灰度阈值。benchmark 仍由 direct plain locator
保持。完整证据见 `archive/research_documents/2026-08-17_root_cleanup/INTENSITY_RESIDUAL_RESULT_20260817.md`。
本轮结束后全仓回归为 `259 passed, 24 warnings`。

## 2026-08-18 补充：100 epoch 截断与 late-cosine 正结果

没有继续灰度分支，也没有访问108 confirmation或270 test。本轮只从昨天648 fit cases
中冻结新的540 fit / 108 fresh-tune subject-clean inner split，检验plain locator前100轮
固定`2e-4`后追加cosine衰减。inner pre100/post100最佳分别在epoch95/168，scan为
`0.723604/0.746335`，subject为`0.719629/0.744497`；subject配对差95% CI
`[+0.014792,+0.038213]`，五项门槛全通过，两checkpoint完整重载差为0。

随后锁定epoch168，在原756 train cases上从头训练且完全不读取validation。唯一一次
outer selector评估得到scan/subject/point hit `0.731008/0.714314/0.925926`，相对冻结的
epoch99 baseline为`+0.010589/+0.009697/-0.009259`，全部outer主门槛通过。108 scans中
80提高，45 subjects中33提高，说明点估计不是由单个异常病例驱动；但outer描述性
subject CI `[-0.001312,+0.020934]`跨0，只能称development指标突破。

新的development checkpoint SHA-256为
`f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`；它尚无test结果，
不得再进入已揭示confirmation/270 test。完整记录见
`archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`，全仓回归更新为
`270 passed, 24 warnings`。

## 2026-08-18 追加：epoch168 一次性 test270 历史新高

在inner和outer流程完成、checkpoint与raw-logit threshold完全固定后，用户明确授权一次
最终270测评。没有提供TTA、后处理、SAM或阈值搜索入口，test结果不回流模型选择。
checkpoint SHA-256为`f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`。

epoch168 scan/subject/point hit为`0.741204/0.723179/0.929630`，相对epoch99为
`+0.015384/+0.013767/+0.029630`。scan配对95% CI`[+0.009186,+0.021574]`，subject
配对95% CI`[+0.004123,+0.022935]`，均为正；205/270 scans与87/119 subjects提高。

因此这是当前项目复用270 internal test上的新历史最高点估计。由于该集合此前已揭示，
它不是pristine blind test、外部确认或公开SOTA，但属于固定方法的一次合法最终测评。
完整记录见`archive/research_records/2026-08-18_plain_long_schedule/RESULT.md`，最终JSON为
`workspace/BraTS/sammed3d_t1ce_wt/plain_long_schedule_20260818/final756_seed20260819_epoch168/FINAL_TEST270_EPOCH168.json`。
全仓回归更新为`273 passed, 24 warnings`。
