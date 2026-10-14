# AFlow BraTS continuation handoff — 2026-08-19

本文件承接`AFLOW_BRATS_HANDOFF_20260813_SINGLE_REPO.md`第26节，是当前最新科研入口。

## 27. true-160³ 全756例epoch200与test270：负结果，停止该方向

新线程已确认本地运行配置为`gpt-5.6-sol`、`model_context_window=1000000`、
`model_auto_compact_token_limit=900000`。为释放构建空间，经用户明确授权删除了可由
`scripts/cache_t1ce_wt_robust_inference_atoms.py`重建的派生
`workspace/BraTS/cache/t1ce_wt_robust_mcts_atoms_f32`，精确释放`21,743,293,543` bytes；
原checkpoint、源cache与重建入口均保留。

最初按第26节冻结了新seed`20260828`的648 fit / 108 inner-tune subject-clean split，并实现了
独立variable-resolution builder/dataset/trainer/analyzer。true-128³/160³ batch8 smoke均通过，
训练峰值allocated分别约`21.51 GB`和`42.01 GB`，两臂sampler权重完全一致，reload差均为0。
随后用户把目标从“严格分辨率归因”改成“优先提高benchmark”，明确不要求paired对齐，并质疑
从756例中扣除108例。正在运行的128³ paired control因此在最后完整epoch63后中止；没有保存
checkpoint，也从未迭代inner-tune tensor，该输出已标成`interrupted_not_a_result`。

替代协议使用全部756 train cases / 346 subjects、seed`20260829`、true-160³ raw-NIfTI cache、
batch8和完整200-epoch schedule，从头训练4,700,897参数plain SegResNet。cache严格从原始NIfTI
执行canonical RAS -> N4 shrink-2 -> image-nonzero crop -> source nonzero z-score ->
trilinear image / nearest-exact label resize到160³，没有使用128 cache作为输入；4个均匀槽位的
图像和标签逐字节复现。200轮共18,800 optimizer steps、耗时`8092.49s`，峰值allocated /
reserved为`39.16 / 43.72 GiB`。训练诊断如下（不是held-out性能）：

| epoch | LR | train loss | augmented train Dice |
|---:|---:|---:|---:|
| 100 | `2e-4` | `0.305017` | `0.699798` |
| 168 | `5.491e-5` | `0.244158` | `0.759964` |
| 200 | `1e-5` | `0.223999` | `0.780221` |

epoch168/200 checkpoint SHA-256分别为
`3d2d7f7d622a94ca3f0d53a31341a650fcfb22ed13f35951a7f40790ca52f9f4`和
`a25e8519b3f9a66ee5c389ad94f0e62ac548dfd6339468637c8f2c3536139b67`；epoch200新鲜reload
最大logit差为0。

用户随后明确授权只用epoch200访问复用internal test270。test也从270个原始NIfTI重新构建
true-160³ cache，禁止将历史128³ test cache二次插值；270 scans / 119 subjects全部完成，
4/4 raw复现逐字节通过。固定raw logit`>0`、无TTA/后处理/SAM的结果为：

| reused internal test270 | true-128³ epoch168 | true-160³ epoch200 | 160-128差 | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | **`0.741203949`** | `0.737060297` | `-0.004143652` | `[-0.010491,+0.001766]` |
| subject Dice | **`0.723179486`** | `0.719728630` | `-0.003450856` | `[-0.012075,+0.004820]` |
| point hit | **`0.929630`** | `0.874074` | `-0.055556` | — |

270 scans中150提高、119下降、1持平；119 subjects中65提高、54下降。case/subject中位差为正，
但均值为负，说明若干较大退化压过了许多小收益。point transition为228 hit→hit、23 hit→miss、
8 miss→hit、11 miss→miss。Dice跨分辨率比较包含nearest-exact标签在128/160网格上的栅格化差异，
但明显的point-hit退化不能由该因素单独解释。

因此对当前benchmark目标，应把“将原plain locator直接从128³升到160³并训练到epoch200”判为
负结果：计算体素约`1.953x`、显存约翻倍，却没有刷新test点估计。停止继续调160尺度、阈值、
checkpoint或后处理；当前历史最佳仍是true-128³ epoch168。完整训练归档见
`archive/research_records/2026-08-18_plain_resolution_160/RESULT.md`，test归档见同目录
`TEST270_RESULT.md`；absolute test report SHA-256为
`973e4cc452e4ea7b9de3a190d5ad293724f2cd94e0a82f861f521a2e15477f35`，配对报告SHA-256为
`23e047a0d1e9b9351b4cbd1dc4916c1a578752c493e1c42243e14f0d026fbf00`。outer selector与
confirmation在本阶段均未再次访问；test已按用户明确授权访问。

## 28. 下一条科研：true-128³ late-cosine EMA

下一条方向冻结为回到true-128³，在不增加推理参数、显存或结构的前提下做late-cosine EMA
权重平均。直接机制假设是：epoch168附近已进入低LR高质量盆地，EMA可降低随机采样/增强导致的
权重抖动与少数大幅退化。新增隔离EMA trainer，不修改任何历史入口；使用全756例训练，固定保存
raw/EMA epoch168与epoch200。先在历史固定validation上比较raw与EMA，不用test挑EMA decay、
起始epoch、checkpoint或阈值；第27节test270结果不得回流调参。

建议冻结的唯一EMA定义：epoch100完成后复制raw权重作为EMA初值，从epoch101每个optimizer step
执行`ema = 0.999 * ema + 0.001 * raw`；floating buffers同样EMA，integer buffers直接复制。
模型、128³ cache、seed、sampling、augmentation、loss、optimizer、batch8、200-epoch schedule与
raw control保持一致。primary为固定epoch200 EMA对固定epoch200 raw；epoch168两者仅作为预先保存的
secondary checkpoint，不根据validation选择EMA decay或起点。
