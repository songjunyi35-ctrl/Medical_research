# Domain-SAGE 3D 正式配对结果（2026-08-14）

## 结论

受 DPD-SAM latent-domain conditioning 启发的 Domain-SAGE 3D **没有提高分割 Dice**，
按预注册判为负消融，不补第二 seed，也不在当前108例 validation 上继续搜索 K、FiLM
rank 或 domain-loss weight。

| arm | best epoch | scan Dice | subject Dice | 相对 frozen base scan |
|---|---:|---:|---:|---:|
| matched plain SAGE | 3 | **`0.701897044`** | **`0.696969673`** | `+0.002853689` |
| Domain-SAGE 3D | 1 | `0.701764545` | `0.696724255` | `+0.002721189` |

Domain-SAGE − plain：

- scan 配对差：`-0.000132499`；
- subject 配对差：`-0.000245418`；
- subject bootstrap 95% CI：`[-0.000848635,+0.000352320]`；
- 57 scans 改善、51 scans 下降；
- 未达到预注册的 scan `+0.001` 且 subject 不下降门槛。

因此当前推荐架构仍是更简单的 **plain SAGE-3D safe adapter**。本轮 seed18 plain
也再次达到 `0.701897`；与已有 seed16/17 合并后，三个 plain seed 的 scan/subject mean
为 `0.702026554 / 0.697096458`，进一步支持 plain SAGE 的可复现性。

## 实验做了什么

该实验没有重写100M参数的 SAM-Med3D image encoder。它冻结 robust seed15 refiner，在
已有1.177M参数 SAGE adapter 上增加：

1. 仅以 T1ce 为输入、coarse通道恒为0的冻结 base bottleneck GAP descriptor；
2. 仅用972个训练病例做逐通道标准化和 K-means++，`K=4`；
3. 两层 domain MLP（128→64→4）预测每个病例的潜在域 affinity；
4. 在 adapter bottleneck 和三级 decoder 上加入 rank-4 affinity-weighted low-rank FiLM；
5. FiLM domain coefficients 全零初始化，保证 epoch 0 是严格 identity；
6. Domain arm 使用 `DiceCE + 0.1 * pseudo-domain CE`。

plain/Domain 两臂使用相同 seed `20260818`、相同 WeightedRandomSampler、增强、batch、
优化器和 early stopping。GPU0=plain、GPU1=Domain；全部972/108病例，batch 2，BF16，
AdamW LR `1e-4`、WD `1e-5`、max 20 epochs、patience 8。两臂 epoch 0 在完整108例上均
严格复现 frozen base scan/subject `0.699043355 / 0.694567241`。

## 潜在域是否真的学到了

fit-only registry 的 train cluster sizes 为 `[360,64,242,306]`，validation nearest
sizes 为 `[36,9,19,44]`，没有空域。训练描述符、归一化统计和 centroids 均不使用
validation 或 test。

Domain arm validation domain accuracy：

- best segmentation epoch 1：`0.861111`；
- early-stop epoch 9：`0.953704`。

域分类越来越准，但 segmentation 从 best `0.701765` 降到 `0.700223`。这说明负结果
不是分域分支完全没学会，而是当前域目标与 residual correction 主任务没有形成有效
协同，后期甚至表现为 auxiliary-task overfitting。

训练域对 target-volume quartile 的 NMI 为 `0.0721`；同一 subject 的纵向 scan 两两
落入相同域的比例为 `62.1%`。descriptor 确实捕获了一些稳定病例/采集特征，但当前
GT-centered、N4、positive-voxel z-score 预处理已经削弱了原始中心风格差异，剩余域
信息不足以稳定改善分割。

按 validation pseudo-domain 做的事后描述中，domain 0/1/2 的平均配对差为
`+0.000328/+0.000628/+0.000790`，domain 3 为 `-0.001064`。这是揭示机制的 post-hoc
观察，不能据此设计 validation-selected router 或只对部分域启用 FiLM；需要新确认集
才能检验该假设。

## 为什么不跑200 epochs

DPD-SAM 的200 epochs属于“50 epoch prompt warm-up + 150 epoch diffusion”完整流程，
并不适用于这个 identity-initialized小adapter配对实验。本轮：

- Domain arm best在epoch 1，连续8轮无新高后于epoch 9正常停止；
- plain best在epoch 3，连续8轮无新高后于epoch 11正常停止；
- plain epoch 11 train Dice继续上升，但validation已降至 `0.692811`；
- Domain epoch 9 domain accuracy升至 `0.953704`，但validation降至 `0.700223`。

因此继续到200 epochs不是“训练更充分”，而是在已经明确的 train/validation 分离后
继续过拟合，并会破坏预注册协议。

## 证据边界

- 数据：972 train / 108 validation，subject-clean；
- 输入：T1ce、WT=`{1,2,3}`；
- 几何：GT-centered N4 128³；
- coarse logits 来自 GT-derived 三次 corrective clicks；
- 108 validation 已被此前研究反复用于开发；
- 270-case test、test manifest 和 test evaluator 全程未访问；
- 所以这是 oracle development negative ablation，不是自动定位或独立 test。

两臂 best checkpoint 已在真实 A100 上独立完整复载；scan、subject、base 和 delta 与
checkpoint内嵌值的差全部严格为0。运行中观察到两卡计算利用率约99%，显存约
`4.3/3.8 GiB`；低显存来自小adapter，不表示GPU计算空闲。

## 证据与哈希

正式目录：`workspace/BraTS/sammed3d_t1ce_wt/domain_sage_pair_20260814/`。

| artifact | SHA-256 |
|---|---|
| registry manifest | `e0e9a3d234a30be6f6f625ed7f6d400c388805728b5960ec1643b6c89f05ed05` |
| registry NPZ | `b8b3b13041010f00ca77ec5bff99ad22a377ba2785968228b5b7ea16a48b9cd0` |
| plain `TRAIN_DONE.json` | `8bfd6f7fe22664735d3aa3a5c7a3a0206ab6b305abcf68b5eda7cd1b2a9ec76c` |
| Domain `TRAIN_DONE.json` | `cb61ad6981745689a02c3585fe10c540a4097932d49df9155c2230727ce8ccf1` |
| plain best checkpoint | `04a12ae665e87873643ac140bb20d37d31923818fd522e18fd932922322be945` |
| Domain best checkpoint | `7bf0f96571f34478515eb7bbf7be8b302c63e656165993bd62d5e9c9387a5ce8` |
| plain independent reload | `922a30edfdbbe8ce136edb7329b1ba22824f0ba33ec4cfb0157742f222818980` |
| Domain independent reload | `b78dec092fa1c58c7c9b5c11a422c5a73b7f89e1496897884dcd69b668f4d499` |
| final comparison JSON | `ffaa1d129a613e782ffe1cbf64f52dcc28ae02173b1f04b3d6ba21cf13321977` |

方法预注册：`DOMAIN_SAGE_PREREG_20260814.md`，SHA-256
`c7b65b60502581f014be2932648e55fdd3049c3b6b58d5486f4f1cb701cee279`。

训练代码与预注册哈希一致；复载脚本 SHA-256 为
`22cbddeb55f9ff68503e2012c1cde5dda0b4f769482141a48d445b65ff183b86`，统计脚本为
`4c7963f517f0cd3e68465f4965ac791b7a78ef37e7c1a770617d44a32a9027e7`。

## 下一步

1. 冻结 plain SAGE-3D，不根据 Domain-SAGE 某些 subgroup 的 post-hoc 点估计改路由；
2. 不继续在当前 oracle crop 上做 style-domain conditioning；
3. 把架构创新转向 **自动定位/自动 prompt**，因为 DPD 的域提示原本就是为 whole-volume
   自动定位与跨中心差异设计，而当前 GT crop 已提前解决了最难的定位问题；
4. 若将来获得新的未暴露确认集，可以一次性检验 frozen plain SAGE 和一个预先冻结的
   domain-3 suppression 假设，但不能先在当前108例继续调。
