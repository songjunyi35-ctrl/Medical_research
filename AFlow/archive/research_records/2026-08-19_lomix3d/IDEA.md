# Experiment idea: 三尺度 LoMix-3D 训练监督

Status: preregistered before implementation, smoke, training, or validation

## 1. 现在的问题是什么？

当前 true-128³ 单模型在 fixed validation108 上已经接近平台：历史 epoch168 的 Scan/Subject Dice 是 `0.731601219/0.722120565`，完全匹配的监督 continuation epoch200 是 `0.733605660/0.723671900`，当前最佳 voxel-KD epoch200 是 `0.734384308/0.724434143`。知识蒸馏相对同训练量 control 只增加 `0.000779/0.000762`。另一方面，把整幅图从 128³ 升到 160³ 的体素计算约为 `1.95×`，reused internal test270 Scan Dice 反而从历史单模 `0.741204` 降到 `0.737060`。

这说明下一步不能只靠增加整图像素；需要在不改 128³ 输入和 1× 部署成本的条件下，让同一网络更充分利用自己已有的粗尺度全局信息和细尺度边界信息。

## 2. 一句话直觉

像让同一个人训练时同时看地图全貌和街道细节，并练习把两种视角合起来；真正部署时仍只交原来那一份最终答案。

## 3. 为什么选择这个idea？

- 当前误差现象：160³ 负结果说明“整图加密”既贵又不稳；voxel-KD 相对 matched control 的增量很小，说明普通最终层监督和 teacher 模仿都接近平台。
- LoMix 针对的部分：SegResNet 的 decoder 本来就有 32³、64³、128³ 三个尺度，但现有训练只监督最终 128³ logit。LoMix 在训练期让粗、细尺度的原始预测及其组合共同提供梯度，不增加输入体素。
- 一手证据：NeurIPS 2025 论文《LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation》及官方仓库。论文三维附录的 MedNeXt-M_K3 初步结果为 LoMix Dice `87.19`，高于末层监督 `86.22`、普通深监督 `86.63` 和固定 mutation `86.84`；但官方公开实现仍只有二维代码。
- 既有负结果没有否定它：true-160³ 改的是输入网格；cross-seed parameter soup 混合的是不同模型参数；logit ensemble 混合的是两个独立模型的推理输出。本实验只改变同一 128³ 模型的训练期多尺度监督，部署仍是原 final head。
- 实现审计边界：官方 `concat` 在 forward 内临时创建卷积，未注册进 optimizer；官方 attention 按 subset 大小共享，与论文每个 subset 独立参数的写法不同。三维适配将预先注册每个 subset 的 `Conv3d`/attention 模块。论文未归一化 Softplus 目标对 alpha 的损失梯度恒为非负，可能整体压低权重，因此将记录全部 alpha/有效权重并设置塌缩停止门。

Primary sources:

- Paper: <https://papers.nips.cc/paper_files/paper/2025/file/5f0e54e34ba4c30fbc6f229986dbffb4-Paper-Conference.pdf>
- OpenReview: <https://openreview.net/forum?id=87c2JwNJa0>
- Official code: <https://github.com/SLDGroup/LoMix>

## 4. 具体怎么做？

输入仍是 image-only brain crop 后的单通道 128³ T1ce。原 4.70M 参数 SegResNet 只做一次原路径编码、解码；三个真实 decoder stage 分别产生 32³、64³、128³ feature。32³/64³ 各用一个新增、预先注册的 `1×1×1` 辅助卷积得到 WT raw logit；128³ 必须复用原 `network.conv_final`（Norm→Act→原卷积），不能另建最终辅助头。粗尺度用显式 final spatial size、trilinear、`align_corners=False` 上采样到 128³。

对三张原始 raw-logit map 的每个非平凡 subset（固定顺序 `01,02,12,012`），固定按 `add,multiply,concat,awf` 顺序产生四种训练期 mutant：逐体素加法、逐体素乘法、channel concat 后注册的 `1×1×1 Conv3d`、以及为该 subset 独立注册的逐体素 attention-weighted fusion。attention 对大小为 `k` 的 subset 把拼接 raw logits `[B,k,D,H,W]` 输入 `Conv3d(k,k,1,bias=True)`，在 stage 维 `softmax(dim=1)`，再对原 k 张 logit 加权求和。这里冻结采用官方运行代码的 raw-logit 语义，而不是论文 3.2 文字中的 probability 语义；不会看 validation 后切换。

每张原始或 mutant map 单独计算与当前 control 相同的 `Dice + 0.1 × CE` segmentation loss。每个 loss 乘 `softplus(alpha)`，alpha 从 0 初始化并与网络一起用同一个 AdamW 更新：

`L_total = sum_u softplus(alpha_u) * DiceCE(z_u, y)`；19项直接求和，不除以19，也不除以权重和。初始总有效权重为 `19×softplus(0)=13.169796`。

teacher：无。LoMix 不是知识蒸馏。验证、test（仅门槛通过后）和部署只使用原 SegResNet 最终 128³ raw logit；辅助 head、fusion module 和 alpha 全部剥离。阈值仍为 `raw logit > 0`，point 仍取 raw-logit argmax。

## 5. 唯一变化与固定项

- 唯一变化：训练期增加上述三尺度、四算子、19-map LoMix 辅助监督。
- 固定初始化：historical true-128³ epoch168 checkpoint，SHA-256 `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`。
- 固定 matched control：已归档 supervised continuation，global epoch169–200；同 epoch control checkpoint 作为逐 epoch 对照。
- 固定训练数据：756 scans / 346 subjects，manifest SHA-256 `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16`；true-128 cache index SHA-256 `d26b823a4715d9f3676d75cddabcf630180d48807352c23042be7faf240b622d`。
- 固定训练协议：seed `20260846`、batch `8`、weighted replacement sampler、756 samples/epoch、shift `8`、bfloat16 autocast、global epoch169–200 late-cosine schedule、checkpoint epochs `176/184/192/200`。plain模型、2个aux、4个concat、4个attention和19个alpha全部进入同一个AdamW参数组，使用相同LR与weight decay `1e-5`，gradient clip `5`覆盖全部训练参数。
- 固定初始化/顺序：先在seed20260846下加载plain epoch168，再按32/64 aux、subsets `01,02,12,012`的concat和attention、19个map names的顺序构造；Conv3d使用当前PyTorch默认初始化，alpha全0。构造完成后重置CPU/CUDA RNG到seed20260846，使主干dropout与matched control可比。
- 固定主模型：MONAI SegResNet，init filters `16`、dropout `0.1`、`blocks_down=(1,2,2,4)`、`blocks_up=(1,1,1)`；部署参数仍为 `4,700,897`。
- 固定 validation：108 scans / 50 subjects，manifest SHA-256 `f85f7df19908c7ab5bc7aafc2ce16fab1ff9786dbf3ea4ee72b97721888d35fb`；cache index SHA-256 `9339e4575561447dd5bd99bc29e909ae1e9f566721d351a244d087a40622f322`（抄写纠正见 `AMENDMENT_001_PRE_IMPLEMENTATION.json`）。
- 固定推理：raw-logit threshold `0`，无 SAM、无测试时增强、无后处理，1× locator forward。
- 固定 access：训练及选择前 `outer_selector_access=false`、`confirmation_access=false`、`test_access=false`。reused test270 不参与任何实现、loss、epoch、权重或 checkpoint 选择。

## 6. 成功、失败和停止规则

- Primary 指标：fixed validation108 Scan Dice；Subject Dice 为共同 promotion 指标。
- checkpoint selection：在 global epochs `176/184/192/200` 中最大 LoMix Scan Dice，Subject Dice 只用于完全相同 Scan 值时打破平局。
- Promotion gate（全部必须通过）：
  1. selected LoMix Scan 和 Subject Dice 都严格高于同 global epoch matched control；
  2. selected LoMix Scan 和 Subject Dice 都严格高于当前 validation 最佳单模 voxel-KD epoch200 (`0.734384308/0.724434143`)；
  3. Point hit 不低于 voxel-KD epoch200 的 `0.935185185 - 1/108`；
  4. 相对 voxel-KD epoch200 的 paired Scan 和 Subject bootstrap 95% CI 下界都不低于 `-0.003`；
  5. formal final checkpoint reload max absolute difference 为 `0`，且剥离辅助模块前后的 eval final logits 完全一致；
  6. 所有预注册 implementation/smoke gates 通过。
- 允许进入 test：仅 `all_gates_passed=true` 后冻结唯一 selected checkpoint、训练/验证报告哈希和新 test evaluator 哈希，写 pre-access receipt，再对 reused internal test270 测评一次。
- Failure/stop rule：任一 gate 未过即不进 test，不搜索 operation、alpha 规则、loss 权重、epoch、阈值或后处理。
- 立即技术停止：NaN/Inf；任一注册的 aux/concat/attention 参数没有 finite nonzero gradient；batch8 GPU smoke OOM 或 peak allocated > `72 GiB`；部署 final logit 与剥离辅助模块后不等价；checkpoint reload 不精确。
- 机制停止：任一保存 epoch 的平均 `softplus(alpha)` 低于初始 `softplus(0)` 的 25%，即 `<0.173286795`，判定原始 LoMix 权重目标塌缩并停止，不改成 Softmax/归一化/正则补救。

更精确的实现冻结和四个matched-control checkpoint哈希见 `AMENDMENT_002_PRE_EXECUTION.json`；它在任何合成测试、GPU smoke、训练或validation前写入。

## 7. 成本与风险

- 训练时间：matched control 32 epoch 实测 `661.0s`；预估 LoMix `15–35min`，以 batch8 smoke 的 step time 外推并在 formal report 中记录。
- 峰值显存：control 约 `20.07 GiB`；论文只给 96³/9-class 的 `<5GB` 额外量，当前 128³/1-class 预估总计 `25–45 GiB`，以 smoke 实测；硬停止线 `72 GiB`。
- 磁盘：不建新 cache；四个 deploy/training checkpoint、日志与报告预估 `<0.25G`。开跑前现场根分区剩余约 `22G`。
- 推理 forwards/延迟：一个 plain locator，`1×`；最终 4.70M 参数、阈值和部署链不变。
- 最大技术风险：19 张 full-volume loss 的训练成本；raw-logit multiplication 的稳定性；论文 Softplus 目标只能直接推动 alpha 下降，其“效用搜索”解释可能不成立；论文三维证据仅为附录初步结果且无官方 3D 代码。

## 8. 最终结果（完成后填写）

- Validation：selected epoch200 Scan `0.732479468`、Subject `0.721607103`、Point `0.935185`。相对matched control为`-0.001126/-0.002065`，相对voxel-KD200为`-0.001905/-0.002827`；promotion失败。
- Test（如授权）：未授权、未访问。
- 置信区间/病例分布：相对KD200 Scan 95% CI `[-0.004242,+0.000247]`，Subject `[-0.005073,-0.000651]`；43/108 scans与20/50 subjects改善，65 scans与30 subjects变差。
- 是否支持原直觉：不支持当前三尺度、raw-logit、未归一化Softplus、epoch168后32轮continuation实现。
- 学到了什么：所有训练期模块有梯度且部署等价，但平均Softplus从`0.693147`整体降至`0.654070`、权重差异很小，最终head overlap反而低于control/KD；这是科研负结果，不是实现失败。
- 下一步：停止本方法且不调参；下一候选为2026 DCD detail distillation，需新预注册。
- 报告/归档/哈希：[RESULT.md](RESULT.md)、[RESULT.json](RESULT.json)；raw validation SHA `ff12f2c74a9c27be5cc9d6fbfe096d22c115fab443b003b70bab1196cb633e65`；selected checkpoint SHA `f9bfcca36a904341f4eeebf1c71f218b997ab3e753be4815daa2789b2c7245fd`。
