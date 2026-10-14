# Experiment idea: Full972梯度中心化平坦解优化

Status: planned and frozen before split generation, implementation, GPU model construction, training, or new evaluation access.

## 链路审视与启发

当前链路是：`raw NIfTI → subject级manifest/角色冻结 → N4 + image-only brain crop + true-128³ cache → Full972随机初始化 + 固定history/EMA teacher + DiceCE/voxel-KD → train-only inner选epoch → fresh-seed full972重训 → fixed validation108 gate → 仅全门通过且获授权才作reused test270报告`。

1. **表示与训练时机比堆算力更可靠。** Image-only brain crop在旧匹配协议约提高Scan `+0.054`；epoch99→168在test提高`+0.015384`；Full972从随机初始化参与相对matched Full756在validation提高Scan/Subject `+0.010943/+0.007982`。限制是Full972相对KD200的Subject CI下界`-0.004484`仍未过门，且test270已复用。缺少未揭示新cohort的独立确认。
2. **当前4.70M模型不是明显欠容量。** True-160³用约`1.95×`体素计算，test Scan反而从`0.741204`降到`0.737060`，Point从`0.929630`降到`0.874074`。Width24把参数从4.70M增到10.57M、final训练从3.56h增到4.86h，但相对matched width16 validation Scan/Subject下降`0.002872/0.003525`并少1 Point。限制是这只否定全图160³和当前width24，不否定针对明确错误的轻量机制；缺少跨fresh seed复现的结构收益。
3. **最直接未解瓶颈是优化轨迹/跨seed稳定性。** Width24在train-only inner相对width16曾提高Scan/Subject约`+0.010634/+0.012193`，fresh-seed final却变成`-0.002872/-0.003525`。Full972 seed32/40/41单模Scan为`0.740013/0.733408/0.736000`；三seed等权logit ensemble相对seed32提高`+0.005829/+0.004418`，但Subject CI下界`-0.005660`未过`-0.003`门。限制是3×ensemble点估计正向但未授权蒸馏；缺少部署1×、不挑seed的稳定训练证据。
4. **Point与Dice不是同一目标。** Hard mL1-ACE把Point从100/108提高到103/108，却让Scan/Subject下降`0.010544/0.011392`；coverage/replay相对control提高Dice `0.001481/0.001890`却少1 Point；DMRFB插值相对KD200提高Dice `0.000299/0.000500`也少1 Point。下一实验以Scan为primary，同时冻结Subject和Point安全门。
5. **Teacher有用，但继续增加监督花样已接近平台。** 固定history/EMA logit ensemble在test相对history提高`+0.003850/+0.004165`；voxel-KD相对matched continuation在validation只再加`+0.000779/+0.000762`。ReCo、LoMix、DCD、KAS都让新增目标学动却没有提高validation。限制是这些负结果只否定各自实现；但它们共同要求下一步先离开loss轴。

本实验采用第3条启发：不改数据、模型或loss，只测试更平坦的优化轨迹能否在两个独立seed阶段都改善泛化。

## 1. 现在的问题是什么？

Full972 random-init voxel-KD已经把fixed validation108提高到Scan/Subject/Point `0.740013/0.731266/103/108`，但不同seed单模Scan跨度达到`0.006605`（`0.733408–0.740013`）。Width24的inner胜利也在fresh seed上完全翻转。当前结果更像被优化轨迹和seed方差限制，而不是被分辨率或参数量限制。

## 2. 一句话直觉

普通训练只找“脚下最低点”；本方法会先轻推一下模型参数，再要求被推后仍答得好，尽量停在宽谷底，而不是只对某个seed友好的尖坑。

## 3. 为什么选择这个idea？

- 2025 GCSAM（Gradient-Centralized Sharpness-Aware Minimization，梯度中心化的锐度感知最小化）先把每个多维参数张量的梯度减去其非输出维均值，再沿该方向扰动参数；第二次反传才由原AdamW更新。论文主张同时降低局部尖锐度和梯度噪声。
- 论文在通用与医学影像分类上比较Adam、SAM及多个变体，报告GCSAM更稳定；官方代码公开且默认`rho=0.05`、non-adaptive。
- 论文与官方实现不是3D分割证据，本实验只能称论文公式的Full972 3D适配，不能称复现或保证有效。
- 160³、width24和DMRFB改变输入/容量/结构；LoMix、DCD、KAS、ReCo改变监督；VM Dice、CVaR改变loss聚合。它们都没有测试“同一Full972模型与loss，只改变参数更新是否寻找平坦解”。
- 原始SAM是2021方法；本实验选择2025 GCSAM，是因为当前瓶颈同时包含sharpness和跨seed梯度噪声，而不是为了年份堆模块。

Primary sources:

- 2025 paper: https://arxiv.org/abs/2501.11584
- 2025 journal record: https://doi.org/10.1109/ACCESS.2025.3624332
- Official implementation: https://github.com/mhassann22/GCSAM
- 2025 ICLR Unified-SAM theory used only as background, not as the implemented method: https://openreview.net/forum?id=8rvqpiTTFv

## 4. 具体怎么做？

同一增强后的单通道T1ce先送入冻结的history168与EMA168 teacher，raw logits固定按`0.75/0.25`混合。Student仍计算`Dice + 0.1×CE + 0.3×T²×voxel KL`，`T=2`。

- `adamw_control`：一次student前传/反传，梯度裁剪到5，再由历史AdamW更新。
- `gcsam_candidate`：第一次student前传/反传得到同一loss梯度。对所有`ndim>1`参数的梯度，保留第0维（输出通道），在其余维上减去均值；用中心化后全局L2范数归一化并乘固定`rho=0.05`，只扰动student参数。随后在**同一batch、同一增强、同一已冻结teacher logits**上第二次前传/反传；对第二次梯度执行与control相同的clip5，精确恢复原参数，再由同一AdamW更新。
- 一维bias/norm参数不做中心化，但参与第二次AdamW更新；不使用adaptive scaling。
- 部署只保留普通4.70M student，不保留扰动器或teacher。

论文公式为`epsilon_GC = rho * g_GC / ||g_GC||`。官方示例代码先计算原梯度范数、后中心化，与公式不完全一致。本实验在访问任何inner/validation前冻结采用论文公式，并用测试要求实际扰动全局范数为`rho`、step后参数精确恢复。不会在结果后比较两种实现。

## 5. 唯一变化与固定项

- 唯一变化：student optimizer update由历史AdamW改为`rho=0.05`、non-adaptive、paper-norm GCSAM包裹同一AdamW。
- 固定：Full972 972 scans / 444 subjects、原生WT体积采样权重、true-128³ cache、4,700,897参数width16模型、dropout0.1、history/EMA teacher与`0.75/0.25`公式、DiceCE、voxel-KD weight0.3、temperature2、batch8、每轮972 requested / 968 consumed / 121 steps、shift8、AdamW超参、late-cosine 200轮schedule、阈值0、无TTA/SAM/后处理、1×部署。
- 新建subject-disjoint train-only inner：固定split seed `20261001`，从972 train角色中选exact 108 scans作inner tune，剩余864作fit。两臂用同一fit/tune、seed `20261002`、初始state和batch index序列。
- Inner只由candidate在epoch101–200按`Scan → Subject → earlier epoch`选择；control只在同epoch比较。
- 若inner过门，full972两臂用fresh seed `20261003`从相同随机初始state重训到同一selected epoch；只有冻结checkpoint和收据后才允许一次fixed validation108。

## 6. 成功、失败和停止规则

- Primary：train-only inner与final fixed validation的Scan Dice；Subject Dice和更早epoch只作冻结tie-break/安全指标；Point为共同安全门。
- Inner promotion：candidate必须严格超过same-epoch control的Scan与Subject；Point最多低1/108；Scan/Subject 10,000次paired subject/bootstrap 95% CI下界均`>=-0.003`；全部工程门通过。失败则不跑final。
- Final promotion：candidate必须严格超过fresh-seed matched control和archived Full972 seed32 `0.740012535/0.731266217`的Scan/Subject，Point至少103/108；相对两者的Scan/Subject paired 95% CI下界均`>=-0.003`；全部工程门通过。
- Test：final全部门通过也只产生“test候选”；因test270已复用，必须另写pre-access receipt并得到用户明确授权才可访问。Outer selector与confirmation已回收为train角色，绝不访问或重新赋予评测身份。
- Engineering stop：hash/split/initial-state/batch-index不一致；first/second loss、gradient或参数非有限；teacher有梯度；中心化/扰动/restore合同失败；batch8 peak allocated超过24GiB；control inner/final超过6h、candidate inner超过9h或final超过8h；checkpoint reload不exact，立即停止。
- 失败后不调`rho`、centralization维度、adaptive flag、clip、epoch、LR、seed、threshold或postprocess。

## 7. 成本与风险

- 训练：inner两卡并行，control预计约4.5h、candidate约6–8h；若过门，final再约3.6h/5–7h。总墙钟预计11–15h。
- 峰值显存：control历史约20.29GiB；candidate预计约20.5–22GiB，硬门24GiB。
- 磁盘：复用约9.5GB train cache与现有validation cache；split/测试/6个左右checkpoint与报告预计<300MB，不建新cache。当前可用约15GB。
- 推理：4,700,897参数、1×forward；无新增部署延迟或复杂度。
- 最大技术风险：两次student反传增加训练时间，但平坦性可能只改善训练扰动鲁棒性而不提高Dice；论文主要是2D分类，不保证3D分割迁移。

## 8. 最终结果（完成后填写）

- Train-only inner：待运行。
- Final validation：仅inner全部过门后允许。
- Test：仅final全部门通过、冻结收据且用户明确授权后允许。
- 置信区间/病例分布：待运行。
- 是否支持原直觉：待运行。
- 学到了什么：待运行。
- 下一步：待运行。
- 报告/归档/哈希：待运行。
